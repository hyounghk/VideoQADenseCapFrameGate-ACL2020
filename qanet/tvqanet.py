import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from context_query_attention import StructuredAttention_bi, StructuredAttention_frame
from encoder import StackedEncoder
from self_attention import MultiHeadedAttention
from torch.nn.utils.weight_norm import weight_norm


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)

def mask_logits_sum(target, mask):
    return target * mask + (1 - mask) * 0

class LinearWrapper(nn.Module):

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearWrapper, self).__init__()
        self.relu = relu
        layers = [nn.LayerNorm(in_hsz)] if layer_norm else []
        layers += [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        if self.relu:
            return F.relu(self.conv(x), inplace=True) 
        else:
            return self.conv(x)  


class TVQANet(nn.Module):
    def __init__(self, opt):
        super(TVQANet, self).__init__()
        self.sub_flag = opt.sub_flag
        self.dense_flag = opt.dense_flag
        self.vfeat_flag = opt.vfeat_flag
        self.vfeat_size = opt.vfeat_size
        self.scale = opt.scale
        self.dropout = opt.dropout
        self.hsz = opt.hsz
        self.bsz = None
        self.num_a = 5
        self.flag_cnt = 3

        print("self.flag_cnt", self.flag_cnt)

        self.wd_size = 768
        self.bridge_hsz = 300

        self.bert_word_encoding_fc = nn.Sequential(
            nn.LayerNorm(self.wd_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.wd_size, self.bridge_hsz),
            nn.ReLU(True),
            nn.LayerNorm(self.bridge_hsz),
        )

        if self.sub_flag:
            print("Activate sub branch")

        if self.vfeat_flag:
            print("Activate vid branch")
            self.vid_fc = nn.Sequential(
                nn.LayerNorm(self.vfeat_size),
                nn.Dropout(self.dropout),
                nn.Linear(self.vfeat_size, self.bridge_hsz),
                nn.ReLU(True),
                nn.LayerNorm(self.bridge_hsz)
            )

        if self.flag_cnt == 3:
            self.concat_fc = nn.Sequential(
                nn.LayerNorm(4 * self.hsz),
                nn.Dropout(self.dropout),
                nn.Linear(4 * self.hsz, self.hsz),
                nn.ReLU(True),
                nn.LayerNorm(self.hsz),
            )

            self.concat_fusion = nn.Sequential(
                nn.LayerNorm(4 * self.hsz),
                nn.Dropout(self.dropout),
                nn.Linear(4 * self.hsz, self.hsz),
                nn.ReLU(True),
                nn.LayerNorm(self.hsz),
            )

        self.input_embedding = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.bridge_hsz, self.hsz),
            nn.ReLU(True),
            nn.LayerNorm(self.hsz),
        )

        self.input_encoder = StackedEncoder(n_blocks=opt.input_encoder_n_blocks,
                                            n_conv=opt.input_encoder_n_conv,
                                            kernel_size=opt.input_encoder_kernel_size,
                                            num_heads=opt.input_encoder_n_heads,
                                            hidden_size=self.hsz,
                                            dropout=self.dropout)


        self.str_attn_bi = StructuredAttention_bi(dropout=self.dropout,
                                            scale=opt.scale)  

        self.str_attn_frame = StructuredAttention_frame(dropout=self.dropout,
                                            scale=opt.scale) 

        self.c2q_down_projection = nn.Sequential(
            nn.LayerNorm(3 * self.hsz),
            nn.Dropout(self.dropout),
            nn.Linear(3*self.hsz, self.hsz),
            nn.ReLU(True),
        )

        self.cls_encoder = StackedEncoder(n_blocks=opt.cls_encoder_n_blocks,
                                          n_conv=opt.cls_encoder_n_conv,
                                          kernel_size=opt.cls_encoder_kernel_size,
                                          num_heads=opt.cls_encoder_n_heads,
                                          hidden_size=self.hsz,
                                          dropout=self.dropout)

        self.classifier = LinearWrapper(in_hsz=self.hsz * 6,
                                        out_hsz=1,
                                        layer_norm=True,
                                        dropout=self.dropout,
                                        relu=False)



        self.MultiHeadAtt = MultiHeadedAttention(4, self.hsz)

        self.QuestionAtt = LinearWrapper(in_hsz=self.hsz * 2,
                                                out_hsz=1,
                                                layer_norm=True,
                                                dropout=self.dropout,
                                                relu=False)

        self.QuestionAtt_global = LinearWrapper(in_hsz=self.hsz * 2,
                                                out_hsz=1,
                                                layer_norm=True,
                                                dropout=self.dropout,
                                                relu=False)

    def load_word_embedding(self, pretrained_embedding, requires_grad=False):
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embedding.weight.requires_grad = requires_grad


    def forward(self, batch):

        self.bsz = batch["qas_mask"].size(0)

        bsz = self.bsz
        num_a = self.num_a
        hsz = self.hsz


        a_embed = self.base_encoder(batch["qas_bert"].view(bsz*num_a, -1, self.wd_size), 
                                    batch["qas_mask"].view(bsz * num_a, -1),  
                                    self.bert_word_encoding_fc,
                                    self.input_embedding,
                                    self.input_encoder)  
        a_embed = a_embed.view(bsz, num_a, 1, -1, hsz) 
        a_mask = batch["qas_mask"].view(bsz, num_a, 1, -1) 

        attended_sub, attended_vid, attended_vid_mask, attended_sub_mask = (None, ) * 4


        if self.dense_flag:

            num_imgs, num_dense_words = batch["dense_cap_mask"].shape[1:3]
            dense_cap_embed = self.base_encoder(batch["dense_bert"].view(bsz*num_imgs, num_dense_words, -1), 
                                          batch["dense_cap_mask"].view(bsz*num_imgs, num_dense_words), 
                                          self.bert_word_encoding_fc,
                                          self.input_embedding,
                                          self.input_encoder)  

            dense_cap_embed = dense_cap_embed.contiguous().view(bsz, 1, num_imgs, num_dense_words, -1)  
            dense_cap_mask = batch["dense_cap_mask"].view(bsz, 1, num_imgs, num_dense_words)  

            attended_dense_cap_q, attended_dense_cap_d, attended_dense_cap_mask_q, attended_dense_cap_mask_d = \
                self.qa_ctx_attention_bi(a_embed, dense_cap_embed, a_mask, dense_cap_mask)


        if self.sub_flag:
            num_imgs, num_words = batch["sub_mask"].shape[1:3]
            sub_embed = self.base_encoder(batch["sub_bert"].view(bsz*num_imgs, num_words, -1),  
                                          batch["sub_mask"].view(bsz * num_imgs, num_words),  
                                          self.bert_word_encoding_fc,
                                          self.input_embedding,
                                          self.input_encoder)  

            sub_embed = sub_embed.contiguous().view(bsz, 1, num_imgs, num_words, -1)  
            sub_mask = batch["sub_mask"].view(bsz, 1, num_imgs, num_words)  

            attended_sub_q, attended_sub_s, attended_sub_mask_q, attended_sub_mask_s = \
                self.qa_ctx_attention_bi(a_embed, sub_embed, a_mask, sub_mask)


        if self.vfeat_flag:
            num_imgs, num_regions = batch["vid"].shape[1:3]
            vid_embed = self.get_visual_embedding(batch["vid"])  

            vid_embed_orig = self.base_encoder(vid_embed.view(bsz*num_imgs, num_regions, -1),  
                                          batch["vid_mask"].view(bsz * num_imgs, num_regions),  
                                          self.vid_fc,
                                          self.input_embedding,
                                          self.input_encoder) 

            vid_embed = vid_embed_orig.contiguous().view(bsz, 1, num_imgs, num_regions, -1)  
            vid_mask = batch["vid_mask"].view(bsz, 1, num_imgs, num_regions)  

            noun_mask = None

            attended_vid_q, attended_vid_v, attended_vid_mask_q, attended_vid_mask_v = \
                self.qa_ctx_attention_bi(a_embed, vid_embed, a_mask, vid_mask)



        if self.flag_cnt == 3:

            attended_dense_cap, attended_dense_cap_mask = self.max_and_fusion(attended_dense_cap_q, attended_dense_cap_d, attended_dense_cap_mask_q, attended_dense_cap_mask_d)
            attended_sub, attended_sub_mask = self.max_and_fusion(attended_sub_q, attended_sub_s, attended_sub_mask_q, attended_sub_mask_s)
            attended_vid, attended_vid_mask = self.max_and_fusion(attended_vid_q, attended_vid_v, attended_vid_mask_q, attended_vid_mask_v)

            attended_sub_vid_frame, attended_vid_frame, attended_sub_vid_mask_frame, attended_vid_mask_frame = \
                self.frame_wise_attention_wo_max(attended_sub, attended_vid, attended_sub_mask, attended_vid_mask)


            visual_text_embedding_sub_vid = torch.cat([attended_sub_vid_frame,
                                               attended_vid_frame,
                                               attended_sub_vid_frame * attended_vid_frame,
                                               attended_sub_vid_frame + attended_vid_frame], dim=-1) 
            visual_text_embedding_sub_vid = self.concat_fc(visual_text_embedding_sub_vid) 


            attended_sub_cap_frame, attended_cap_frame, attended_sub_cap_mask_frame, attended_dense_cap_mask_frame = \
                self.frame_wise_attention_wo_max(attended_sub, attended_dense_cap, attended_sub_mask, attended_dense_cap_mask)


            cap_text_embedding_sub_cap = torch.cat([attended_sub_cap_frame,
                                               attended_cap_frame,
                                               attended_sub_cap_frame * attended_cap_frame,
                                               attended_sub_cap_frame + attended_cap_frame], dim=-1)  
            cap_text_embedding_sub_cap = self.concat_fc(cap_text_embedding_sub_cap)  



            visual_text_embedding_sub_vid_new, cap_text_embedding_sub_cap_new, _, _ = \
                self.frame_wise_attention_wo_max_multi_head(visual_text_embedding_sub_vid, cap_text_embedding_sub_cap, attended_sub_mask, attended_dense_cap_mask)

            visual_cap_text_embedding = visual_text_embedding_sub_vid_new + cap_text_embedding_sub_cap_new


            out_vid, max_statement_sm_sigmoid = self.classfier_head_multi_proposal(
                visual_cap_text_embedding, attended_sub_vid_mask_frame)

        return out_vid, max_statement_sm_sigmoid

    @classmethod
    def base_encoder(cls, data, data_mask, init_encoder, downsize_encoder, input_encoder):

        data = downsize_encoder(init_encoder(data))
        return input_encoder(data, data_mask)

    def get_visual_embedding(self, vid):
        vid = F.normalize(vid, p=2, dim=-1)
        return vid

    def qa_ctx_attention_bi(self, qa_embed, ctx_embed, qa_mask, ctx_mask):

        num_img, num_region = ctx_mask.shape[2:]

        u_a, u_b, s_mask, s_mask_b = self.str_attn_bi(
            qa_embed, ctx_embed, qa_mask, ctx_mask)  
        qa_embed = qa_embed.repeat(1, 1, num_img, 1, 1)
        ctx_embed = ctx_embed.repeat(1, 5, 1, 1, 1)
        mixed = torch.cat([qa_embed,
                           u_a,
                           qa_embed*u_a], dim=-1)  
        mixed = self.c2q_down_projection(mixed)  

        mixed_b = torch.cat([ctx_embed,
                           u_b,
                           ctx_embed*u_b], dim=-1) 
        mixed_b = self.c2q_down_projection(mixed_b)  

        mixed_mask = (s_mask.sum(-1) != 0).float() 
        mixed_mask_b = (s_mask_b.sum(-1) != 0).float()  
        return mixed, mixed_b, mixed_mask, mixed_mask_b


    def max_and_fusion(self, attended_vid_cap_sub, attended_cap_vid_sub, attended_vid_mask, attended_dense_cap_mask):
        bsz, num_a, num_imgs, num_q = attended_vid_mask.size()
        bsz, num_a, num_imgs, num_c = attended_dense_cap_mask.size()

        attended_vid_cap_sub = attended_vid_cap_sub.view(bsz*num_a*num_imgs, num_q, -1)  
        attended_vid_mask_frame = attended_vid_mask.view(bsz*num_a*num_imgs, num_q)  

        attended_vid_cap_sub_frame= torch.max(mask_logits(attended_vid_cap_sub, attended_vid_mask_frame.unsqueeze(2)), 1)[0]  
        attended_vid_cap_sub_frame = attended_vid_cap_sub_frame.view(bsz, num_a, num_imgs, -1)  
        attended_vid_mask_frame = (attended_vid_mask_frame.sum(1) != 0).float().view(bsz, num_a, num_imgs)  

        attended_cap_vid_sub = attended_cap_vid_sub.view(bsz*num_a*num_imgs, num_c, -1)  
        attended_dense_cap_mask_frame = attended_dense_cap_mask.view(bsz*num_a*num_imgs, num_c)  

        attended_cap_vid_sub_frame = torch.max(mask_logits(attended_cap_vid_sub, attended_dense_cap_mask_frame.unsqueeze(2)), 1)[0]  
        attended_cap_vid_sub_frame = attended_cap_vid_sub_frame.view(bsz, num_a, num_imgs, -1) 
        attended_dense_cap_mask_frame = (attended_dense_cap_mask_frame.sum(1) != 0).float().view(bsz, num_a, num_imgs)  

        attended_vid_cap_sub_frame = mask_logits_sum(attended_vid_cap_sub_frame, attended_vid_mask_frame.unsqueeze(-1))
        attended_cap_vid_sub_frame = mask_logits_sum(attended_cap_vid_sub_frame, attended_dense_cap_mask_frame.unsqueeze(-1))
        
        joint_ = torch.cat([attended_vid_cap_sub_frame, attended_cap_vid_sub_frame,
                             attended_vid_cap_sub_frame * attended_cap_vid_sub_frame,
                             attended_vid_cap_sub_frame + attended_cap_vid_sub_frame], dim=-1)
        joint_ = self.concat_fusion(joint_)

        return joint_, attended_vid_mask_frame

    def frameQuestionAsnwerAtt_gate_multi_res(self, statement, statement_mask):
        bsz, num_a, num_img, d = statement.size()

        statement_att = statement.view(bsz*num_a, num_img, -1)  
        statement_att_mask = statement_mask.view(bsz*num_a, num_img) 
        statement_att = self.MultiHeadAtt(statement_att, statement_att_mask) 

        statement_att_1 = statement_att[:,:num_img/2,:].view(bsz, num_a, num_img/2, -1)
        statement_att_2 = statement_att[:,num_img/2:,:].view(bsz, num_a, num_img/2, -1)

        return statement_att_1, statement_att_2 

    def frame_wise_attention_wo_max_multi_head(self, attended_vid_cap_sub_frame, attended_cap_vid_sub_frame, attended_vid_mask_frame, attended_dense_cap_mask_frame):

        gate_input = torch.cat([attended_vid_cap_sub_frame, attended_cap_vid_sub_frame], dim=2)
        gate_input_mask = torch.cat([attended_vid_mask_frame, attended_dense_cap_mask_frame], dim=2)
        multi_att_1, multi_att_2 = self.frameQuestionAsnwerAtt_gate_multi_res(gate_input, gate_input_mask)

        mixed_1 = torch.cat([attended_vid_cap_sub_frame, multi_att_1, attended_vid_cap_sub_frame * multi_att_1], dim=-1)
        mixed_1 = self.c2q_down_projection(mixed_1)

        mixed_2 = torch.cat([attended_cap_vid_sub_frame, multi_att_2, attended_cap_vid_sub_frame * multi_att_2], dim=-1)
        mixed_2 = self.c2q_down_projection(mixed_2)

        attended_vid_cap_sub_frame_att = mixed_1 + mask_logits_sum(attended_vid_cap_sub_frame, attended_vid_mask_frame.unsqueeze(-1))
        attended_cap_vid_sub_frame_att = mixed_2 + mask_logits_sum(attended_cap_vid_sub_frame, attended_dense_cap_mask_frame.unsqueeze(-1))

        return attended_vid_cap_sub_frame_att, attended_cap_vid_sub_frame_att, attended_vid_mask_frame, attended_dense_cap_mask_frame


    def frame_wise_attention_wo_max(self, attended_vid_cap_sub_frame, attended_cap_vid_sub_frame, attended_vid_mask_frame, attended_dense_cap_mask_frame):

        attended_vid_cap_sub_frame_att, _, _, _ = \
        self.frame_attention(attended_vid_cap_sub_frame, attended_cap_vid_sub_frame, attended_vid_mask_frame, attended_dense_cap_mask_frame,
            noun_mask=None,
            non_visual_vectors=None)

        attended_vid_cap_sub_frame_att = attended_vid_cap_sub_frame_att + mask_logits_sum(attended_vid_cap_sub_frame, attended_vid_mask_frame.unsqueeze(-1))

        attended_cap_vid_sub_frame_att, _, _, _ = \
        self.frame_attention(attended_cap_vid_sub_frame, attended_vid_cap_sub_frame, attended_dense_cap_mask_frame, attended_vid_mask_frame,
            noun_mask=None,
            non_visual_vectors=None)

        attended_cap_vid_sub_frame_att = attended_cap_vid_sub_frame_att + mask_logits_sum(attended_cap_vid_sub_frame, attended_dense_cap_mask_frame.unsqueeze(-1))

        return attended_vid_cap_sub_frame_att, attended_cap_vid_sub_frame_att, attended_vid_mask_frame, attended_dense_cap_mask_frame


    def frame_attention(self, qa_embed, ctx_embed, qa_mask, ctx_mask, noun_mask, non_visual_vectors):

        num_img = ctx_mask.shape[2]

        u_a, raw_s, s_mask, s_normalized = self.str_attn_frame(
            qa_embed, ctx_embed, qa_mask, ctx_mask)  

        mixed = torch.cat([qa_embed,
                           u_a,
                           qa_embed*u_a], dim=-1) 
        mixed = self.c2q_down_projection(mixed)  
        mixed_mask = (s_mask.sum(-1) != 0).float()  
        return mixed, mixed_mask, raw_s, s_normalized

    def frameQuestionAsnwerAtt_sigmoid(self, statement, statement_mask):
        statement_att = self.QuestionAtt(statement) 
        statement_att_masked = mask_logits(statement_att, statement_mask.unsqueeze(-1)) 
        statement_att_masked_sfm = torch.sigmoid(statement_att_masked) 
        statement_attened = (statement * statement_att_masked_sfm).sum(1)
        return statement_attened, (statement_att_masked, statement_att_masked_sfm) 

    def frameQuestionAsnwerAtt_sigmoid_global(self, statement, statement_mask):
        statement_att = self.QuestionAtt_global(statement) 
        statement_att_masked = mask_logits(statement_att, statement_mask.unsqueeze(-1)) 
        statement_att_masked_sfm = torch.sigmoid(statement_att_masked) 
        statement_attened = (statement * statement_att_masked_sfm).sum(1)
        return statement_attened 

    

    def get_proposals(self, max_statement, max_statement_mask):

        bsz, num_a, num_img, _ = max_statement_mask.shape

        max_statement_sm = max_statement.view(bsz*num_a, num_img, -1)  
        max_statement_mask_sm = max_statement_mask.view(bsz*num_a, num_img)  

        max_statement_local, max_statement_sm_sigmoid = self.frameQuestionAsnwerAtt_sigmoid(max_statement_sm, max_statement_mask_sm) 
        max_statement_global = self.frameQuestionAsnwerAtt_sigmoid_global(max_statement_sm, max_statement_mask_sm) 

        max_statement_local = max_statement_local.view(bsz, num_a, -1) 
        max_statement_global = max_statement_global.view(bsz, num_a, -1)  

        bsz, num_a, num_img, _ = max_statement_mask.shape

        cur_global_max_max_statement = torch.max(mask_logits(max_statement, max_statement_mask), 2)[0]

        cur_global_max_max_statement = torch.cat([cur_global_max_max_statement, max_statement_global, max_statement_local], dim=-1) 
        
        return cur_global_max_max_statement, max_statement_sm_sigmoid


    def classfier_head_multi_proposal(self, statement, statement_mask):

        bsz, num_a, num_img = statement_mask.shape
        statement = statement.view(bsz*num_a, num_img, -1)  
        statement_mask = statement_mask.view(bsz*num_a, num_img)  
        statement_1 = self.cls_encoder(statement, statement_mask)  
        statement = self.cls_encoder(statement_1, statement_mask) 
        statement = torch.cat([statement, statement_1], dim=-1)

        max_statement_mask = statement_mask.view(bsz, num_a, num_img, 1)  

        stacked_max_statement = statement.view(bsz, num_a, num_img, -1)  
        max_max_statement, max_statement_sm_sigmoid = self.get_proposals(stacked_max_statement, max_statement_mask)  

        answer_scores = self.classifier(max_max_statement).squeeze(2) 
        return answer_scores, max_statement_sm_sigmoid  

