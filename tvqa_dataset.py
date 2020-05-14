from __future__ import absolute_import, division, print_function

import os
import sys
import h5py
import pickle
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import json
import time


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def files_exist(filepath_list):
    for ele in filepath_list:
        if not os.path.exists(ele):
            return False
    return True


def flat_list_of_lists(l):
    return [item for sublist in l for item in sublist]



def load_glove(filename):

    glove = {}
    with open(filename) as f:
        for line in f.readlines():
            values = line.strip("\n").split(" ")
            word = values[0]
            vector = np.asarray([float(e) for e in values[1:]])
            glove[word] = vector
    return glove




class TVQADataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.is_eval = mode != "train"  
        self.raw_train = load_json(opt.train_path)
        self.raw_test = load_json(opt.test_path)
        self.raw_valid = load_json(opt.valid_path)
        self.indices_train = load_json("./path/indicesDicttrain.json")
        self.indices_valid = load_json("./path/indicesDictvalid.json")
        self.indices_test = load_json("./path/indicesDicttest.json")
        self.sub_flag = "sub" in opt.input_streams
        self.dense_flag = "dense" in opt.input_streams
        self.vfeat_flag = "vfeat" in opt.input_streams
      

        if self.dense_flag:

            start_time = time.time()
            print("laoding densecap_dict...")
            self.dense_cap = load_json('./path/densecap_dict_new.json')
            print(time.time() - start_time)

        self.glove_embedding_path = opt.glove_path
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()
        self.cur_indices_dict = self.get_cur_indices()

        self.frm_cnt_dict = load_json("./path/frm_cnt_cache.json") 

        self.word2idx_path = "./cache/word2idx.pickle"
        self.idx2word_path = "./cache/idx2word.pickle"
        self.vocab_embedding_path = "./cache/vocab_embedding.pickle"
        self.embedding_dim = 300
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        self.offset = len(self.word2idx)
        text_keys = ["a0", "a1", "a2", "a3", "a4", "q", "sub_text"]
        if not files_exist([self.word2idx_path, self.idx2word_path, self.vocab_embedding_path]):
            print("\nNo cache founded.")
            self.build_word_vocabulary(text_keys, word_count_threshold=2)
        else:
            print("\nLoading cache ...")
            self.word2idx = load_pickle(self.word2idx_path)
            self.idx2word = load_pickle(self.idx2word_path)
            self.vocab_embedding = load_pickle(self.vocab_embedding_path)

        self.indicesDict_train = {}
        self.indicesDict_val = {}
        self.vFeatCheck = {}

    def set_mode(self, mode):
        self.mode = mode
        self.is_eval = mode != "train"
        self.cur_data_dict = self.get_cur_dict()
        self.cur_indices_dict = self.get_cur_indices()

    def get_cur_dict(self):
        if self.mode == 'train':
            return self.raw_train
        elif self.mode == 'valid':
            return self.raw_valid
        elif self.mode == 'test':
            return self.raw_test

    def get_cur_indices(self):
        if self.mode == 'train':
            return self.indices_train
        elif self.mode == 'valid':
            return self.indices_valid
        elif self.mode == 'test':
            return self.indices_test


    def __len__(self):
        return len(self.cur_data_dict)

    def __getitem__(self, index):

        items = {}
        items["vid_name"] = self.cur_data_dict[index]["vid_name"]
        vid_name = items["vid_name"] 
        items["qid"] = self.cur_data_dict[index]["qid"]
        qid = items["qid"]  
        frm_cnt = self.frm_cnt_dict[vid_name]
 
        located_img_ids = [1, 7]
        start_img_id, end_img_id = 1, 7

        indices = self.cur_indices_dict[vid_name]["indices"]

        indices = np.array(indices) - 1  

        items["ts_label"] = self.get_ts_label(self.cur_data_dict[index]["ts"][0],
                                              self.cur_data_dict[index]["ts"][1],
                                              frm_cnt,
                                              indices,
                                              fps=3)

        items["ts"] = self.cur_data_dict[index]["ts"]  
        items["image_indices"] = (indices + 1).tolist()  

        answer_keys = ["a0", "a1", "a2", "a3", "a4"]
        answer_nums = ["0", "1", "2", "3", "4"]
        qa_sentences = [self.numericalize(self.cur_data_dict[index]["q"]
                        + " " + self.cur_data_dict[index][k], eos=False) for k in answer_keys]

        qa_sentences_bert = [torch.from_numpy(
            np.load(os.path.join('./path/', 'qa_f_' + str(qid) + "_" + k + ".npy"), allow_pickle=True)[0]) for k in answer_nums]

        items["qas"] = qa_sentences
        items["qas_bert"] = qa_sentences_bert


        if self.dense_flag:

            captions = []
            for capidx in indices:
                assert self.dense_cap[vid_name][capidx]["frame_num"] == capidx + 1
                captions.append(" ".join(self.dense_cap[vid_name][capidx]["captions"]))

            items["dense_cap"] = [self.numericalize(e, eos=False) for e in captions]
            if self.mode == 'test':
                items["dense_bert"] = [torch.from_numpy(np.load(os.path.join('./path/', 'd_f_' + vid_name + "_" + str(n) + ".npy"), allow_pickle=True)[0]) for n in range(len(indices))]
            else:
                items["dense_bert"] = [torch.from_numpy(np.load(os.path.join('./path/', 'd_f_' + vid_name + "_" + str(n) + ".npy"), allow_pickle=True)[0]) for n in range(len(indices))]
                    


        else:
            items["dense_cap"] = [torch.zeros(2, 2)] * 2

            

        if self.sub_flag:

            sub_time = self.cur_data_dict[index]["sub_time"]
            img_aligned_sub_indices, raw_sub_n_tokens = self.get_aligned_sub_indices(
                indices + 1,
                self.cur_data_dict[index]["sub_text"],
                sub_time,
                mode="nearest")

            sub_roberta = [np.load(os.path.join('./path/', 's_f_' + vid_name + "_" + str(n) + ".npy"), allow_pickle=True) for n in range(len(raw_sub_n_tokens))]

            items["sub_bert"] = [torch.from_numpy(np.concatenate([sub_roberta[in_idx][0] for in_idx in e], axis=0))
                                 for e in img_aligned_sub_indices]


            aligned_sub_text = self.get_aligned_sub(self.cur_data_dict[index]["sub_text"],
                                                    img_aligned_sub_indices)
            items["sub"] = [self.numericalize(e, eos=False) for e in aligned_sub_text]


        else:
            items["sub_bert"] = [torch.zeros(2, 2)] * 2
            items["sub"] = [torch.zeros(2, 2)] * 2

        if self.mode == 'test':
            ca_idx = 0
        else:
            ca_idx = int(self.cur_data_dict[index]["answer_idx"])
        items["target"] = ca_idx

        if self.vfeat_flag:       
            if self.mode == 'test':
                cur_vfeat = np.load(os.path.join('./path/', vid_name + ".npy"), allow_pickle=True)
            else:
                cur_vfeat = np.load(os.path.join('./path/', vid_name + ".npy"), allow_pickle=True)

            items["vfeat"] = [torch.from_numpy(e) for e in cur_vfeat]
        else:
            items["vfeat"] = [torch.zeros(2, 2)] * 2

        return items

    @classmethod
    def get_ts_label(cls, st, ed, num_frame, indices, fps=3):

        max_num_frame = 300.
        if num_frame > max_num_frame:
            st, ed = [(max_num_frame / num_frame) * fps * ele for ele in [st, ed]]
        else:
            st, ed = [fps * ele for ele in [st, ed]]

        start_idx = np.searchsorted(indices, st, side="left")
        end_idx = np.searchsorted(indices, ed, side="right")
        max_len = len(indices)
        if not start_idx < max_len:
            start_idx -= 1
        if not end_idx < max_len:
            end_idx -= 1

        if start_idx == end_idx:
            st_ed = [start_idx, end_idx]
        else:
            st_ed = [start_idx, end_idx-1] 

        return st_ed  
       

    @classmethod
    def line_to_words(cls, line, eos=True, downcase=True):
        eos_word = "<eos>"
        words = line.lower().split() if downcase else line.split()
        words = [w for w in words]
        words = words + [eos_word] if eos else words
        return words

    @classmethod
    def find_match(cls, subtime, value, mode="larger", span=1.5):

        if mode == "nearest":  
            return sorted((np.abs(subtime - value)).argsort()[:2].tolist())
        elif mode == "span":  
            return_indices = np.nonzero(np.abs(subtime - value) < span)[0].tolist()
            if value <= 2:
                return_indices = np.nonzero(subtime - 2 <= 0)[0].tolist() + return_indices
            return return_indices
        elif mode == "larger":
            idx = max(0, np.searchsorted(subtime, value, side="left") - 1)
            return_indices = [idx - 1, idx, idx + 1]
            return_indices = [idx for idx in return_indices if 0 <= idx < len(subtime)]
            return return_indices

    @classmethod
    def get_aligned_sub_indices(cls, img_ids, subtext, subtime, fps=3, mode="larger"):

        subtext = subtext.lower().split(" <eos> ")  
        raw_sub_n_tokens = [len(s.split()) for s in subtext]

        assert len(subtime) == len(subtext)
        img_timestamps = np.array(img_ids) / fps  
        img_aligned_sentence_indices = []  
        for t in img_timestamps:
            img_aligned_sentence_indices.append(cls.find_match(subtime, t, mode=mode))
        return img_aligned_sentence_indices, raw_sub_n_tokens

    @classmethod
    def get_aligned_sub(cls, subtext, img_aligned_sentence_indices):
        subtext = subtext.lower().split(" <eos> ")  
        return [" ".join([subtext[inner_idx] for inner_idx in e]) for e in img_aligned_sentence_indices]

    def numericalize(self, sentence, eos=True, match=False):

        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in self.line_to_words(sentence, eos=eos)] 
        return sentence_indices

    def build_word_vocabulary(self, text_keys, word_count_threshold=0):

        print("Building word vocabulary starts.\n")
        all_sentences = []
        for k in text_keys:

            all_sentences.extend([ele[k] for ele in self.raw_train])

        word_counts = {}
        for sentence in all_sentences:
            for w in self.line_to_words(sentence, eos=False, downcase=True):
                word_counts[w] = word_counts.get(w, 0) + 1


        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in self.word2idx.keys()]
        print("Vocabulary Size %d (<pad> <unk> <eos> excluded) using word_count_threshold %d.\n" %
              (len(vocab), word_count_threshold))


        for idx, w in enumerate(vocab):
            self.word2idx[w] = idx + self.offset
            self.idx2word[idx + self.offset] = w

        print("word2idx size: %d, idx2word size: %d.\n" % (len(self.word2idx), len(self.idx2word)))

        print("Loading glove embedding at path : %s.\n" % self.glove_embedding_path)
        glove_full = load_glove(self.glove_embedding_path)
        print("Glove Loaded, building word2idx, idx2word mapping.\n")

        glove_matrix = np.zeros([len(self.idx2word), self.embedding_dim])
        glove_keys = glove_full.keys()
        for i in tqdm(range(len(self.idx2word))):
            w = self.idx2word[i]
            w_embed = glove_full[w] if w in glove_keys else np.random.randn(self.embedding_dim) * 0.4
            glove_matrix[i, :] = w_embed
        self.vocab_embedding = glove_matrix
        print("vocab embedding size is :", glove_matrix.shape)

        print("Saving cache files at ./cache.\n")
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        pickle.dump(self.word2idx, open(self.word2idx_path, 'w'))
        pickle.dump(self.idx2word, open(self.idx2word_path, 'w'))
        pickle.dump(glove_matrix, open(self.vocab_embedding_path, 'w'))

        print("Building  vocabulary done.\n")


def pad_sequences_2d(sequences, dtype=torch.long, fromwhere=None):

    bsz = len(sequences)
    para_lengths = [len(seq) for seq in sequences]
    max_para_len = max(para_lengths)
    sen_lengths = [[len(word_seq) for word_seq in seq] for seq in sequences]
    max_sen_len = max(flat_list_of_lists(sen_lengths))

    if isinstance(sequences[0], torch.Tensor):
        extra_dims = sequences[0].shape[2:]
    elif isinstance(sequences[0][0], torch.Tensor):
        extra_dims = sequences[0][0].shape[1:]
    else:
        sequences = [[torch.LongTensor(word_seq) for word_seq in seq] for seq in sequences]
        extra_dims = ()

    padded_seqs = torch.zeros((bsz, max_para_len, max_sen_len) + extra_dims, dtype=dtype)
    mask = torch.zeros(bsz, max_para_len, max_sen_len).float()
    sen_lengths_tensor = torch.zeros(bsz, max_para_len).long()

    for b_i in range(bsz):
        for sen_i, sen_l in enumerate(sen_lengths[b_i]):
            padded_seqs[b_i, sen_i, :sen_l] = sequences[b_i][sen_i]
            mask[b_i, sen_i, :sen_l] = 1
            sen_lengths_tensor[b_i, sen_i] = sen_l
    if fromwhere is None:
        return padded_seqs, mask , None
    else:
        return padded_seqs, mask, sen_lengths_tensor



def pad_sequences_1d(sequences, dtype=torch.long, fromwhere=None):

    if isinstance(sequences[0], list):
        sequences = [torch.tensor(s, dtype=dtype) for s in sequences]
    extra_dims = sequences[0].shape[1:] 
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = torch.zeros(len(sequences), max(lengths)).float()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    if fromwhere is None:
        return padded_seqs, mask , None
    else:
        return padded_seqs, mask , torch.LongTensor(lengths)


def make_mask_from_length(lengths):
    mask = torch.zeros(len(lengths), max(lengths)).float()
    for idx, l in enumerate(lengths):
        mask[idx, :l] = 1
    return mask


def pad_collate(data):

    batch = {}
    batch["qas"], batch["qas_mask"], _ = pad_sequences_2d([d["qas"] for d in data], dtype=torch.long)
    batch["qas_bert"], _, _ = pad_sequences_2d([d["qas_bert"] for d in data], dtype=torch.float)
    batch["sub"], batch["sub_mask"], _  = pad_sequences_2d([d["sub"] for d in data], dtype=torch.long)
    batch["dense_cap"], batch["dense_cap_mask"], batch["dense_cap_l"] = pad_sequences_2d([d["dense_cap"] for d in data], dtype=torch.long, fromwhere="dense")
    batch["sub_bert"], batch["sub_mask"], _ = pad_sequences_2d([d["sub_bert"] for d in data], dtype=torch.float)
    batch["dense_bert"], batch["dense_cap_mask"], batch["dense_cap_l"] = pad_sequences_2d([d["dense_bert"] for d in data], dtype=torch.float, fromwhere="dense")
    batch["vid_name"] = [d["vid_name"] for d in data]
    batch["qid"] = [d["qid"] for d in data]
    batch["target"] = torch.tensor([d["target"] for d in data], dtype=torch.long)
    batch["vid"], batch["vid_mask"], __main__ = pad_sequences_2d([d["vfeat"] for d in data], dtype=torch.float)

    if data[0]["ts_label"] is None:
        batch["ts_label"] = None
    elif isinstance(data[0]["ts_label"], list):  
        batch["ts_label"] = dict(
            st=torch.LongTensor([d["ts_label"][0] for d in data]),
            ed=torch.LongTensor([d["ts_label"][1] for d in data]),
        )
        batch["ts_label_mask"] = make_mask_from_length([len(d["image_indices"]) for d in data])
    elif isinstance(data[0]["ts_label"], torch.Tensor):  
        batch["ts_label"], batch["ts_label_mask"], _ = pad_sequences_1d([d["ts_label"] for d in data], dtype=torch.float)
    else:
        raise NotImplementedError

    batch["ts"] = [d["ts"] for d in data]
    batch["image_indices"] = [d["image_indices"] for d in data]

    return batch


def prepare_inputs(batch, max_len_dict=None, device="cuda"):

    model_in_dict = {}
    max_qa_l = min(batch["qas"].shape[2], max_len_dict["max_qa_l"])
    model_in_dict["qas"] = batch["qas"][:, :, :max_qa_l].to(device)
    model_in_dict["qas_bert"] = batch["qas_bert"][:, :, :max_qa_l].to(device)
    model_in_dict["qas_mask"] = batch["qas_mask"][:, :, :max_qa_l].to(device)

    model_in_dict["sub"] = batch["sub"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_sub_l"]].to(device)
    model_in_dict["dense_cap"] = batch["dense_cap"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_dc_l"]].to(device)
    model_in_dict["dense_bert"] = batch["dense_bert"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_dc_l"]].to(device)
    model_in_dict["sub_bert"] = batch["sub_bert"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_sub_l"]].to(device)
    model_in_dict["sub_mask"] = batch["sub_mask"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_sub_l"]].to(device)
    model_in_dict["dense_cap_mask"] = batch["dense_cap_mask"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_dc_l"]].to(device)
    max_dc_l = min(batch["dense_cap"].shape[2], max_len_dict["max_dc_l"])
    model_in_dict["dense_cap_l"] = batch["dense_cap_l"].clamp(min=1, max=max_dc_l)

    ctx_keys = ["vid"]
    for k in ctx_keys:
        max_l = min(batch[k].shape[1], max_len_dict["max_{}_l".format(k)])
        model_in_dict[k] = batch[k][:, :max_l].to(device)
        mask_key = "{}_mask".format(k)
        model_in_dict[mask_key] = batch[mask_key][:, :max_l].to(device)


    if batch["ts_label"] is None:
        model_in_dict["ts_label"] = None
        model_in_dict["ts_label_mask"] = None
    elif isinstance(batch["ts_label"], dict):  
        model_in_dict["ts_label"] = dict(
            st=batch["ts_label"]["st"].to(device),
            ed=batch["ts_label"]["ed"].to(device),
        )
        model_in_dict["ts_label_mask"] = batch["ts_label_mask"][:, :max_len_dict["max_vid_l"]].to(device)
    else:
        model_in_dict["ts_label"] = batch["ts_label"][:, :max_len_dict["max_vid_l"]].to(device)
        model_in_dict["ts_label_mask"] = batch["ts_label_mask"][:, :max_len_dict["max_vid_l"]].to(device)


    model_in_dict["target"] = batch["target"].to(device)

    model_in_dict["qid"] = batch["qid"]
    model_in_dict["vid_name"] = batch["vid_name"]

    targets = model_in_dict["target"]
    qids = model_in_dict["qid"]
    model_in_dict["ts"] = batch["ts"]

    model_in_dict["image_indices"] = batch["image_indices"]
    return model_in_dict, targets, qids


