import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class StructuredAttention_bi(nn.Module):

    def __init__(self, dropout=0.1, scale=100):

        super(StructuredAttention_bi, self).__init__()
        self.dropout = dropout
        self.scale = scale

    def forward(self, C, Q, c_mask, q_mask):

        bsz, _, num_img, num_region, hsz = Q.shape

        S, S_mask = self.similarity(C, Q, c_mask, q_mask)  
        S_c = F.softmax(S * self.scale, dim=-1)
        S_q = F.softmax(S * self.scale, dim=-2)

        S_c = S_c * S_mask  
        S_q = S_q * S_mask  

        A_c = torch.matmul(S_c, Q)  

        A_q = torch.matmul(S_q.transpose(-2, -1), C) 
        return A_c, A_q, S_mask, S_mask.transpose(-2, -1)

    def similarity(self, C, Q, c_mask, q_mask):

        C = F.dropout(F.normalize(C, p=2, dim=-1), p=self.dropout, training=self.training)
        Q = F.dropout(F.normalize(Q, p=2, dim=-1), p=self.dropout, training=self.training)

        S_mask = torch.matmul(c_mask.unsqueeze(-1), q_mask.unsqueeze(-2))  
        S = torch.matmul(C, Q.transpose(-2, -1))  
        masked_S = S - 1e10*(1 - S_mask) 
        return masked_S, S_mask

class StructuredAttention_frame(nn.Module):

    def __init__(self, dropout=0.1, scale=100):

        super(StructuredAttention_frame, self).__init__()
        self.dropout = dropout
        self.scale = scale

    def forward(self, C, Q, c_mask, q_mask):

        bsz, _, num_img, hsz = Q.shape

        S, S_mask = self.similarity(C, Q, c_mask, q_mask)  
        S_ = F.softmax(S * self.scale, dim=-1)
        S_ = S_ * S_mask  
        A = torch.matmul(S_, Q) 
        return A, S, S_mask, S_

    def similarity(self, C, Q, c_mask, q_mask):

        C = F.dropout(F.normalize(C, p=2, dim=-1), p=self.dropout, training=self.training)
        Q = F.dropout(F.normalize(Q, p=2, dim=-1), p=self.dropout, training=self.training)

        S_mask = c_mask.unsqueeze(-1)
        S = torch.matmul(C, Q.transpose(-2, -1)) 
        masked_S = S - 1e10*(1 - S_mask) 
        return masked_S, S_mask
