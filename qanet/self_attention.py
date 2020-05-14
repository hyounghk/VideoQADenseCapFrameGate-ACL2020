__author__ = "Jie Lei"
"""
The code here are borrowed from The Annotated Transformer [1], with
minor modifications.

[1] http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""
import torch.nn as nn
import torch
import math
import copy


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, nh, d_model, dropout=0.1):

        super(MultiHeadedAttention, self).__init__()
        assert d_model % nh == 0
        self.d_k = d_model // nh
        self.nh = nh
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):

        bsz = x.size(0)
        if mask is not None:
            mask = mask.view(bsz, 1, -1, 1)  

        query, key, value = [l(x).view(bsz, -1, self.nh, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (x, x, x))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)  

        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.nh * self.d_k)  
        return self.linears[-1](x)  

    def attention(self, query, key, value, mask=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  
