import math
import torch
import torch.nn as nn


class PositionEncoding(nn.Module):

    def __init__(self, n_filters=128, max_len=500):

        super(PositionEncoding, self).__init__()

        pe = torch.zeros(max_len, n_filters)  
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  

    def forward(self, x):

        pe = self.pe.data[:x.size(-2), :]  
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x
