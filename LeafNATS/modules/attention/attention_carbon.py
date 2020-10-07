'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch
import torch.nn.functional as F


class MultiHeadedAttention_Carbon(torch.nn.Module):
    '''
    Implement of multi-head attention.
    '''

    def __init__(self, n_heads, hidden_size,
                 factor_size, drop_rate):
        super().__init__()

        assert hidden_size % n_heads == 0
        self.n_dk = hidden_size // n_heads
        self.n_heads = n_heads

        self.proj_query = torch.nn.Linear(hidden_size, factor_size)
        self.tran_query = torch.nn.Linear(factor_size, hidden_size)
        self.proj_key = torch.nn.Linear(hidden_size, factor_size)
        self.tran_key = torch.nn.Linear(factor_size, hidden_size)
        self.proj_value = torch.nn.Linear(hidden_size, factor_size)
        self.tran_value = torch.nn.Linear(factor_size, hidden_size)
        self.proj_output = torch.nn.Linear(hidden_size, factor_size)
        self.tran_output = torch.nn.Linear(factor_size, hidden_size)

        self.dropout = torch.nn.Dropout(drop_rate, inplace=True)

    def forward(self, query, key, value, mask=None):
        '''
        Input: embedding.
        '''
        bsize = query.size(0)

        query = self.tran_query(F.leaky_relu(self.proj_query(query), 0.2))
        query = query.view(
            bsize, -1, self.n_heads, self.n_dk).transpose(1, 2)
        key = self.tran_key(F.leaky_relu(self.proj_key(key), 0.2))
        key = key.view(
            bsize, -1, self.n_heads, self.n_dk).transpose(1, 2)
        value = self.tran_value(F.leaky_relu(self.proj_value(value), 0.2))
        value = value.view(
            bsize, -1, self.n_heads, self.n_dk).transpose(1, 2)
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.n_dk)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e20)
        attn = torch.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        cv = attn @ value
        cv = cv.transpose(1, 2)
        cv = cv.contiguous().view(bsize, -1, self.n_heads*self.n_dk)

        return self.dropout(self.tran_output(F.leaky_relu(self.proj_output(cv), 0.2)))
