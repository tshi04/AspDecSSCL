'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch
import torch.nn.functional as F


class MultiHeadedAttention_Basic(torch.nn.Module):
    '''
    Implement of multi-head attention.
    '''

    def __init__(self, n_heads,
                 hidden_size, drop_rate):
        super().__init__()

        assert hidden_size % n_heads == 0
        self.n_dk = hidden_size // n_heads
        self.n_heads = n_heads

        self.proj_query = torch.nn.Linear(hidden_size, hidden_size)
        self.proj_key = torch.nn.Linear(hidden_size, hidden_size)
        self.proj_value = torch.nn.Linear(hidden_size, hidden_size)
        self.drop = torch.nn.Dropout(drop_rate)

        self.proj_output = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        '''
        Input: embedding.
        '''
        batch_size = query.size(0)

        query = self.proj_query(query)
        query = query.view(batch_size, -1, self.n_heads,
                           self.n_dk).transpose(1, 2)
        key = self.proj_key(key)
        key = key.view(batch_size, -1, self.n_heads,
                       self.n_dk).transpose(1, 2)
        value = self.proj_value(value)
        value = value.view(batch_size, -1, self.n_heads,
                           self.n_dk).transpose(1, 2)
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.n_dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        cv = attn @ value
        cv = cv.transpose(1, 2)
        cv = cv.contiguous().view(batch_size, -1, self.n_heads*self.n_dk)

        return self.drop(self.proj_output(cv))
