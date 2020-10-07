'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import numpy as np
import torch
from torch.autograd import Variable


class AttentionSelf(torch.nn.Module):

    def __init__(
            self, input_size, hidden_size,
            dropout_rate=None,
            device=torch.device("cpu")):
        '''
        implementation of self-attention.
        '''
        super().__init__()
        self.dropout_rate = dropout_rate
        self.device = device

        self.ff1 = torch.nn.Linear(input_size, hidden_size)
        self.ff2 = torch.nn.Linear(hidden_size, 1, bias=False)
        if dropout_rate is not None:
            self.model_drop = torch.nn.Dropout(dropout_rate)

    def forward(self, input_, mask=None):
        '''
        input vector: input_
        output:
            attn_: attention weights
            ctx_vec: context vector
        '''
        attn_ = torch.tanh(self.ff1(input_))
        attn_ = self.ff2(attn_).squeeze(2)
        if mask is not None:
            attn_ = attn_.masked_fill(mask == 0, -1e9)
        # dropout method 1.
        # if self.dropout_rate is not None:
        #     drop_mask = Variable(torch.ones(attn_.size())).to(self.device)
        #     drop_mask = self.model_drop(drop_mask)
        #     attn_ = attn_.masked_fill(drop_mask == 0, -1e9)

        attn_ = torch.softmax(attn_, dim=1)
        # dropout method 2.
        if self.dropout_rate is not None:
            attn_ = self.model_drop(attn_)
        ctx_vec = torch.bmm(attn_.unsqueeze(1), input_).squeeze(1)

        return attn_, ctx_vec
