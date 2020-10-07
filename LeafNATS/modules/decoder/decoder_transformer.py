'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.modules.attention.attention_multi_head import \
    MultiHeadedAttention_Basic
from LeafNATS.modules.utils.LayerNormalization import LayerNormalization
from LeafNATS.modules.utils.PositionwiseFeedForward import \
    PositionwiseFeedForward_Basic


class TransformerLayer(torch.nn.Module):
    '''
    Implementation of Transformer Decoder Layer.
    '''

    def __init__(self, input_size,
                 n_heads, drop_rate,
                 device=torch.device("cpu")):
        super().__init__()
        # multi-head attention
        self.attnSelf = MultiHeadedAttention_Basic(
            n_heads, input_size, drop_rate).to(device)
        self.attnEnc = MultiHeadedAttention_Basic(
            n_heads, input_size, drop_rate).to(device)
        # layer normalization
        self.norm1 = LayerNormalization(input_size).to(device)
        self.norm2 = LayerNormalization(input_size).to(device)
        self.norm3 = LayerNormalization(input_size).to(device)
        # layer feed-forward
        self.pos_ff = PositionwiseFeedForward_Basic(
            input_size, input_size*4, input_size, drop_rate
        ).to(device)

        self.drop = torch.nn.Dropout(drop_rate).to(device)

    def forward(self, src_, tgt_, src_mask=None, tgt_mask=None):
        '''
        Transformer Layer
        '''
        tgt_ = self.norm1(
            tgt_ + self.drop(self.attnSelf(tgt_, tgt_, tgt_, tgt_mask)))
        tgt_ = self.norm2(
            tgt_ + self.drop(self.attnEnc(tgt_, src_, src_, src_mask)))
        tgt_ = self.norm2(tgt_ + self.pos_ff(tgt_))

        return self.drop(tgt_)


class TransformerDecoder(torch.nn.Module):
    '''
    Implementation of Transformer Encoder
    '''

    def __init__(self, input_size, n_heads, n_layers,
                 drop_rate, device=torch.device("cpu")):
        super().__init__()
        self.n_heads = n_heads
        self.device = device
        self.n_layers = n_layers

        self.tf_layers = torch.nn.ModuleList(
            [TransformerLayer(input_size, n_heads, drop_rate, device)
             for k in range(n_layers)]).to(device)

    def forward(self, src_, tgt_, src_mask=None):
        '''
        Transformer
        '''
        batch_size = tgt_.size(0)
        seq_len = tgt_.size(1)
        mask_ = 1-np.triu(np.ones([seq_len, seq_len]), k=1)
        tgt_mask = Variable(torch.LongTensor(mask_)).to(self.device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(
            0).expand(batch_size, self.n_heads, -1, -1)

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).unsqueeze(1)

        tgt_ = self.tf_layers[0](src_, tgt_, src_mask, tgt_mask)
        for k in range(1, self.n_layers):
            tgt_ = self.tf_layers[k](src_, tgt_, src_mask, tgt_mask)

        return tgt_
