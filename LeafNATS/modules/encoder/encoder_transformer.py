'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch

from LeafNATS.modules.attention.attention_multi_head import \
    MultiHeadedAttention_Basic
from LeafNATS.modules.utils.LayerNormalization import LayerNormalization
from LeafNATS.modules.utils.PositionwiseFeedForward import \
    PositionwiseFeedForward_Basic


class TransformerLayer(torch.nn.Module):
    '''
    Implementation of Transformer
    '''

    def __init__(self, input_size, n_heads, drop_rate,
                 device=torch.device("cpu")):
        super().__init__()
        # multi-head attention
        self.attentionMH = MultiHeadedAttention_Basic(
            n_heads, input_size, drop_rate).to(device)
        # layer normalization
        self.norm = LayerNormalization(input_size).to(device)
        # layer feed-forward
        self.pos_ff = PositionwiseFeedForward_Basic(
            input_size, input_size*4, input_size, drop_rate
        ).to(device)
        self.drop = torch.nn.Dropout(drop_rate).to(device)

    def forward(self, input_, mask=None):
        '''
        Transformer Layer
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        hidden = self.attentionMH(input_, input_, input_, mask)
        hidden = self.norm(input_ + self.drop(hidden))
        hidden = self.norm(hidden + self.pos_ff(hidden))

        return self.drop(hidden)


class TransformerEncoder(torch.nn.Module):
    '''
    Implementation of Transformer Encoder
    '''

    def __init__(self, input_size, n_heads, n_layers,
                 drop_rate, device=torch.device("cpu")):
        super().__init__()

        self.n_layers = n_layers

        self.tf_layers = torch.nn.ModuleList(
            [TransformerLayer(input_size, n_heads, drop_rate, device)
             for k in range(n_layers)]).to(device)

    def forward(self, input_, mask=None):
        '''
        Transformer
        '''
        output = []
        out = self.tf_layers[0](input_, mask)
        output.append(out)
        for k in range(1, self.n_layers):
            out = self.tf_layers[k](out, mask)
            output.append(out)

        return output
