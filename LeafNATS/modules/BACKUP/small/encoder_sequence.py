'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re

import torch

from .linear import LinearSmall


class EncoderLayer(torch.nn.Module):

    def __init__(self, input_size, factor_size):
        '''
        Implementation Sequence Encoder
        '''
        super().__init__()

        self.ff_input = LinearSmall(input_size, input_size*5, factor_size)

    def forward(self, input_):
        '''
        forward
        '''
        proj_vec = self.ff_input(input_)
        hidden, cell, gateIN, gateOUT, gateFG = proj_vec.chunk(5, 2)

        gateIN = torch.sigmoid(gateIN)
        gateOUT = torch.sigmoid(gateOUT)
        gateFG = torch.sigmoid(gateFG)

        cell = gateFG * torch.tanh(cell)
        cell = torch.cumsum(cell, 1)

        hidden = gateIN * torch.tanh(hidden) + cell

        return input_ + gateOUT * hidden


class EncoderSequence(torch.nn.Module):

    def __init__(self, input_size, factor_size, n_layers):
        '''
        Implementation Sequence Encoder
        '''
        super().__init__()

        self.n_layers = n_layers

        self.enc_model = torch.nn.ModuleList(
            [EncoderLayer(input_size, factor_size)
             if k == 0 else EncoderLayer(input_size, factor_size)
             for k in range(n_layers)])

    def forward(self, input_):
        '''
        forward
        '''
        print(torch.flip(input_, 1))
        out = self.enc_model[0](input_)
        for k in range(1, self.n_layers):
            out = self.enc_model[k](out)

        return out
