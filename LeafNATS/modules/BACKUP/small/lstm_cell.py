'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re

import torch

from .linear import LinearSmall


class LSTMCellSmall(torch.nn.Module):

    def __init__(
            self, input_size,
            hidden_size, factor_size):
        '''
        Implementation Linear Layer Small
        '''
        super().__init__()

        self.hidden_size = hidden_size

        self.ff_input = LinearSmall(input_size, hidden_size*4, factor_size)
        self.ff_hidden = LinearSmall(hidden_size, hidden_size*4, factor_size)

    def forward(self, input_, hidden_):
        '''
        forward
        '''
        (h0, c0) = hidden_

        batch_size = input_.size(0)
        gates = self.ff_input(input_) + self.ff_hidden(h0)
        i1, f1, g1, o1 = gates.chunk(4, 1)

        i1 = torch.sigmoid(i1)
        f1 = torch.sigmoid(f1)
        g1 = torch.tanh(g1)
        o1 = torch.sigmoid(o1)
        c1 = f1 * c0 + i1 * g1
        h1 = o1 * torch.tanh(c1)

        return h1, (h1, c1)
