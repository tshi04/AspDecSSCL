'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch
from LeafNATS.modules.activation.functions import gelu


class PositionwiseFeedForward_Basic(torch.nn.Module):
    '''
    Implementation of Positionwise FeedForward Network.
    '''

    def __init__(self, input_size,
                 hidden_size, output_size, drop_rate):
        super().__init__()

        self.ff1 = torch.nn.Linear(input_size, hidden_size)
        self.ff2 = torch.nn.Linear(hidden_size, output_size)
        self.drop = torch.nn.Dropout(drop_rate, inplace=True)

    def forward(self, input_):

        output = self.drop(gelu(self.ff1(input_)))
        output = self.drop(self.ff2(output))

        return output
