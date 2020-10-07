'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re

import torch


class LinearSmall(torch.nn.Module):

    def __init__(
            self, input_size,
            output_size, factor_size):
        '''
        Implementation Linear Layer Small
        '''
        super().__init__()

        self.ff1 = torch.nn.Linear(input_size, factor_size)
        self.ff2 = torch.nn.Linear(factor_size, output_size)

    def forward(self, input_):
        '''
        forward
        '''
        output = self.ff2(torch.relu(self.ff1(input_)))

        return output
