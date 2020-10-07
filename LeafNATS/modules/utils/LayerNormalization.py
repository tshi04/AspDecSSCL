'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch


class LayerNormalization(torch.nn.Module):
    '''
    Transformer Layer Normalization.
    '''

    def __init__(self, size, eps=1e-5, bias=False):
        super().__init__()

        self.eps = eps
        self.bias = bias

        self.gamma = torch.nn.Parameter(torch.ones(size))
        self.register_parameter('gamma', self.gamma)

        if bias:
            self.beta = torch.nn.Parameter(torch.zeros(size))
            self.register_parameter('beta', self.beta)

    def forward(self, input_):
        '''
        gamma * (input - mean) / (std + eps) + bias
        '''
        mean = torch.mean(input_, -1, keepdim=True)
        std = torch.mean((input_-mean)*(input_-mean), -1, keepdim=True)
        
        output = self.gamma*(input_-mean)/torch.sqrt(std + self.eps)
        if self.bias:
            output = output + self.beta

        return output
