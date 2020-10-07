'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch


def gelu(input_):
    '''
    GELU activation function.
    '''
    return input_ * 0.5 * (1.0 + torch.erf(input_ / math.sqrt(2.0)))


def maxout(input_, pool_size):
    '''
    maxout activation
    '''
    input_size = list(input_.size())
    assert input_.size(-1) % pool_size == 0

    out_size = input_.size(-1) // pool_size
    input_size[-1] = out_size
    input_size.append(pool_size)

    return input_.view(*input_size).max(-1)
