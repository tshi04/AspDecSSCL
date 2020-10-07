'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch


class PositionalEmbedding(torch.nn.Module):
    '''
    Implementation of Positional Embedding.
    '''

    def __init__(self, hidden_size, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = hidden_size

        self.posEmb = torch.zeros(10000, hidden_size, dtype=torch.float)
        self.posEmb.require_grad = False

        position = torch.arange(10000, dtype=torch.float).unsqueeze(1)
        p_term1 = torch.arange(0, hidden_size, 2, dtype=torch.float)
        p_term2 = - math.log(10000.0) / hidden_size
        inv_term = torch.exp(p_term1 * p_term2)

        posEmb_input = position * inv_term
        self.posEmb[:, 0::2] = torch.sin(posEmb_input)
        self.posEmb[:, 1::2] = torch.cos(posEmb_input)

        self.posEmb = self.posEmb.unsqueeze(0).to(device)

    def forward(self, input_):
        '''
        input_: Input sequence.
        '''
        seq_len = input_.size(1)
        pos_emb = self.posEmb[:, :seq_len]

        return pos_emb
