'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch
import torch.nn.functional as F

from LeafNATS.modules.embedding.position_embedding import PositionalEmbedding
from LeafNATS.modules.utils.LayerNormalization import LayerNormalization


class CarbonEmbeddings(torch.nn.Module):
    '''
    Implementation of BERT embedding layer.
    Light Weight. 
    '''

    def __init__(self, vocab_size, hidden_size, factor_size,
                 device=torch.device("cpu")):
        super().__init__()

        self.word_embeddings = torch.nn.Embedding(
            vocab_size, factor_size)
        self.word_trans = torch.nn.Linear(
            factor_size, hidden_size)
        self.position_embeddings = PositionalEmbedding(
            factor_size, device)
        self.position_trans = torch.nn.Linear(
            factor_size, hidden_size)
        self.norm = LayerNormalization(hidden_size)

    def forward(self, input_tokens):
        '''
        input_tokens: input sequence token ids.
        input_seg: input segment ids.
        '''
        word_vec = self.word_embeddings(input_tokens)
        word_vec = self.word_trans(
            F.leaky_relu(word_vec, 0.2))
        position_vec = self.position_embeddings(input_tokens)
        position_vec = self.position_trans(
            F.leaky_relu(position_vec, 0.2))
        output_vec = self.norm(word_vec + position_vec)

        return output_vec
