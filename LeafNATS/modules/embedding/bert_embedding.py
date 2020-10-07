'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch

from LeafNATS.modules.embedding.position_embedding import PositionalEmbedding
from LeafNATS.modules.utils.LayerNormalization import LayerNormalization


class BertEmbeddings(torch.nn.Module):
    '''
    Implementation of BERT embedding layer.
    Light Weight. 
    Not follow the original implementation.
    '''

    def __init__(self, vocab_size, hidden_size,
                 drop_rate, device=torch.device("cpu")):
        super().__init__()

        self.device = device

        self.word_embeddings = torch.nn.Embedding(
            vocab_size, hidden_size).to(device)
        self.segment_embeddings = torch.nn.Embedding(
            2, hidden_size).to(device)
        self.position_embeddings = PositionalEmbedding(
            hidden_size, device).to(device)

        self.proj2vocab = torch.nn.Linear(
            hidden_size, vocab_size).to(device)
        self.proj2vocab.weight.data = self.word_embeddings.weight.data

        self.dropout = torch.nn.Dropout(drop_rate).to(device)

    def forward(self, input_tokens, input_seg=None):
        '''
        input_tokens: input sequence token ids.
        input_seg: input segment ids.
        '''
        word_vec = self.word_embeddings(input_tokens)
        position_vec = self.position_embeddings(input_tokens)
        if input_seg is None:
            input_seg = torch.zeros(
                input_tokens.size(), dtype=torch.long).to(self.device)

        seg_vec = self.segment_embeddings(input_seg)
        output_vec = word_vec + position_vec + seg_vec
        output_vec = self.dropout(output_vec)

        return output_vec

    def get_word_embedding(self, input_tokens):
        '''
        Get word embedding only.
        '''
        return self.word_embeddings(input_tokens)

    def get_vec2vocab(self, input_):
        '''
        get a vector:
            size = vocab size
        later, pass this vector to softmax layer to probability.
        '''
        return self.proj2vocab(input_)
