'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from LeafNATS.modules.encoder.encoder_cnn import EncoderCNN
from LeafNATS.modules.utils.LayerNormalization import LayerNormalization
from LeafNATS.modules.utils.PositionwiseFeedForward import \
    PositionwiseFeedForward_Basic


class CarbonLayer(torch.nn.Module):
    '''
    Implementation of Transformer
    '''

    def __init__(self, hidden_size, factor_size,
                 n_heads, drop_rate):
        super().__init__()
        self.n_heads = n_heads
        # convolution
        self.conv1 = torch.nn.Conv2d(1, n_heads, (3, n_heads), padding=(1, 0))
        self.conv2 = torch.nn.Conv2d(1, n_heads, (5, n_heads), padding=(2, 0))
        # layer normalization
        self.norm = LayerNormalization(hidden_size)
        # layer feed-forward
        self.ff = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(drop_rate, inplace=True)

    def forward(self, input_, mask=None):
        '''
        Transformer Layer
        '''
        (bsize, seq_len, hsize) = input_.size()
        input_vec = input_.view(
            bsize, seq_len, bsize/self.n_heads, self.n_heads)
        input_vec = input_.view(
            bsize, seq_len*bsize/self.n_heads, self.n_heads)
        hidden1 = self.conv1(input_.unsqueeze(1)).squeeze(1)
        hidden = self.norm(input_ + hidden1)
        hidden2 = self.conv2(input_.unsqueeze(1)).squeeze(1)
        hidden = self.norm(hidden + hidden2)
        hidden = self.norm(hidden + self.ff(hidden))

        return self.dropout(hidden)


class EncoderCarbon(torch.nn.Module):
    '''
    Implementation of Transformer Encoder
    '''

    def __init__(self, hidden_size, factor_size,
                 n_layers, n_heads, drop_rate,
                 device=torch.device("cpu")):
        super().__init__()
        self.n_layers = n_layers

        self.tf_layers = torch.nn.ModuleList(
            [CarbonLayer(hidden_size, factor_size, n_heads, drop_rate)
             for k in range(n_layers)])

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
