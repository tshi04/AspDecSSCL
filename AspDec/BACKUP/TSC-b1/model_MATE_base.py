'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .model_ABAE_base import modelABAEBase


class modelMATEBase(modelABAEBase):
    '''
    Natural Language Generation Base
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_pipelines(self):
        '''
        Build pipeline from input to output.
        '''
        loss = self.build_pipe()

        return loss
