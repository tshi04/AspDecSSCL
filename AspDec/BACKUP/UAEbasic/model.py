'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable

from UAE.model_UAE_base import modelUAEBase


class modelUAE(modelUAEBase):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        build all models.
        '''
        hidden_size = 300

        self.base_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'], hidden_size
        ).to(self.args.device)

        self.train_models['gate'] = torch.nn.Linear(
            hidden_size, hidden_size
        ).to(self.args.device)
        self.train_models['value'] = torch.nn.Linear(
            hidden_size, hidden_size
        ).to(self.args.device)

        self.train_models['kernel'] = torch.nn.Linear(
            self.args.hidden_size, self.args.hidden_size, bias=False
        ).to(self.args.device)

        self.train_models['weight_aspect'] = torch.nn.Linear(
            self.args.hidden_size, self.batch_data['n_aspects']
        ).to(self.args.device)

    def build_pos(self):
        '''
        data pipe
        '''
        # src text
        with torch.no_grad():
            text_emb = self.base_models['embedding'](
                self.batch_data['pos_text'])
        emb_gate = torch.sigmoid(self.train_models['gate'](text_emb))
        emb_valu = torch.relu(self.train_models['value'](text_emb))
        text_emb = text_emb*(1-emb_gate) + emb_valu*emb_gate

        emb_avg = torch.mean(text_emb, dim=1)
        emb_avg = self.train_models['kernel'](emb_avg)
        attn = emb_avg.unsqueeze(1) @ text_emb.transpose(1, 2)
        attn = attn.squeeze(1)
        attn = attn.masked_fill(
            self.batch_data['pos_text_mask'] == 0, -1e20)
        attn = torch.softmax(attn, dim=1)
        pos_vec = attn.unsqueeze(1) @ text_emb
        pos_vec = pos_vec.squeeze(1)

        # aspect
        with torch.no_grad():
            aspect_emb = self.base_models['embedding'](
                self.batch_data['aspect_text'])
        emb_gate = torch.sigmoid(self.train_models['gate'](aspect_emb))
        emb_valu = torch.relu(self.train_models['value'](aspect_emb))
        aspect_emb = aspect_emb*(1-emb_gate) + emb_valu*emb_gate

        w_aspect = self.train_models['weight_aspect'](pos_vec)
        w_aspect = torch.softmax(w_aspect, dim=1)
        aspect_vec = w_aspect.unsqueeze(1) @ aspect_emb
        aspect_vec = aspect_vec.squeeze(1)

        return pos_vec, aspect_vec, aspect_emb, w_aspect

    def build_neg(self):
        '''
        data pipe
        '''
        with torch.no_grad():
            text_emb = self.base_models['embedding'](
                self.batch_data['neg_text'])
        emb_gate = torch.sigmoid(self.train_models['gate'](text_emb))
        emb_valu = torch.relu(self.train_models['value'](text_emb))
        text_emb = text_emb*(1-emb_gate) + emb_valu*emb_gate

        emb_avg = torch.mean(text_emb, dim=1)
        emb_avg = self.train_models['kernel'](emb_avg)
        attn = emb_avg.unsqueeze(1) @ text_emb.transpose(1, 2)
        attn = attn.squeeze(1)
        attn = attn.masked_fill(
            self.batch_data['neg_text_mask'] == 0, -1e20)
        attn = torch.softmax(attn, dim=1)
        neg_vec = attn.unsqueeze(1) @ text_emb
        neg_vec = neg_vec.squeeze(1)

        return neg_vec
