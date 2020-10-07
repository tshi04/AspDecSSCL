'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable
from transformers import BertModel, AlbertModel, DistilBertModel

from AspDec.model_kd_base import modelKDBase


class modelKD(modelKDBase):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        build all models.
        '''
        if self.args.pretrained_model == 'bert':
            hidden_size = 768
            self.pretrained_models['encoder'] = BertModel.from_pretrained(
                'bert-base-uncased',
                output_hidden_states=True,
                output_attentions=True
            ).to(self.args.device)
        if self.args.pretrained_model == 'bertlarge':
            hidden_size = 1024
            self.pretrained_models['encoder'] = BertModel.from_pretrained(
                'bert-large-uncased',
                output_hidden_states=True,
                output_attentions=True
            ).to(self.args.device)
        if self.args.pretrained_model == 'albert':
            hidden_size = 768
            self.pretrained_models['encoder'] = AlbertModel.from_pretrained(
                'albert-base-v2',
                output_hidden_states=True,
                output_attentions=True
            ).to(self.args.device)
        if self.args.pretrained_model == 'distilbert':
            hidden_size = 768
            self.pretrained_models['encoder'] = DistilBertModel.from_pretrained(
                'distilbert-base-uncased', 
                output_hidden_states=True,
                output_attentions=True
            ).to(self.args.device)

        self.train_models['attn_kernel'] = torch.nn.Linear(
            hidden_size, hidden_size
        ).to(self.args.device)

        self.train_models['classifier'] = torch.nn.Linear(
            hidden_size, self.batch_data['n_aspects']
        ).to(self.args.device)

    def build_pipe(self):
        '''
        data pipe
        '''
        bsize = self.batch_data['sen_text_var'].size(0)

        with torch.no_grad():
            if self.args.pretrained_model == 'bert':
                sen_enc = self.pretrained_models['encoder'](
                    self.batch_data['sen_text_var'],
                    self.batch_data['sen_pad_mask'])[0]
            if self.args.pretrained_model == 'bertlarge':
                sen_enc = self.pretrained_models['encoder'](
                    self.batch_data['sen_text_var'],
                    self.batch_data['sen_pad_mask'])[0]
            if self.args.pretrained_model == 'distilbert':
                sen_enc = self.pretrained_models['encoder'](
                    self.batch_data['sen_text_var'],
                    self.batch_data['sen_pad_mask'])[0]
            if self.args.pretrained_model == 'albert':
                sen_enc = self.pretrained_models['encoder'](
                    self.batch_data['sen_text_var'],
                    self.batch_data['sen_pad_mask'])[0]

        mask_ = self.batch_data['sen_pad_mask']
        sen_enc = sen_enc * mask_.unsqueeze(2)
        enc_avg = torch.sum(sen_enc, dim=1)
        norm = torch.sum(mask_, dim=1, keepdim=True) + 1e-20
        enc_avg = enc_avg.div(norm.expand_as(enc_avg))
        enc_trn = self.train_models['attn_kernel'](sen_enc)
        attn_ = enc_avg.unsqueeze(1) @ enc_trn.transpose(1, 2)
        attn_ = self.args.smooth_factor*torch.tanh(attn_.squeeze(1))
        attn_ = attn_.masked_fill(mask_ == 0, -1e20)
        attn_ = torch.softmax(attn_, dim=1)
        ctx_ = attn_.unsqueeze(1) @ sen_enc
        ctx_ = ctx_.squeeze(1)

        logits = torch.tanh(self.train_models['classifier'](ctx_))
        prob = torch.softmax(logits, dim=-1)

        return prob
