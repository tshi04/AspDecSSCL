'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable
from transformers import BertModel

from AspExt.model_TS_base import modelTSBase
from LeafNATS.modules.encoder.encoder_cnn import EncoderCNN
from LeafNATS.modules.attention.attention_self import AttentionSelf


class modelTS(modelTSBase):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        build all models.
        '''
        self.pretrained_models['bert'] = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states=True,
            output_attentions=True
        ).to(self.args.device)

        self.train_models['attn_kernel'] = torch.nn.Linear(
            768, 768
        ).to(self.args.device)

        self.train_models['classifier'] = torch.nn.Linear(
            768, self.batch_data['n_aspects']
        ).to(self.args.device)

    def build_pipe(self):
        '''
        data pipe
        '''
        bsize = self.batch_data['sen_text_var'].size(0)

        with torch.no_grad():
            sen_enc = self.pretrained_models['bert'](
                self.batch_data['sen_text_var'], 
                self.batch_data['sen_pad_mask'])[0]
        
        enc_avg = torch.sum(sen_enc, dim=1)
        mask_ = self.batch_data['sen_pad_mask']
        norm = torch.sum(mask_, dim=1, keepdim=True) + 1e-20
        enc_avg = enc_avg.div(norm.expand_as(enc_avg))
        enc_trn = self.train_models['attn_kernel'](sen_enc)
        attn_ = enc_avg.unsqueeze(1) @ enc_trn.transpose(1, 2)
        attn_ = self.args.lambda_*torch.tanh(attn_.squeeze(1))
        attn_ = attn_.masked_fill(mask_ == 0, -1e20)
        attn_ = torch.softmax(attn_, dim=1)
        ctx_ = attn_.unsqueeze(1) @ sen_enc
        ctx_ = ctx_.squeeze(1)

        logits = torch.tanh(self.train_models['classifier'](ctx_))
        prob = torch.softmax(logits, dim=-1)

        return prob
