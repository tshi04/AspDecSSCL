'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable

from AspExt.model_PreTrain_base import modelPreTrainBase
from transformers import BertModel


class modelBERTEmb(modelPreTrainBase):

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

        self.train_models['bert_emb'] = torch.nn.Linear(
            768, len(self.batch_data['bert_vocab']), bias=False
        ).to(self.args.device)

        wd_mask = torch.ones(
            len(self.batch_data['bert_vocab'])).to(self.args.device)
        wd_mask[self.batch_data['bert_vocab']['[PAD]']] = 0
        wd_mask[self.batch_data['bert_vocab']['[CLS]']] = 0
        wd_mask[self.batch_data['bert_vocab']['[SEP]']] = 0
        self.loss_criterion = torch.nn.NLLLoss(wd_mask).to(self.args.device)

    def build_pipe(self):
        '''
        data pipe
        '''
        with torch.no_grad():
            input_enc = self.pretrained_models['bert'](
                self.batch_data['input_ids'],
                self.batch_data['pad_mask'])[0]

        output = self.train_models['bert_emb'](input_enc)
        output = torch.softmax(output, dim=2)
        output = torch.log(output)

        loss = self.loss_criterion(
            output.view(-1, len(self.batch_data['bert_vocab'])),
            self.batch_data['input_ids'].view(-1))

        return loss
        
