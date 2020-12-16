'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable

from LeafNATS.modules.attention.attention_self import AttentionSelf
from LeafNATS.modules.attention.nats_attention_encoder import AttentionEncoder
from LeafNATS.modules.encoder2decoder.nats_encoder2decoder import \
    natsEncoder2Decoder
from LeafNATS.modules.encoder.nats_encoder_rnn import natsEncoder
from UAE.model_UAE_base import modelUAEBase


class modelUAE(modelUAEBase):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        build all models.
        '''
        self.base_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'],
            self.args.emb_size
        ).to(self.args.device)

        self.train_models['aspect_embedding'] = torch.nn.Embedding(
            self.batch_data['n_aspects'],
            self.args.emb_size
        ).to(self.args.device)

        self.train_models['attn_kernel'] = torch.nn.Linear(
            self.args.emb_size,
            self.args.emb_size
        ).to(self.args.device)

        self.train_models['asp_weight'] = torch.nn.Linear(
            self.args.emb_size,
            self.batch_data['n_aspects']
        ).to(self.args.device)

    def build_encoder(self, input_, mask_):
        '''
        encoder
        '''
        bsize = input_.size(0)
        seq_len = input_.size(1)

        with torch.no_grad():
            emb_ = self.base_models['embedding'](input_)
        emb_ = emb_ * mask_.unsqueeze(2)
        emb_avg = torch.sum(emb_, dim=1)
        norm = torch.sum(mask_, dim=1, keepdim=True) + 1e-20
        emb_avg = emb_avg.div(norm.expand_as(emb_avg))
        emb_avg = self.train_models['attn_kernel'](emb_avg)
        attn_ = emb_avg.unsqueeze(1) @ emb_.transpose(1, 2)
        attn_ = torch.tanh(attn_.squeeze(1))
        attn_ = attn_.masked_fill(mask_ == 0, -1e20)
        attn_ = torch.softmax(attn_, dim=1)
        ctx_ = attn_.unsqueeze(1) @ emb_
        ctx_ = ctx_.squeeze(1)

        return attn_, ctx_

    def compute_distance(self, vec1, vec2):
        '''
        Compute distances
        '''
        vec1 = vec1 / (vec1.norm(p=2, dim=1, keepdim=True) + 1e-20)
        vec2 = vec2 / (vec2.norm(p=2, dim=1, keepdim=True) + 1e-20)

        if self.args.distance == 'cosine':
            score = vec1.unsqueeze(1) @ vec2.unsqueeze(2)
            score = score.squeeze(1).squeeze(1)

        return score

    def build_pipe(self):
        '''
        data pipe
        '''
        bsize = self.batch_data['pos_sen_var'].size(0)

        attn_pos, ctx_pos = self.build_encoder(
            self.batch_data['pos_sen_var'],
            self.batch_data['pos_pad_mask'])

        asp_weight = self.train_models['asp_weight'](ctx_pos)
        asp_weight = torch.softmax(asp_weight, dim=1)
        asp = torch.LongTensor(range(self.batch_data['n_aspects']))
        asp = Variable(asp).to(self.args.device)
        asp = asp.unsqueeze(0).repeat(bsize, 1)
        asp_emb = self.train_models['aspect_embedding'](asp)

        if self.args.task == 'train' or self.args.task == 'validate':
            asp_enc = asp_weight.unsqueeze(1) @ asp_emb
            asp_enc = asp_enc.squeeze(1)

            score_pos = self.compute_distance(asp_enc, ctx_pos)

            loss_arr = []
            for itm in self.batch_data['neg_examples']:
                attn_neg, ctx_neg = self.build_encoder(itm[0], itm[1])

                score_neg = self.compute_distance(asp_enc, ctx_neg)

                diff = torch.relu(1.0 - score_pos + score_neg)
                loss_arr.append(diff)

            loss = torch.cat(loss_arr, 0)
            loss = torch.mean(loss)

            return loss, asp_emb
        else:
            return asp_weight
