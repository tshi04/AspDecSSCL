'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable

from AspExt.model_ABAE_base import modelABAEBase


class modelABAE(modelABAEBase):

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
            self.args.emb_size,
            padding_idx=0
        ).to(self.args.device)

        self.train_models['attn_kernel'] = torch.nn.Linear(
            self.args.emb_size,
            self.args.emb_size
        ).to(self.args.device)

        self.train_models['asp_weight'] = torch.nn.Linear(
            self.args.emb_size, self.batch_data['n_aspects']-1
        ).to(self.args.device)

    def init_base_model_params(self):
        '''
        Initialize Base Model Parameters.
        '''
        emb_para = torch.FloatTensor(
            self.batch_data['pretrain_emb']).to(self.args.device)
        self.base_models['embedding'].weight = torch.nn.Parameter(emb_para)

        if self.args.task[-5:] == 'train':
            aspect_para = self.batch_data['aspect_centroid']
            aspect_para = torch.nn.Parameter(aspect_para)
            self.train_models['aspect_embedding'].weight = aspect_para

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
        emb_trn = self.train_models['attn_kernel'](emb_)
        attn_ = emb_avg.unsqueeze(1) @ emb_trn.transpose(1, 2)
        attn_ = self.args.lambda_*torch.tanh(attn_.squeeze(1))
        attn_ = attn_.masked_fill(mask_ == 0, -1e20)
        attn_ = torch.softmax(attn_, dim=1)
        ctx_ = attn_.unsqueeze(1) @ emb_
        ctx_ = ctx_.squeeze(1)

        return ctx_

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

        ctx_pos = self.build_encoder(
            self.batch_data['pos_sen_var'],
            self.batch_data['pos_pad_mask'])

        asp_weight = self.train_models['asp_weight'](ctx_pos)
        asp_weight = torch.softmax(asp_weight, dim=1)

        if self.args.task[-5:] == 'train':
            asp = torch.LongTensor(range(1, self.batch_data['n_aspects']))
            asp = Variable(asp).to(self.args.device)
            asp = asp.unsqueeze(0).repeat(bsize, 1)
            asp_emb = self.train_models['aspect_embedding'](asp)
            asp_enc = asp_weight.unsqueeze(1) @ asp_emb
            asp_enc = asp_enc.squeeze(1)
            score_pos = self.compute_distance(asp_enc, ctx_pos)

            loss_arr = []
            for itm in self.batch_data['neg_examples']:
                ctx_neg = self.build_encoder(itm[0], itm[1])
                score_neg = self.compute_distance(asp_enc, ctx_neg)

                diff = torch.relu(1.0 - score_pos + score_neg)
                loss_arr.append(diff)
            loss = torch.cat(loss_arr, 0)
            loss = torch.mean(loss)

            return loss, asp_emb
        else:
            return asp_weight

    def get_embedding_weights(self):
        '''
        Get embedding matrix
        '''
        emb = self.base_models['embedding'].weight
        asp_emb = self.train_models['aspect_embedding'].weight

        return emb, asp_emb
