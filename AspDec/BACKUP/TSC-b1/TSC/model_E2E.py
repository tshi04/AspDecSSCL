'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable

from AspExt.model_E2E_base import modelE2Ebase
from LeafNATS.modules.encoder.encoder_cnn import EncoderCNN


class modelE2E(modelE2Ebase):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        build all models.
        '''
        self.train_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'],
            self.args.emb_size
        ).to(self.args.device)

        self.train_models['aspect_embedding'] = torch.nn.Embedding(
            self.args.n_clusters,
            self.args.emb_size
        ).to(self.args.device)

        self.train_models['attn_kernel'] = torch.nn.Linear(
            self.args.emb_size,
            self.args.emb_size
        ).to(self.args.device)

        self.train_models['asp_weight'] = torch.nn.Linear(
            self.args.emb_size, self.args.n_clusters
        ).to(self.args.device)

    def build_encoder(self, input_, mask_):
        '''
        encoder
        '''
        bsize = input_.size(0)
        seq_len = input_.size(1)

        emb_avg = torch.sum(input_, dim=1)
        norm = torch.sum(mask_, dim=1, keepdim=True) + 1e-20
        emb_avg = emb_avg.div(norm.expand_as(emb_avg))
        emb_trn = self.train_models['attn_kernel'](input_)
        attn_ = emb_avg.unsqueeze(1) @ emb_trn.transpose(1, 2)
        attn_ = torch.tanh(attn_.squeeze(1))
        attn_ = attn_.masked_fill(mask_ == 0, -1e20)
        attn_ = torch.softmax(attn_, dim=1)
        ctx_ = attn_.unsqueeze(1) @ input_
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
        bsize = self.batch_data['pos_uae_var'].size(0)

        pos_uae_emb = self.train_models['embedding'](
            self.batch_data['pos_uae_var'])
        pos_uae_emb = pos_uae_emb * \
            self.batch_data['pos_uae_mask'].unsqueeze(2)
        # pos_reg_emb = self.train_models['embedding'](
        #     self.batch_data['pos_reg_var'])
        # pos_reg_emb = pos_reg_emb * \
        #     self.batch_data['pos_reg_mask'].unsqueeze(2)

        pos_uae_enc = self.build_encoder(
            pos_uae_emb, self.batch_data['pos_uae_mask'])
        # pos_reg_enc = self.build_encoder(
        #     pos_reg_emb, self.batch_data['pos_reg_mask'])

        pos_uae_weight = 5.0*torch.tanh(
            self.train_models['asp_weight'](pos_uae_enc))
        pos_uae_weight = torch.softmax(pos_uae_weight, dim=1)

        # pos_reg_weight = 5.0*torch.tanh(
        #     self.train_models['asp_weight'](pos_reg_enc))
        # pos_reg_weight = torch.softmax(pos_reg_weight, dim=1)

        if self.args.task[-5:] == 'train':
            w2v_uae_emb = self.train_models['embedding'](
                self.batch_data['w2v_uae_var'])
            w2v_uae_emb = w2v_uae_emb * \
                self.batch_data['w2v_uae_mask'].unsqueeze(2)
            w2v_uae_ov = w2v_uae_emb @ pos_uae_enc.unsqueeze(2)
            w2v_uae_ov = w2v_uae_ov.squeeze(2)
            w2v_uae_ov = torch.sigmoid(w2v_uae_ov)
            w2v_uae_loss = torch.mean(w2v_uae_ov)

            # w2v_reg_emb = self.train_models['embedding'](
            #     self.batch_data['w2v_reg_var'])
            # w2v_reg_emb = w2v_reg_emb * \
            #     self.batch_data['w2v_reg_mask'].unsqueeze(2)
            # w2v_reg_ov = w2v_reg_emb @ pos_uae_enc.unsqueeze(2)
            # w2v_reg_ov = w2v_reg_ov.squeeze(2)
            # w2v_reg_ov = torch.sigmoid(w2v_reg_ov)
            # w2v_reg_loss = torch.mean(w2v_reg_ov)

            asp = torch.LongTensor(range(self.args.n_clusters))
            asp = Variable(asp).to(self.args.device)
            asp = asp.unsqueeze(0).repeat(bsize, 1)
            asp_emb = self.train_models['aspect_embedding'](asp)
            asp_enc = pos_uae_weight.unsqueeze(1) @ asp_emb
            asp_enc = asp_enc.squeeze(1)

            rec_score_pos = self.compute_distance(asp_enc, pos_uae_enc)
            # sim_score_pos = self.compute_distance(pos_uae_enc, pos_reg_enc)

            loss_rec = []
            for itm in self.batch_data['neg_examples']:
                neg_uae_emb = self.train_models['embedding'](itm[0])
                neg_uae_emb = neg_uae_emb * itm[1].unsqueeze(2)
                neg_uae_enc = self.build_encoder(neg_uae_emb, itm[1])

                # neg_reg_emb = self.train_models['embedding'](itm[2])
                # neg_reg_emb = neg_reg_emb * itm[3].unsqueeze(2)
                # neg_reg_enc = self.build_encoder(neg_reg_emb, itm[3])

                rec_score_neg = self.compute_distance(asp_enc, neg_uae_enc)
                # sim_score_neg = self.compute_distance(pos_uae_enc, neg_reg_enc)

                diff_rec = torch.relu(1.0 - rec_score_pos + rec_score_neg)
                loss_rec.append(diff_rec)

            loss_rec = torch.cat(loss_rec, 0)
            loss_rec = torch.mean(loss_rec)

            # loss_weight = torch.mean(torch.norm(
            #     pos_uae_weight-pos_reg_weight, p=2, dim=1))
            
            return loss_rec + w2v_uae_loss, asp_emb
        else:
            return pos_uae_weight

    def get_embedding_weights(self):
        '''
        Get embedding matrix
        '''
        emb = self.train_models['embedding'].weight
        asp_emb = self.train_models['aspect_embedding'].weight

        return emb, asp_emb
