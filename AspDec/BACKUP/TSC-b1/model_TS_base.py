'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from transformers import BertTokenizer

from .end2end_TS import End2EndTSBase
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


class modelTSBase(End2EndTSBase):
    '''
    Teacher Student Base
    '''

    def __init__(self, args):
        super().__init__(args=args)

        self.pretrained_models = {}
        self.loss_criterion = torch.nn.NLLLoss(
        ).to(self.args.device)

    def build_vocabulary(self):
        '''
        build vocabulary
        '''
        vocab2id = tokenizer.get_vocab()
        id2vocab = {vocab2id[wd]: wd for wd in vocab2id}
        vocab_size = len(vocab2id)

        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

        asp_map = []
        fp = open(os.path.join(
            '../nats_results', 'aspect_mapping.txt'), 'r')
        for line in fp:
            itm = line.split()
            asp_map.append(itm[1])
        fp.close()
        self.batch_data['aspect_mapping'] = asp_map
        asp_lb = [wd for wd in sorted(list(set(asp_map))) if wd != 'nomap']
        self.batch_data['aspect_labels'] = asp_lb
        self.batch_data['n_aspects'] = len(asp_lb)

    def build_optimizer(self, params):
        '''
        Build model optimizer
        '''
        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)

        return optimizer

    def build_batch(self, batch_id):
        '''
        get batch data
        '''
        vocab2id = self.batch_data['vocab2id']

        path_ = os.path.join('..', 'nats_results')
        fkey_ = self.args.task
        batch_size = self.args.batch_size
        file_ = os.path.join(path_, 'batch_{}_{}'.format(
            fkey_, batch_size), str(batch_id))
        sen_text_arr = []
        sen_tokens = []
        labels_hard = []
        labels_soft = []
        sen_len = 0
        fp = open(file_, 'r')
        for line in fp:
            itm = json.loads(line)
            senid = [
                vocab2id[wd] for wd in itm['text_bert'].split()]
            if self.args.task[-5:] == 'train' and len(senid) < 3:
                continue
            senid = [vocab2id['[CLS]']] + senid + [vocab2id['[SEP]']]
            sen_text_arr.append(senid)
            sen_tokens.append(itm['text_bert'])
            if len(senid) > sen_len:
                sen_len = len(senid)
            if self.args.task[-5:] == 'train':
                asp_wt = {
                    wd: 0 for wd in self.batch_data['aspect_mapping'] 
                    if wd != 'nomap'}
                for k, wt in enumerate(itm['aspect_weight']):
                    wd = self.batch_data['aspect_mapping'][k]
                    if wd == 'nomap':
                        continue
                    asp_wt[wd] += wt
                asp_wt = [[wd, asp_wt[wd]] for wd in asp_wt]
                asp_wt = sorted(asp_wt)
                labels_soft.append([wd[1] for wd in asp_wt])
                asp_rank = np.argsort([wd[1] for wd in asp_wt])[::-1]
                labels_hard.append(asp_rank[0])
            else:
                labels_hard.append(itm['label'])
        fp.close()

        sen_len = min(sen_len, self.args.max_seq_len)
        sen_text = []
        for itm in sen_text_arr:
            out = itm[:sen_len]
            pad = [vocab2id['[PAD]'] for _ in range(sen_len-len(out))]
            out = [vocab2id['[CLS]']] + out + [vocab2id['[SEP]']] + pad
            sen_text.append(out)
        sen_text_var = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)
        sen_pad_mask = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)
        sen_pad_mask[sen_pad_mask != vocab2id['[PAD]']] = -1
        sen_pad_mask[sen_pad_mask == vocab2id['[PAD]']] = 0
        sen_pad_mask = -sen_pad_mask

        self.batch_data['sen_tokens'] = sen_tokens
        self.batch_data['sen_text_var'] = sen_text_var
        self.batch_data['sen_pad_mask'] = sen_pad_mask

        if self.args.task[-5:] == 'train':
            labels_hard = Variable(torch.LongTensor(
                labels_hard)).to(self.args.device)
            labels_soft = Variable(torch.FloatTensor(
                labels_soft)).to(self.args.device)

            self.batch_data['labels_hard'] = labels_hard
            self.batch_data['labels_soft'] = labels_soft

        else:
            self.batch_data['labels'] = labels_hard

    def build_pipe(self):
        '''
        data pipe
        '''
        raise NotImplementedError

    def build_pipelines(self):
        '''
        Build pipeline from input to output.
        '''
        prob = self.build_pipe()
        logprob = torch.log(prob)

        soft_labels = self.batch_data['labels_soft']
        uct = -soft_labels * torch.log(soft_labels+1e-20)
        uct = torch.sum(uct, dim=1)

        idx_arr = []
        for k, sen in enumerate(self.batch_data['sen_tokens']):
            if uct[k] > 1.0:
                continue
            if uct[k] > 0.7 and self.batch_data['labels_hard'][k] == 5:
                continue
            idx_arr.append(k)
        logprob = logprob[idx_arr]
        label = self.batch_data['labels_hard'][idx_arr]
        loss = self.loss_criterion(logprob, label)

        return loss

    def test_worker(self):
        '''
        For testing.
        '''
        prob = self.build_pipe()
        topidx = prob.topk(k=1)[1].squeeze().data.cpu().numpy().tolist()

        for k in range(len(self.batch_data['sen_tokens'])):
            output = {}
            output['text'] = self.batch_data['sen_tokens'][k]
            output['pred_label'] = self.batch_data['aspect_labels'][topidx[k]]
            output['gold_label'] = self.batch_data['labels'][k]
            self.test_data.append(output)
