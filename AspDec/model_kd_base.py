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
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm import tqdm
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from .end2end_AspDec import End2EndAspDecBase


class modelKDBase(End2EndAspDecBase):
    '''
    Teacher Student Base
    '''

    def __init__(self, args):
        super().__init__(args=args)

        self.pretrained_models = {}
        self.loss_criterion = torch.nn.NLLLoss(
        ).to(self.args.device)

        if self.args.pretrained_model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')
        if self.args.pretrained_model == 'bertlarge':
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-large-uncased')
        if self.args.pretrained_model == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(
                'albert-base-v2')
        if self.args.pretrained_model == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased')

    def build_vocabulary(self):
        '''
        build vocabulary
        '''
        vocab2id = self.tokenizer.get_vocab()
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
        id2vocab = self.batch_data['id2vocab']

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
            senid = self.tokenizer.encode(itm['text_reg'])
            if self.args.pretrained_model == 'bert':
                senid = senid[1:-1]
            if self.args.pretrained_model == 'bertlarge':
                senid = senid[1:-1]
            if self.args.pretrained_model == 'albert':
                senid = senid[1:-1]
            if self.args.pretrained_model == 'distilbert':
                senid = senid[1:-1]
            if self.args.task[-5:] == 'train' and \
                    len(senid) < self.args.min_seq_len:
                continue
            sen_tokens.append([id2vocab[wd] for wd in senid])
            sen_text_arr.append(senid)
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
            if self.args.pretrained_model == 'bert':
                pad = [0 for _ in range(sen_len-len(out))]
                out = [101] + out + [102] + pad
            if self.args.pretrained_model == 'bertlarge':
                pad = [0 for _ in range(sen_len-len(out))]
                out = [101] + out + [102] + pad
            if self.args.pretrained_model == 'distilbert':
                pad = [0 for _ in range(sen_len-len(out))]
                out = [101] + out + [102] + pad
            if self.args.pretrained_model == 'albert':
                pad = [0 for _ in range(sen_len-len(out))]
                out = [2] + out + [3] + pad
            sen_text.append(out)
        sen_text_var = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)
        sen_pad_mask = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)
        if self.args.pretrained_model == 'bert':
            sen_pad_mask[sen_pad_mask != 0] = -1
            sen_pad_mask[sen_pad_mask == 0] = 0
        if self.args.pretrained_model == 'bertlarge':
            sen_pad_mask[sen_pad_mask != 0] = -1
            sen_pad_mask[sen_pad_mask == 0] = 0
        if self.args.pretrained_model == 'distilbert':
            sen_pad_mask[sen_pad_mask != 0] = -1
            sen_pad_mask[sen_pad_mask == 0] = 0
        if self.args.pretrained_model == 'albert':
            sen_pad_mask[sen_pad_mask != 0] = -1
            sen_pad_mask[sen_pad_mask == 0] = 0
        sen_pad_mask = -sen_pad_mask

        self.batch_data['sen_tokens'] = sen_tokens
        self.batch_data['sen_text_var'] = sen_text_var
        self.batch_data['sen_pad_mask'] = sen_pad_mask

        if self.args.task[-5:] == 'train':
            labels_hard = Variable(torch.LongTensor(
                labels_hard)).to(self.args.device)
            labels_soft = Variable(torch.FloatTensor(
                labels_soft)).to(self.args.device)
            norm = torch.sum(labels_soft, dim=1, keepdim=True)
            labels_soft = labels_soft.div(norm.expand_as(labels_soft))

            self.batch_data['labels_hard'] = labels_hard
            self.batch_data['labels_soft'] = labels_soft

        else:
            self.batch_data['labels'] = labels_hard

    def aspect_worker(self):
        '''
        Aspect keywords
        '''
        return

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
        logprob = torch.log(prob + 1e-20)

        soft_labels = self.batch_data['labels_soft']
        uct = -soft_labels * torch.log(soft_labels+1e-20)
        uct = torch.sum(uct, dim=1)

        asp_lb = self.batch_data['aspect_labels']
        idx_arr = []
        for k, sen in enumerate(self.batch_data['sen_tokens']):
            if uct[k] > self.args.thresh_aspect:
                continue
            lb = asp_lb[self.batch_data['labels_hard'][k]]
            if uct[k] > self.args.thresh_general and lb == 'none':
                continue
            idx_arr.append(k)
        if len(idx_arr) == 0:
            idx_arr.append(0)
        logprob = logprob[idx_arr]
        label = self.batch_data['labels_hard'][idx_arr]
        loss = self.loss_criterion(logprob, label)

        return loss

    def test_worker(self):
        '''
        For testing.
        '''
        prob = self.build_pipe()
        if self.args.none_type:
            topidx = prob.topk(k=1)[1].squeeze().data.cpu().numpy().tolist()
        else:
            topidx = prob.topk(k=2)[1].squeeze().data.cpu().numpy().tolist()
            for k in range(len(topidx)):
                if self.batch_data['aspect_labels'][topidx[k][0]] == 'none':
                    topidx[k] = topidx[k][1]
                else:
                    topidx[k] = topidx[k][0]

        for k in range(len(self.batch_data['sen_tokens'])):
            output = {}
            output['text'] = self.batch_data['sen_tokens'][k]
            output['pred_label'] = self.batch_data['aspect_labels'][topidx[k]]
            output['gold_label'] = self.batch_data['labels'][k]
            self.test_data.append(output)

    def evaluate_worker(self, input_):
        '''
        Used for evaluation
        '''
        asp_labels = []
        pred = []
        gold = []
        for itm in input_:
            if itm['pred_label'] in itm['gold_label'].split(','):
                pred.append(itm['pred_label'])
                gold.append(itm['pred_label'])
            else:
                pred.append(itm['pred_label'])
                gold.append(itm['gold_label'].split(',')[0])
            for wd in itm['gold_label'].split(','):
                asp_labels.append(wd)

        asp_labels = sorted(list(set(asp_labels)))
        asp_map = {wd: k for k, wd in enumerate(asp_labels)}

        pred = [asp_map[wd] for wd in pred]
        gold = [asp_map[wd] for wd in gold]

        return f1_score(gold, pred, average='macro')
