'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from LeafNATS.data.utils import load_vocab_pretrain
from LeafNATS.engines.end2end_large import End2EndBase


class modelUAEBase(End2EndBase):
    '''
    Natural Language Generation Base
    '''

    def __init__(self, args):
        super().__init__(args=args)

        self.loss_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1).to(self.args.device)

    def build_vocabulary(self):
        '''
        build vocabulary
        '''
        vocab2id, id2vocab, pretrain_vec = load_vocab_pretrain(
            os.path.join(self.args.data_dir, self.args.file_pretrain_vocab),
            os.path.join(self.args.data_dir, self.args.file_pretrain_vec))
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['pretrain_emb'] = pretrain_vec
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

        self.batch_data['id2aspect'] = {}
        self.batch_data['aspect'] = []
        fp = open(os.path.join(self.args.data_dir, self.args.file_aspect), 'r')
        for k, line in enumerate(fp):
            self.batch_data['id2aspect'][k] = line[:-1]
            self.batch_data['aspect'].append(line[:-1])
        fp.close()
        self.batch_data['n_aspects'] = len(self.batch_data['id2aspect'])
        print(self.batch_data['id2aspect'])

    def build_optimizer(self, params):
        '''
        Build model optimizer
        '''
        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)

        return optimizer

    def init_base_model_params(self):
        '''
        Initialize Base Model Parameters.
        '''
        emb_para = torch.FloatTensor(
            self.batch_data['pretrain_emb']).to(self.args.device)
        self.base_models['embedding'].weight = torch.nn.Parameter(emb_para)

        for model_name in self.base_models:
            if model_name == 'embedding':
                continue
            fl_ = os.path.join(self.args.base_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))

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
        pos_text = []
        pos_tokens = []
        neg_text = []
        aspect_arr = []
        labels = []
        pos_text_len = 0
        fp = open(file_, 'r')
        for line in fp:
            line = json.loads(line)
            itm = line['text_fine']
            senid = [
                vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                for wd in itm['text']]
            pos_text.append(senid)
            pos_tokens.append(itm['text'])
            if len(senid) > pos_text_len:
                pos_text_len = len(senid)
            senid = []
            for wd in itm['text']:
                if wd in vocab2id and wd not in itm['noun']+itm['adj']:
                    senid.append(vocab2id[wd])
                else:
                    senid.append(vocab2id['<unk>'])
            neg_text.append(senid)
            senid = [vocab2id[wd] for wd in self.batch_data['aspect']]
            aspect_arr.append(senid)
            if self.args.task == 'test':
                labels.append(line['label_text'])
        fp.close()
        print(neg_text)

        pos_text = [
            itm + [vocab2id['<pad>'] for _ in range(pos_text_len-len(itm))]
            for itm in pos_text]
        pos_text_var = Variable(torch.LongTensor(
            pos_text)).to(self.args.device)
        pos_text_mask = Variable(torch.LongTensor(
            pos_text)).to(self.args.device)
        pos_text_mask[pos_text_mask != vocab2id['<pad>']] = -1
        pos_text_mask[pos_text_mask == vocab2id['<pad>']] = 0
        pos_text_mask = -pos_text_mask

        neg_text = [
            itm + [vocab2id['<pad>'] for _ in range(pos_text_len-len(itm))]
            for itm in neg_text]
        neg_text_var = Variable(torch.LongTensor(
            neg_text)).to(self.args.device)

        aspect_var = Variable(torch.LongTensor(
            aspect_arr)).to(self.args.device)

        self.batch_data['pos_text'] = pos_text_var
        self.batch_data['neg_text'] = neg_text_var
        self.batch_data['aspect_text'] = aspect_var
        self.batch_data['pos_text_mask'] = pos_text_mask
        self.batch_data['neg_text_mask'] = pos_text_mask
        if self.args.task == 'test':
            self.batch_data['pos_tokens'] = pos_tokens
            self.batch_data['label'] = labels

    def build_pos(self):
        '''
        data pipe
        '''
        raise NotImplementedError

    def build_neg(self):
        '''
        data pipe
        '''
        raise NotImplementedError

    def build_pipelines(self):
        '''
        Build pipeline from input to output.
        '''
        pos_vec, aspect_vec, aspect_enc, w_aspect = self.build_pos()
        neg_vec = self.build_neg()
        score_pos = aspect_vec.unsqueeze(1) @ pos_vec.unsqueeze(2)
        score_neg = aspect_vec.unsqueeze(1) @ neg_vec.unsqueeze(2)
        loss = torch.mean(torch.relu(1.0 + score_neg - score_pos))
        
        cross = aspect_enc @ aspect_enc.transpose(1, 2)
        diag = torch.eye(
            aspect_enc.size(1), aspect_enc.size(1)
        ).to(self.args.device)
        diag = diag.unsqueeze(0).repeat(aspect_enc.size(0), 1, 1)
        loss_cross = cross - diag
        loss_cross = loss_cross * loss_cross

        return loss + torch.mean(loss_cross)

    def test_worker(self):
        '''
        For testing.
        '''
        pos_vec, aspect_vec, aspect_enc, w_aspect = self.build_pos()
        idx = w_aspect.topk(k=1)[1].squeeze(1).data.cpu().numpy()[0]

        self.test_data['text'] = ' '.join(self.batch_data['pos_tokens'][0])
        self.test_data['gold'] = self.batch_data['label'][0]
        self.test_data['pred'] = self.batch_data['id2aspect'][idx]
