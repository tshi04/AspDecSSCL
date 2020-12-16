'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from transformers import BertTokenizer

from LeafNATS.engines.end2end_pretrain import End2EndBase

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


class modelPreTrainBase(End2EndBase):
    '''
    Natural Language Generation Base
    '''

    def __init__(self, args):
        super().__init__(args=args)

        self.pretrained_models = {}
        self.base_models = {}

    def build_vocabulary(self):
        '''
        build vocabulary
        '''
        bert_vocab = tokenizer.get_vocab()
        vocab = {}
        fp = open(os.path.join(
            self.args.data_dir, self.args.file_train), 'r')
        for line in tqdm(fp):
            itm = json.loads(line)
            for wd in itm['text_bert'].split():
                try:
                    vocab[wd] += 1
                except:
                    vocab[wd] = 1
        fp.close()
        vocab2id = {}
        id2vocab = {}
        for wd in vocab:
            if vocab[wd] >= self.args.min_count:
                vocab2id[wd] = bert_vocab[wd]
                id2vocab[bert_vocab[wd]] = wd
        self.batch_data['bert_vocab'] = bert_vocab
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['vocab_size'] = len(vocab2id)
        print('The vocabulary size: {}'.format(len(vocab2id)))

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
        vocab2id = self.batch_data['bert_vocab']

        path_ = os.path.join('..', 'nats_results')
        fkey_ = self.args.task
        batch_size = self.args.batch_size
        file_ = os.path.join(path_, 'batch_{}_{}'.format(
            fkey_, batch_size), str(batch_id))
        sen_text = []
        sen_len = 0
        fp = open(file_, 'r')
        for line in fp:
            itm = json.loads(line)
            senid = [
                vocab2id[wd] for wd in itm['text_bert']
                if wd in vocab2id]
            if len(senid) > self.args.max_seq_len:
                senid = senid[:self.args.max_seq_len]
            senid = [vocab2id['[CLS]']] + senid + [vocab2id['[SEP]']]
            sen_text.append(senid)
            if len(senid) > sen_len:
                sen_len = len(senid)        
        fp.close()
        
        sen_text = [
            itm[:sen_len]
            + [vocab2id['[PAD]'] for _ in range(sen_len-len(itm))]
            for itm in sen_text]
        sen_text_var = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)
        sen_pad_mask = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)
        sen_pad_mask[sen_pad_mask != vocab2id['[PAD]']] = -1
        sen_pad_mask[sen_pad_mask == vocab2id['[PAD]']] = 0
        sen_pad_mask = -sen_pad_mask

        self.batch_data['input_ids'] = sen_text_var
        self.batch_data['pad_mask'] = sen_pad_mask

    def build_pipe(self):
        '''
        data pipe
        '''
        raise NotImplementedError

    def build_pipelines(self):
        '''
        Build pipeline from input to output.
        '''
        loss = self.build_pipe()

        return loss

    def dump_embeddings(self):
        '''
        dump word embeddings
        '''
        self.build_vocabulary()
        self.build_models()
        for model in self.train_models:
            self.base_models[model] = self.train_models[model]
        self.init_base_model_params()
        vocab = []
        for wd in self.batch_data['vocab2id']:
            vocab.append([wd, self.batch_data['vocab2id'][wd]])
        vocab = sorted(vocab)
        index = [itm[1] for itm in vocab]
        vec = self.base_models['bert_emb'].weight[index]
        vec = vec.data.cpu().numpy()

        print('Begin to dump files...')
        if not os.path.exists(self.args.cluster_dir):
            os.mkdir(self.args.cluster_dir)
        np.save(os.path.join(
            self.args.cluster_dir, self.args.file_wordvec), vec)
        fout = open(os.path.join(
            self.args.cluster_dir, self.args.file_vocab), 'w')
        for k, itm in enumerate(vocab):            
            fout.write('{} {}\n'.format(itm[0], itm[1]))
        fout.close()
        
