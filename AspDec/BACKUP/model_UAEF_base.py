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

from LeafNATS.data.utils import load_vocab_pretrain
from UAE.ABAE.evaluation import eval_aspect_coherence

from .model_UAE_base import modelUAEBase


class modelUAEFBase(modelUAEBase):
    '''
    Natural Language Generation Base
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_vocabulary(self):
        '''
        build vocabulary
        '''
        cluster_dir = '../cluster_results'
        vocab2id, id2vocab, pretrain_vec = load_vocab_pretrain(
            os.path.join(cluster_dir, 'vocab'),
            os.path.join(cluster_dir, 'word2vec_vec.npy'))
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['pretrain_emb'] = pretrain_vec
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

        aspect_noun = np.loadtxt(os.path.join(
            cluster_dir, 'kmc_noun.txt'), dtype=float)
        aspect_adj = np.loadtxt(os.path.join(
            cluster_dir, 'kmc_adj.txt'), dtype=float)
        aspect_noun = torch.FloatTensor(aspect_noun).to(self.args.device)
        aspect_adj = torch.FloatTensor(aspect_adj).to(self.args.device)
        aspect_vec = torch.cat((aspect_noun, aspect_adj), dim=0)
        self.batch_data['aspect_vec'] = aspect_vec
        self.batch_data['n_aspects'] = aspect_vec.shape[0]

        aspect_adj = np.loadtxt(os.path.join(
            cluster_dir, 'kmc_adj.txt'), dtype=float)
        
        self.batch_data['aspect_adj'] = aspect_adj
        self.batch_data['n_aspects_adj'] = aspect_adj.shape[0]

    def init_base_model_params(self):
        '''
        Initialize Base Model Parameters.
        '''
        emb_para = torch.FloatTensor(
            self.batch_data['pretrain_emb']).to(self.args.device)
        self.base_models['embedding'].weight = torch.nn.Parameter(emb_para)

        if self.args.task == 'train':
            aspect_para = self.batch_data['aspect_vec']
            aspect_para = torch.nn.Parameter(aspect_para)
            self.train_models['aspect_embedding'].weight = aspect_para

        for model_name in self.base_models:
            if model_name == 'embedding' or model_name == 'aspect_embedding':
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
        sen_text = []
        sen_tokens = []
        mask_words = []
        labels = []
        sen_text_len = 0
        fp = open(file_, 'r')
        for line in fp:
            itm = json.loads(line)
            senid = [1 if wd in ['noun', 'adj'] else 0 for wd in itm['text_pos']]
            mask_words.append(senid)
            if self.args.task == 'train':
                if len(mask_words) == 0:
                    continue
            senid = [
                vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                for wd in itm['text_fine']]
            sen_text.append(senid)
            sen_tokens.append(itm['text_fine'])
            if len(senid) > sen_text_len:
                sen_text_len = len(senid)
            try:
                labels.append(itm['label'])
            except:
                labels.append('None')
        fp.close()

        sen_text_len = min(sen_text_len, self.args.max_seq_len)

        sen_text = [
            itm[:sen_text_len]
            + [vocab2id['<pad>'] for _ in range(sen_text_len-len(itm))]
            for itm in sen_text]
        sen_text_var = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)
        sen_pad_mask = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)
        sen_pad_mask[sen_pad_mask != vocab2id['<pad>']] = -1
        sen_pad_mask[sen_pad_mask == vocab2id['<pad>']] = 0
        sen_pad_mask = -sen_pad_mask

        mask_words = [
            itm[:sen_text_len] + [0 for _ in range(sen_text_len-len(itm))]
            for itm in mask_words]
        mask_words = Variable(torch.FloatTensor(
            mask_words)).to(self.args.device)

        self.batch_data['sen_tokens'] = sen_tokens
        self.batch_data['pos_sen_var'] = sen_text_var
        self.batch_data['pos_pad_mask'] = sen_pad_mask
        self.batch_data['pos_words_mask'] = mask_words
        self.batch_data['label'] = labels

        if self.args.task == 'train' or self.args.task == 'validate':
            neg_examples = []
            for k in range(20):
                random.shuffle(sen_text)
                neg_text_var = Variable(torch.LongTensor(
                    sen_text)).to(self.args.device)
                neg_pad_mask = Variable(torch.LongTensor(
                    sen_text)).to(self.args.device)
                neg_pad_mask[neg_pad_mask != vocab2id['<pad>']] = -1
                neg_pad_mask[neg_pad_mask == vocab2id['<pad>']] = 0
                neg_pad_mask = -neg_pad_mask
                neg_examples.append([neg_text_var, neg_pad_mask])

            self.batch_data['neg_examples'] = neg_examples

    def build_pipe(self):
        '''
        data pipe
        '''
        raise NotImplementedError

    def build_pipelines(self):
        '''
        Build pipeline from input to output.
        '''
        loss, asp_vec = self.build_pipe()

        asp_norm = asp_vec / \
            (asp_vec.norm(p=2, dim=2, keepdim=True) + 1e-20)
        cross = asp_norm @ asp_norm.transpose(1, 2)
        diag = torch.eye(cross.size(1)).to(self.args.device)
        diag = diag.unsqueeze(0).expand_as(cross)
        diff = cross - diag
        loss_cross = diff.norm(p=2)

        return loss + loss_cross

    def test_worker(self):
        '''
        For testing.
        '''
        asp_weights = self.build_pipe()
        asp_weights = asp_weights.data.cpu().numpy().tolist()

        for k in range(len(self.batch_data['sen_tokens'])):
            output = {}
            output['text'] = self.batch_data['sen_tokens'][k]
            output['aspect_weight'] = asp_weights[k]
            output['gold_label'] = self.batch_data['label'][k]
            self.test_data.append(output)

    def aspect_worker(self):
        '''
        Aspect keywords
        '''
        emb = self.base_models['embedding'].weight
        emb = emb.unsqueeze(0)

        asp_emb = self.train_models['aspect_embedding'].weight
        asp_emb = asp_emb.unsqueeze(0)

        score = asp_emb @ emb.transpose(1, 2)
        score = score.squeeze(0)

        top_idx = score.topk(
            k=self.args.n_keywords)[1].cpu().numpy().tolist()

        output = []
        for idx in top_idx:
            out = []
            for wd in idx:
                if wd in self.batch_data['id2vocab']:
                    out.append(self.batch_data['id2vocab'][wd])
            output.append(out)

        for k, itm in enumerate(output):
            print('{}: {}'.format(k+1, ' '.join(itm)))

        if self.args.evaluate_coherence:
            eval_aspect_coherence(
                cluster_dir, self.args.file_term_doc,
                self.args.file_vocab, self.args.n_keywords,
                weight=score.data.cpu().numpy().tolist())
