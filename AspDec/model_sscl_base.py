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
from sklearn.metrics import f1_score
from torch.autograd import Variable

from AspDec.SSCL.evaluation import eval_aspect_coherence
from LeafNATS.data.utils import load_vocab_pretrain

from .end2end_AspDec import End2EndAspDecBase


class modelSSCLBase(End2EndAspDecBase):
    '''
    Aspect Detection
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_vocabulary(self):
        '''
        build vocabulary
        '''
        cluster_dir = '../cluster_results'
        file_wordvec = 'vectors_w2v.npy'
        file_vocab = 'vocab.txt'
        file_kmeans_centroid = 'aspect_centroid.txt'

        vocab2id, id2vocab, pretrain_vec = load_vocab_pretrain(
            os.path.join(cluster_dir, file_vocab),
            os.path.join(cluster_dir, file_wordvec))
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['pretrain_emb'] = pretrain_vec
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

        aspect_vec = np.loadtxt(os.path.join(
            cluster_dir, file_kmeans_centroid), dtype=float)
        aspect_vec = torch.FloatTensor(aspect_vec).to(self.args.device)
        self.batch_data['aspect_centroid'] = aspect_vec
        self.batch_data['n_aspects'] = aspect_vec.shape[0]

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
        sen_text = []
        uae_tokens = []
        reg_tokens = []
        asp_mapping = []
        labels = []
        sen_text_len = 0
        fp = open(file_, 'r')
        for line in fp:
            itm = json.loads(line)
            senid = [
                vocab2id[wd] for wd in itm['text_uae'].split()
                if wd in vocab2id]
            if self.args.task[-5:] == 'train' and \
                len(senid) < self.args.min_seq_len:
                continue
            sen_text.append(senid)
            uae_tokens.append(itm['text_uae'])
            reg_tokens.append(itm['text_reg'])
            if len(senid) > sen_text_len:
                sen_text_len = len(senid)
            try:
                labels.append(itm['label'].lower())
            except:
                labels.append('none')
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

        self.batch_data['uae_tokens'] = uae_tokens
        self.batch_data['reg_tokens'] = reg_tokens

        self.batch_data['pos_sen_var'] = sen_text_var
        self.batch_data['pos_pad_mask'] = sen_pad_mask
        self.batch_data['label'] = labels

        if self.args.task[-5:] == 'train':
            neg_examples = []
            neg_text = sen_text
            neg_asp_mapping = asp_mapping
            
            for k in range(len(sen_text)):
                neg_text = neg_text[1:] + [neg_text[0]]
                neg_text_var = Variable(torch.LongTensor(
                    neg_text)).to(self.args.device)
                neg_pad_mask = Variable(torch.LongTensor(
                    neg_text)).to(self.args.device)
                neg_pad_mask[neg_pad_mask != vocab2id['<pad>']] = -1
                neg_pad_mask[neg_pad_mask == vocab2id['<pad>']] = 0
                neg_pad_mask = -neg_pad_mask
                neg_examples.append(
                    [neg_text_var, neg_pad_mask])
            self.batch_data['neg_examples'] = neg_examples

        if batch_id % self.args.checkpoint == 0:
            self.aspect_worker()

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
        
        return loss + loss_cross * (0.1 + self.args.warmup_step/self.global_steps)

    def test_worker(self):
        '''
        For testing.
        '''
        asp_weights = self.build_pipe()
        asp_weights = asp_weights.data.cpu().numpy().tolist()

        for k in range(len(self.batch_data['uae_tokens'])):
            output = {}
            output['text_uae'] = self.batch_data['uae_tokens'][k]
            output['text_reg'] = self.batch_data['reg_tokens'][k]
            output['aspect_weight'] = asp_weights[k]
            output['label'] = self.batch_data['label'][k]
            self.test_data.append(output)

    def get_embedding_weights(self):
        '''
        Get embedding matrix
        '''
        raise NotImplementedError

    def aspect_worker(self):
        '''
        Aspect keywords
        '''
        emb, asp_emb = self.get_embedding_weights()

        emb = emb.unsqueeze(0)
        asp_emb = asp_emb.unsqueeze(0)

        score = asp_emb @ emb.transpose(1, 2)
        score = score.squeeze(0)

        top_idx = score.topk(
            k=self.args.n_keywords, dim=1)[1].cpu().numpy().tolist()

        output = []
        for idx in top_idx:
            out = []
            for wd in idx:
                if wd in self.batch_data['id2vocab']:
                    out.append(self.batch_data['id2vocab'][wd])
            output.append(out)

        if self.args.task[-5:] == 'train':
            fout = open(os.path.join(
                '../nats_results', 'sscl_aspect_keywords.txt'), 'w')
            for itm in output:
                fout.write('{}\n'.format(' '.join(itm)))
            fout.close()
        else:
            fout = open(os.path.join(
                '../nats_results', 'test_sscl_aspect_keywords.txt'), 'w')
            for itm in output:
                fout.write('{}\n'.format(' '.join(itm)))
            fout.close()

    def evaluate_worker(self, input_):
        '''
        Used for evaluation
        '''
        aspect_label = []
        fp = open('../nats_results/aspect_mapping.txt', 'r')
        for line in fp:
            aspect_label.append(line.split()[1])
        fp.close()

        ignore_type = ['nomap']
        if not self.args.none_type:
            ignore_type.append('none')
        tmp = {wd: -1 for wd in aspect_label if not wd in ignore_type}
        label = {}
        for k, wd in enumerate(sorted(list(tmp))):
            label[wd] = k

        pred = []
        gold = []
        for itm in input_:
            arr = np.argsort(itm['aspect_weight'])[::-1]
            for k in arr:
                if not aspect_label[k] in ignore_type:
                    pp = aspect_label[k]
                    break
            pred.append(label[pp])
            try:
                gold.append(label[itm['label']])
            except:
                lb = itm['label'].split(',')
                if pp in lb:
                    gold.append(label[pp])
                else:
                    gold.append(label[lb[0]])

        return f1_score(gold, pred, average='macro')
