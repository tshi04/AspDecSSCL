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

from AspExt.TSC.evaluation import eval_aspect_coherence
from LeafNATS.data.utils import load_vocab_pretrain

from .end2end_AspExt import End2EndAspExtBase


class modelABAEBase(End2EndAspExtBase):
    '''
    Autoencoder based aspect extraction
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_vocabulary(self):
        '''
        build vocabulary
        '''
        vocab2id, id2vocab, pretrain_vec = load_vocab_pretrain(
            os.path.join(self.args.cluster_dir, self.args.file_vocab),
            os.path.join(
                self.args.cluster_dir,
                self.args.file_wordvec + '.npy'))
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['pretrain_emb'] = pretrain_vec
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

        aspect_vec = np.loadtxt(os.path.join(
            self.args.cluster_dir,
            self.args.file_kmeans_centroid), dtype=float)
        aspect_vec = torch.FloatTensor(aspect_vec).to(self.args.device)
        pad = torch.zeros(self.args.emb_size).to(self.args.device)
        aspect_vec = torch.cat((pad.unsqueeze(0), aspect_vec), dim=0)
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
        bert_tokens = []
        asp_mapping = []
        labels = []
        sen_text_len = 0
        fp = open(file_, 'r')
        for line in fp:
            itm = json.loads(line)
            senid = [
                vocab2id[wd] for wd in itm['text_uae'].split()
                if wd in vocab2id]
            if self.args.task[-5:] == 'train' and len(senid) < 3:
                continue
            sen_text.append(senid)
            uae_tokens.append(itm['text_uae'])
            reg_tokens.append(itm['text_reg'])
            bert_tokens.append(itm['text_bert'])
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

        self.batch_data['uae_tokens'] = uae_tokens
        self.batch_data['reg_tokens'] = reg_tokens
        self.batch_data['bert_tokens'] = bert_tokens

        self.batch_data['pos_sen_var'] = sen_text_var
        self.batch_data['pos_pad_mask'] = sen_pad_mask
        self.batch_data['label'] = labels

        if self.args.task[-5:] == 'train':
            neg_examples = []
            neg_text = sen_text
            neg_asp_mapping = asp_mapping
            for k in range(20):
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

        for k in range(len(self.batch_data['uae_tokens'])):
            output = {}
            output['text_bert'] = self.batch_data['bert_tokens'][k]
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
        asp_emb = asp_emb[1:]
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

        fout = open(os.path.join(
            '../nats_results', self.args.file_keywords_dump), 'w')
        for itm in output:
            fout.write('{}\n'.format(' '.join(itm)))
        fout.close()

        for k, itm in enumerate(output):
            print('{}: {}'.format(k+1, ' '.join(itm)))

        if self.args.evaluate_coherence:
            eval_aspect_coherence(
                self.args.cluster_dir, self.args.file_term_doc,
                self.args.file_vocab, self.args.n_keywords,
                weight=score.data.cpu().numpy().tolist())
