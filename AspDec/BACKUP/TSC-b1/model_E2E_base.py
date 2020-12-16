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

from AspExt.TSC.evaluation import eval_aspect_coherence
from LeafNATS.data.utils import load_vocab_pretrain

from .end2end_AspExt import End2EndAspExtBase


class modelE2Ebase(End2EndAspExtBase):
    '''
    Natural Language Generation Base
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_vocabulary(self):
        '''
        build vocabulary
        '''
        try:
            fp = open(os.path.join(
                '../nats_results', 'vocab.pickled'), 'rb')
            vocab2id = pickle.load(fp)
            fp.close()
        except:
            vocab = {}
            fp = open(os.path.join(
                self.args.data_dir, self.args.file_train), 'r')
            for line in tqdm(fp):
                itm = json.loads(line)
                for wd in itm['text_reg'].split():
                    try:
                        vocab[wd] += 1
                    except:
                        vocab[wd] = 1
                for wd in itm['text_uae'].split():
                    try:
                        vocab[wd] += 1
                    except:
                        vocab[wd] = 1
            fp.close()
            vocab_arr = []
            for wd in vocab:
                if vocab[wd] > self.args.min_count*2:
                    vocab_arr.append(wd)
            vocab_arr = sorted(vocab_arr)
            vocab2id = {'<pad>': 0, '<unk>': 1}
            lenv = len(vocab2id)
            for k, wd in enumerate(vocab_arr):
                vocab2id[wd] = k + lenv

            out_dir = '../nats_results'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            fout = open(os.path.join(
                out_dir, 'vocab.pickled'), 'wb')
            pickle.dump(vocab2id, fout)
            fout.close()

        id2vocab = {vocab2id[wd]: wd for wd in vocab2id}
        vocab_size = len(vocab2id)

        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

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
        pos_uae = []
        pos_reg = []
        sen_tokens = []
        labels = []
        pos_uae_len = 0
        pos_reg_len = 0
        fp = open(file_, 'r')
        for line in fp:
            itm = json.loads(line)
            senid = [
                vocab2id[wd] for wd in itm['text_uae'].split()
                if wd in vocab2id]
            if self.args.task[-5:] == 'train' and len(senid) < 3:
                continue
            pos_uae.append(senid)
            sen_tok = [
                wd for wd in itm['text_uae'].split() if wd in vocab2id]
            sen_tokens.append(sen_tok)
            if len(senid) > pos_uae_len:
                pos_uae_len = len(senid)
            senid = [
                vocab2id[wd] for wd in itm['text_reg'].split()
                if wd in vocab2id]
            pos_reg.append(senid)
            if len(senid) > pos_reg_len:
                pos_reg_len = len(senid)
            try:
                labels.append(itm['label'])
            except:
                labels.append('None')
        fp.close()

        pos_uae_len = min(pos_uae_len, self.args.max_seq_len)
        pos_reg_len = min(pos_reg_len, self.args.max_seq_len)

        pos_uae = [
            itm[:pos_uae_len]
            + [vocab2id['<pad>'] for _ in range(pos_uae_len-len(itm))]
            for itm in pos_uae]
        pos_uae_var = Variable(torch.LongTensor(
            pos_uae)).to(self.args.device)
        pos_uae_mask = Variable(torch.LongTensor(
            pos_uae)).to(self.args.device)
        pos_uae_mask[pos_uae_mask != vocab2id['<pad>']] = -1
        pos_uae_mask[pos_uae_mask == vocab2id['<pad>']] = 0
        pos_uae_mask = -pos_uae_mask

        # pos_reg = [
        #     itm[:pos_reg_len]
        #     + [vocab2id['<pad>'] for _ in range(pos_reg_len-len(itm))]
        #     for itm in pos_reg]
        # pos_reg_var = Variable(torch.LongTensor(
        #     pos_reg)).to(self.args.device)
        # pos_reg_mask = Variable(torch.LongTensor(
        #     pos_reg)).to(self.args.device)
        # pos_reg_mask[pos_reg_mask != vocab2id['<pad>']] = -1
        # pos_reg_mask[pos_reg_mask == vocab2id['<pad>']] = 0
        # pos_reg_mask = -pos_reg_mask

        w2v_uae = []
        vocab_arr = list(vocab2id)
        for itm in pos_uae:
            random.shuffle(vocab_arr)
            out = [wd for wd in itm if wd != vocab2id['<pad>']]
            for j in range(len(vocab_arr)):
                if len(out) == pos_uae_len:
                    break
                if vocab_arr[j] not in out:
                    out.append(vocab2id[vocab_arr[j]])
            w2v_uae.append(out)
        w2v_uae_var = Variable(torch.LongTensor(
            w2v_uae)).to(self.args.device)
        w2v_uae_mask = pos_uae_mask.clone()
        w2v_uae_mask[w2v_uae_mask == 0] = -1.0
        w2v_uae_mask = -w2v_uae_mask

        # w2v_reg = []
        # vocab_arr = list(vocab2id)
        # for itm in pos_reg:
        #     random.shuffle(vocab_arr)
        #     out = [wd for wd in itm if wd != vocab2id['<pad>']]
        #     for j in range(len(vocab_arr)):
        #         if len(out) == pos_reg_len:
        #             break
        #         if vocab_arr[j] not in out:
        #             out.append(vocab2id[vocab_arr[j]])
        #     w2v_reg.append(out)
        # w2v_reg_var = Variable(torch.LongTensor(
        #     w2v_reg)).to(self.args.device)
        # w2v_reg_mask = pos_reg_mask.clone()
        # w2v_reg_mask[w2v_reg_mask == 0] = -1.0

        self.batch_data['sen_tokens'] = sen_tokens
        self.batch_data['pos_uae_var'] = pos_uae_var
        self.batch_data['pos_uae_mask'] = pos_uae_mask
        # self.batch_data['pos_reg_var'] = pos_reg_var
        # self.batch_data['pos_reg_mask'] = pos_reg_mask
        self.batch_data['w2v_uae_var'] = w2v_uae_var
        self.batch_data['w2v_uae_mask'] = w2v_uae_mask
        # self.batch_data['w2v_reg_var'] = w2v_reg_var
        # self.batch_data['w2v_reg_mask'] = w2v_reg_mask
        self.batch_data['label'] = labels

        if self.args.task[-5:] == 'train':
            neg_examples = []
            neg_uae = pos_uae
            # neg_reg = pos_reg
            for k in range(20):
                neg_uae = neg_uae[1:] + [neg_uae[0]]
                neg_uae_var = Variable(torch.LongTensor(
                    neg_uae)).to(self.args.device)
                neg_uae_mask = Variable(torch.LongTensor(
                    neg_uae)).to(self.args.device)
                neg_uae_mask[neg_uae_mask != vocab2id['<pad>']] = -1
                neg_uae_mask[neg_uae_mask == vocab2id['<pad>']] = 0
                neg_uae_mask = -neg_uae_mask

                # neg_reg = neg_reg[1:] + [neg_reg[0]]
                # neg_reg_var = Variable(torch.LongTensor(
                #     neg_reg)).to(self.args.device)
                # neg_reg_mask = Variable(torch.LongTensor(
                #     neg_reg)).to(self.args.device)
                # neg_reg_mask[neg_reg_mask != vocab2id['<pad>']] = -1
                # neg_reg_mask[neg_reg_mask == vocab2id['<pad>']] = 0
                # neg_reg_mask = -neg_reg_mask

                neg_examples.append([
                    neg_uae_var, neg_uae_mask])
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
