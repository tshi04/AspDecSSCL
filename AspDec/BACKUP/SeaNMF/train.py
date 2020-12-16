'''
SeaNMF Training
'''
import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from .model import NMF, SeaNMFL1
from .utils import read_docs, read_vocab


def train(args):

    data_dir = '../cluster_results'

    docs = read_docs(os.path.join(data_dir, args.file_term_doc))
    vocab = read_vocab(os.path.join(data_dir, args.file_vocab))
    n_docs = len(docs)
    n_terms = len(vocab)
    print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

    # non-negative matrix factorization.
    if args.model.lower() == 'nmf':
        print('read term doc matrix')
        dt_mat = np.zeros([n_terms, n_docs])
        for k in range(n_docs):
            for j in docs[k]:
                dt_mat[j, k] += 1.0
        print('term doc matrix done')
        print('-'*50)

        model = NMF(
            dt_mat,
            n_topic=args.n_topics,
            max_iter=args.max_iter,
            max_err=args.max_err)

        model.save_format(
            Wfile=os.path.join(data_dir, 'W.txt'),
            Hfile=os.path.join('H.txt'))

    # SeaNMF
    if args.model.lower() == 'seanmf':
        print('calculate co-occurance matrix')
        dt_mat = np.zeros([n_terms, n_terms])
        for itm in tqdm(docs):
            for kk in itm:
                for jj in itm:
                    dt_mat[int(kk), int(jj)] += 1.0
        print('co-occur done')
        print('-'*50)
        print('calculate PPMI')
        D1 = np.sum(dt_mat)
        SS = D1*dt_mat
        for k in range(n_terms):
            SS[k] /= np.sum(dt_mat[k])
        for k in range(n_terms):
            SS[:, k] /= np.sum(dt_mat[:, k])
        dt_mat = []  # release memory
        SS[SS == 0] = 1.0
        SS = np.log(SS)
        SS[SS < 0.0] = 0.0
        print('PPMI done')
        print('-'*50)

        print('read term doc matrix')
        dt_mat = np.zeros([n_terms, n_docs])
        for k in tqdm(range(n_docs)):
            for j in docs[k]:
                dt_mat[j, k] += 1.0
        print('term doc matrix done')
        print('-'*50)

        model = SeaNMFL1(
            dt_mat, SS,
            alpha=args.alpha,
            beta=args.beta,
            n_topic=args.n_topics,
            max_iter=args.max_iter,
            max_err=args.max_err,
            fix_seed=args.fix_seed)

        model.save_format(
            W1file=os.path.join(data_dir, 'W.txt'),
            W2file=os.path.join(data_dir, 'Wc.txt'),
            Hfile=os.path.join(data_dir, 'H.txt'))

        model.save_format(
            W1file=os.path.join(data_dir, 'aspect_weight.txt'),
            W2file=os.path.join(data_dir, 'Wc.txt'),
            Hfile=os.path.join(data_dir, 'H.txt'))
