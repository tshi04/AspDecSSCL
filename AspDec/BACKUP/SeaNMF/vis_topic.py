'''
Visualize Topics
'''
import os

import numpy as np
from tqdm import tqdm

from .utils import calculate_COH, calculate_PMI, read_docs, read_vocab


def visualization(args):

    data_dir = '../cluster_results'

    docs = read_docs(os.path.join(data_dir, args.file_term_doc))
    vocab = read_vocab(os.path.join(data_dir, args.file_vocab))
    n_docs = len(docs)
    n_terms = len(vocab)
    print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

    dt_mat = np.zeros([n_terms, n_terms])
    dt_vec = np.zeros(n_terms)
    for itm in tqdm(docs):
        for kk in set(itm):
            for jj in set(itm):
                if kk != jj:
                    dt_mat[int(kk), int(jj)] += 1.0
            dt_vec[int(kk)] += 1.0
    print('co-occur done')

    W = np.loadtxt(os.path.join(data_dir, 'W.txt'), dtype=float)
    n_topic = W.shape[1]
    print('n_topic={}'.format(n_topic))

    PMI_arr = []
    COH_arr = []
    for k in tqdm(range(n_topic)):
        topKeywordsIndex = W[:, k].argsort()[::-1][:args.n_keywords]
        PMI_arr.append(calculate_PMI(dt_mat, topKeywordsIndex))
        COH_arr.append(calculate_COH(dt_mat, dt_vec, topKeywordsIndex))
    print('Average PMI={}'.format(np.average(np.array(PMI_arr))))
    print('Average COH={}'.format(np.average(np.array(COH_arr))))

    index = np.argsort(PMI_arr)

    for k in index:
        kw_idx = np.argsort(W[:, k])[::-1][:args.n_keywords]
        print('Topic {} [PMI={}, COH={}]: {}'.format(
            k+1, np.around(PMI_arr[k], 4), np.around(COH_arr[k], 4),
            ' '.join([vocab[w] for w in kw_idx])))
