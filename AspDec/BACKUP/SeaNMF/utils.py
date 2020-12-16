import re

import numpy as np
from tqdm import tqdm


def read_docs(file_name):
    print('read documents')
    print('-'*50)
    docs = []
    fp = open(file_name, 'r')
    for line in tqdm(fp):
        arr = re.split('\s', line[:-1])
        arr = filter(None, arr)
        arr = [int(idx) for idx in arr]
        docs.append(arr)
    fp.close()

    return docs


def read_vocab(file_name):
    print('read vocabulary')
    print('-'*50)
    vocab = []
    fp = open(file_name, 'r')
    for line in tqdm(fp):
        arr = re.split('\s', line[:-1])
        vocab.append(arr[0])
    fp.close()

    return vocab


def calculate_PMI(AA, topKeywordsIndex):
    '''
    PMI evaluation metrics.
    SeaNMF
    '''
    D1 = np.sum(AA)
    n_tp = len(topKeywordsIndex)
    PMI = []
    for index1 in topKeywordsIndex:
        for index2 in topKeywordsIndex:
            if index2 < index1:
                if AA[index1, index2] == 0:
                    PMI.append(0.0)
                else:
                    C1 = np.sum(AA[index1])
                    C2 = np.sum(AA[index2])
                    PMI.append(np.log(AA[index1, index2]*D1/C1/C2))
    avg_PMI = 2.0*np.sum(PMI)/float(n_tp)/(float(n_tp)-1.0)

    return avg_PMI


def calculate_COH(dt_mat, dt_vec, topKeywordsIndex):
    '''
    Coherence evaluation metrics.
    An Unsupervised Neural Attention Model for Aspect Extraction
    '''
    tki = topKeywordsIndex
    n_tp = len(tki)
    COH = 0
    for n in range(1, n_tp):
        for l in range(n):
            d1 = dt_vec[tki[l]]
            d2 = dt_mat[tki[n], tki[l]]
            COH += np.log((d2+1)/d1)

    return COH
