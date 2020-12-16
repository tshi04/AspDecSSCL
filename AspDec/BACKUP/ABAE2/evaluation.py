'''
@author Tian Shi
Please contact tshi@vt.edu
'''
from sklearn.metrics import classification_report
import json
import os

import numpy as np
from tqdm import tqdm

from .utils import calculate_COH, calculate_PMI, read_docs, read_vocab


def eval_aspect_coherence(
        work_dir, file_term_doc, file_vocab, n_keywords,
        weight=[], file_aspect_weight='aspect_weight.txt'):

    docs = read_docs(os.path.join(work_dir, file_term_doc))
    vocab = read_vocab(os.path.join(work_dir, file_vocab))
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

    if weight == []:
        weight = np.loadtxt(os.path.join(
            work_dir, file_aspect_weight), dtype=float)
    else:
        weight = np.array(weight)
    n_topic = weight.shape[1]
    print('n_topic={}'.format(n_topic))

    PMI_arr = []
    COH_arr = []
    for k in tqdm(range(n_topic)):
        topKeywordsIndex = []
        for wd in weight[:, k].argsort()[::-1]:
            topKeywordsIndex.append(wd)
        topKeywordsIndex = topKeywordsIndex[:n_keywords]
        PMI_arr.append(calculate_PMI(dt_mat, topKeywordsIndex))
        COH_arr.append(calculate_COH(dt_mat, dt_vec, topKeywordsIndex))
    print('Average PMI={}'.format(np.average(np.array(PMI_arr))))
    print('Average COH={}'.format(np.average(np.array(COH_arr))))

    index = np.argsort(PMI_arr)

    for k in index:
        kw_idx = []
        for wd in np.argsort(weight[:, k])[::-1]:
            kw_idx.append(wd)
        kw_idx = kw_idx[:n_keywords]
        print('Topic {} [PMI={}, COH={}]: {}'.format(
            k+1, np.around(PMI_arr[k], 4), np.around(COH_arr[k], 4),
            ' '.join([vocab[w] for w in kw_idx])))


def evaluate_classification(args):

    aspect_label = []
    fp = open('../nats_results/aspect_mapping.txt', 'r')
    for line in fp:
        aspect_label.append(line.split()[1])
    fp.close()

    tmp = {wd: -1 for wd in aspect_label if wd != 'none'}
    label = {}
    for k, wd in enumerate(tmp):
        label[wd] = k

    fp = open(os.path.join('../nats_results', args.file_output), 'r')
    pred = []
    gold = []
    for line in fp:
        itm = json.loads(line)
        arr = np.argsort(itm['aspect_weight'])[::-1]
        asp_arr = [aspect_label[k] for k in arr if aspect_label[k] != 'none']
        if asp_arr[0] != itm['gold_label']:
            print(' '.join(itm['text']))
            print(arr)
            print(asp_arr, itm['gold_label'])
            print()
        for j, k in enumerate(arr):
            if aspect_label[k] != 'none':
                pp = aspect_label[k]
                break
        pred.append(label[pp])
        gold.append(label[itm['gold_label']])
    fp.close()
    print(classification_report(gold, pred, target_names=list(label)))
