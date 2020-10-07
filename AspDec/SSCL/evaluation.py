'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os

import numpy as np
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from .utils import calculate_COH, calculate_PMI, read_docs, read_vocab


def eval_aspect_coherence(
        work_dir='../cluster_results', 
        file_term_doc='doc_term_mat.txt',
        file_vocab='vocab.txt',
        n_keywords=10, weight=[], 
        file_aspect_weight='aspect_weight.txt'):

    docs = read_docs(os.path.join(work_dir, file_term_doc))
    vocab = read_vocab(os.path.join(work_dir, file_vocab))
    n_docs = len(docs)
    n_terms = len(vocab)
    print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

    dt_mat = np.zeros([n_terms, n_terms])
    dt_vec = np.zeros(n_terms)
    for k, itm in tqdm(enumerate(docs)):
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


def evaluate_sscl_classification(args):
    '''
    Evaluate SSCL Classification.
    '''
    aspect_label = []
    fp = open('../nats_results/aspect_mapping.txt', 'r')
    for line in fp:
        aspect_label.append(line.split()[1])
    fp.close()

    ignore_type = ['nomap']
    if not args.none_type:
        ignore_type.append('none')

    tmp = {wd: -1 for wd in aspect_label if not wd in  ignore_type}
    label = {}
    for k, wd in enumerate(sorted(list(tmp))):
        label[wd] = k

    fp = open(os.path.join('../nats_results', args.file_output), 'r')
    pred = []
    gold = []
    for line in fp:
        itm = json.loads(line)
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
    fp.close()

    print(classification_report(
        gold, pred, target_names=list(label), digits=3))


def evaluate_ts_classification(args):
    '''
    Evaluate Teacher-Student Model
    '''
    asp_labels = []
    pred = []
    gold = []
    fp = open(os.path.join('../nats_results', args.file_output), 'r')
    for line in fp:
        itm = json.loads(line)
        if itm['pred_label'] in itm['gold_label'].split(','):
            pred.append(itm['pred_label'])
            gold.append(itm['pred_label'])
        else:
            pred.append(itm['pred_label'])
            gold.append(itm['gold_label'].split(',')[0])
        for wd in itm['gold_label'].split(','):
            asp_labels.append(wd)
    fp.close()

    asp_labels = sorted(list(set(asp_labels)))
    asp_map = {wd: k for k, wd in enumerate(asp_labels)}

    pred = [asp_map[wd] for wd in pred]
    gold = [asp_map[wd] for wd in gold]

    print(classification_report(
        gold, pred, target_names=asp_labels, digits=3))
