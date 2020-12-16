'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from .utils import calculate_COH, calculate_PMI, read_docs, read_vocab


def counter(input_):

    vocab = {'NOUN': 0, 'ADJ': 0}
    for wd in input_:
        try:
            vocab[wd] += 1
        except:
            continue

    return vocab


def run_kmeans(args):
    '''
    Run kmeans.
    '''
    cluster_dir = '../cluster_results'
    emb_matrix = np.load(os.path.join(cluster_dir, 'word2vec_vec.npy'))

    fp = open(os.path.join(
        args.data_dir, args.file_train), 'r')
    vocab2pos = {}
    for line in tqdm(fp):
        itm = json.loads(line)
        for k, wd in enumerate(itm['text_fine']):
            try:
                vocab2pos[wd].append(itm['text_pos'][k])
            except:
                vocab2pos[wd] = [itm['text_pos'][k]]
    fp.close()

    for wd in vocab2pos:
        vocab2pos[wd] = counter(vocab2pos[wd])

    vocab_noun = []
    vocab_adj = []
    fp = open(os.path.join(cluster_dir, 'vocab'), 'r')
    for line in fp:
        [wd, idx] = line.split()
        if vocab2pos[wd]['NOUN'] > 10:
            vocab_noun.append([wd, int(idx)])
        if vocab2pos[wd]['ADJ'] > 10:
            vocab_adj.append([wd, int(idx)])
    fp.close()
    print('vocab size: NOUN {}, ADJ {}.'.format(
        len(vocab_noun), len(vocab_adj)))

    mat_noun = np.zeros([len(vocab_noun), emb_matrix.shape[1]])
    mat_adj = np.zeros([len(vocab_adj), emb_matrix.shape[1]])

    for k, itm in enumerate(vocab_noun):
        mat_noun[k] = emb_matrix[itm[1]]
    for k, itm in enumerate(vocab_adj):
        mat_adj[k] = emb_matrix[itm[1]]

    [n_noun, n_adj] = [int(wd) for wd in args.n_clusters.split(',')]
    n_clusters = n_noun + n_adj

    print('run kmeans for noun.')
    km_noun = KMeans(n_clusters=n_noun, random_state=args.kmeans_seeds)
    km_noun.fit(mat_noun)
    noun_clusters = km_noun.cluster_centers_
    print('run kmeans for adj.')
    km_adj = KMeans(n_clusters=n_adj, random_state=args.kmeans_seeds)
    km_adj.fit(mat_adj)
    adj_clusters = km_adj.cluster_centers_

    np.savetxt(os.path.join(cluster_dir, 'kmc_noun.txt'), noun_clusters)
    np.savetxt(os.path.join(cluster_dir, 'kmc_adj.txt'), adj_clusters)

    fout = open(os.path.join(cluster_dir, 'vocab_noun'), 'w')
    for itm in vocab_noun:
        fout.write('{} {}\n'.format(itm[0], itm[1]))
    fout.close()
    fout = open(os.path.join(cluster_dir, 'vocab_adj'), 'w')
    for itm in vocab_adj:
        fout.write('{} {}\n'.format(itm[0], itm[1]))
    fout.close()

    w_noun = np.matmul(mat_noun, noun_clusters.transpose())
    w_noun = np.array(w_noun)
    np.savetxt(os.path.join(cluster_dir, 'aspect_noun.txt'), w_noun)
    w_adj = np.matmul(mat_adj, adj_clusters.transpose())
    w_adj = np.array(w_adj)
    np.savetxt(os.path.join(cluster_dir, 'aspect_adj.txt'), w_adj)

    if not os.path.exists('../nats_results'):
        os.mkdir('../nats_results')
    fout = open(os.path.join('../nats_results', 'aspect_noun.txt'), 'w')
    for k in range(n_noun):
        fout.write('{} {}\n'.format(k+1, 'none'))
    fout.close()
    fout = open(os.path.join('../nats_results', 'aspect_adj.txt'), 'w')
    for k in range(n_adj):
        fout.write('{} {}\n'.format(k+1, 'none'))
    fout.close()


def get_cluster_keywords(args):

    cluster_dir = '../cluster_results'
    for aspect in ['noun', 'adj']:
        vocab_ = []
        fp = open(os.path.join(cluster_dir, 'vocab_{}'.format(aspect)), 'r')
        for line in fp:
            wd = line.split()[0]
            vocab_.append(wd)
        fp.close()

        weight_ = np.loadtxt(os.path.join(
            cluster_dir, 'aspect_{}.txt'.format(aspect)), dtype=float)

        print('-'*50)
        print(aspect)
        for k in range(weight_.shape[1]):
            idx = np.argsort(weight_[:, k])[::-1][:args.n_keywords]
            kw = [vocab_[j] for j in idx]
            print('{}: {}'.format(k+1, kw))


def data_process(args):

    cluster_dir = '../cluster_results'
    # vocabulary
    print('Vocabulary')
    fp = open(os.path.join(cluster_dir, 'vocab'), 'r')
    vocab2id = {}
    for line in fp:
        itm = line.split()
        vocab2id[itm[0]] = itm[1]
    fp.close()

    print('create document term matrix')
    data_arr = []
    fp = open(os.path.join(
        args.data_dir, args.file_train), 'r')
    fout = open(os.path.join(cluster_dir, 'doc_term_mat.txt'), 'w')
    cnt = 0
    for line in tqdm(fp):
        itm = json.loads(line)['text_fine']
        itm = [str(vocab2id[wd]) for wd in itm if wd in vocab2id]
        itm = ' '.join(itm)
        fout.write(itm+'\n')
        cnt += 1
    fp.close()
    fout.close()
    print('Number of documents = {}.'.format(cnt))
