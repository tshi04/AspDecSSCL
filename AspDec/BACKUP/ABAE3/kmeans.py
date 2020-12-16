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


def run_kmeans(args):
    '''
    Run kmeans.
    '''
    emb_matrix = np.load(os.path.join(
        args.cluster_dir, args.file_wordvec + '.npy'))

    fp = open(os.path.join(
        args.cluster_dir, args.file_vocab), 'r')
    vocab2id = {}
    for line in fp:
        itm = line.split()
        vocab2id[itm[0]] = int(itm[1])
    fp.close()
    
    kmeans_init = np.zeros([args.n_clusters, args.emb_size])
    tok_init = [
        'beef', 'banana', 'wine', 'pepper', 'cuisine', 
        'wall', 'space', 
        'waiter', 'rude',
        'bill', 'birthday', 
        'street', 'excellent', 'think']
    for k, wd in enumerate(tok_init):
        kmeans_init[k] = emb_matrix[vocab2id[wd]]

    print('run kmeans...')
    try:
        km = KMeans(
            n_clusters=args.n_clusters,
            init=kmeans_init,
            random_state=args.kmeans_seeds)
    except:
        km = KMeans(
            n_clusters=args.n_clusters,
            random_state=args.kmeans_seeds)
    km.fit(emb_matrix)
    clusters = km.cluster_centers_
    print('Kmeans done.')

    np.savetxt(os.path.join(
        args.cluster_dir, args.file_kmeans_centroid), clusters)

    weight = np.matmul(emb_matrix, clusters.transpose())
    weight = np.array(weight)
    np.savetxt(os.path.join(
        args.cluster_dir, 'aspect_weight.txt'), weight)
    
    if not os.path.exists('../nats_results'):
        os.mkdir('../nats_results')
    fout = open(os.path.join(
        '../nats_results', 'aspect_mapping.txt'), 'w')
    for k in range(args.n_clusters):
        fout.write('{} {}\n'.format(k+1, 'nomap'))
    fout.close()


def get_cluster_keywords(args):
    
    vocab_ = []
    fp = open(os.path.join(args.cluster_dir, args.file_vocab), 'r')
    for line in fp:
        wd = line.split()[0]
        vocab_.append(wd)
    fp.close()

    weight_ = np.loadtxt(os.path.join(
        args.cluster_dir, 'aspect_weight.txt'), dtype=float)

    print('-'*50)
    for k in range(weight_.shape[1]):
        idx = np.argsort(weight_[:, k])[::-1][:args.n_keywords]
        kw = ' '.join([vocab_[j] for j in idx])
        print('{}: {}'.format(k+1, kw))


def data_process(args):

    # vocabulary
    print('Vocabulary')
    fp = open(os.path.join(
        args.cluster_dir, args.file_vocab), 'r')
    vocab2id = {}
    for line in fp:
        itm = line.split()
        vocab2id[itm[0]] = itm[1]
    fp.close()

    print('create document term matrix')
    data_arr = []
    fp = open(os.path.join(
        args.data_dir, args.file_train), 'r')
    fout = open(os.path.join(
        args.cluster_dir, args.file_term_doc), 'w')
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
