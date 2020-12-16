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

    print('run kmeans...')
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
        fout.write('{} {}\n'.format(k+1, 'none'))
    fout.close()


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
