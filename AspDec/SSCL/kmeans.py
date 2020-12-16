'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os

import numpy as np
from sklearn.cluster import KMeans


def run_kmeans(args):
    '''
    Run kmeans.
    '''
    cluster_dir = '../cluster_results'
    file_wordvec = 'vectors_w2v'
    file_vocab = 'vocab.txt'
    file_kmeans_centroid = 'aspect_centroid.txt'
    nats_results = '../nats_results'
    file_keywords = 'kmeans_aspect_keywords.txt'
    file_asp_map = 'aspect_mapping.txt'
    file_asp_wt = 'aspect_weight.txt'

    emb_matrix = np.load(os.path.join(
        cluster_dir, file_wordvec + '.npy'))

    fp = open(os.path.join(
        cluster_dir, file_vocab), 'r')
    vocab2id = {}
    for line in fp:
        itm = line.split()
        vocab2id[itm[0]] = int(itm[1])
    fp.close()

    print('run kmeans...')
    if args.kmeans_init == 'vanilla':
        km = KMeans(
            n_clusters=args.n_clusters,
            random_state=args.kmeans_seeds)
    else:
        fp = open(os.path.join(args.data_dir, args.kmeans_init), 'r')
        init_ = []
        for line in fp:
            init_.append(line.strip())
        fp.close()
        assert args.n_clusters == len(init_)
        init_vec = np.zeros([args.n_clusters, args.emb_size])
        for k, wd in enumerate(init_):
            init_vec[k] = emb_matrix[vocab2id[wd]]

        km = KMeans(
            n_clusters=args.n_clusters,
            init=init_vec, n_init=1,
            random_state=args.kmeans_seeds)

    km.fit(emb_matrix)
    clusters = km.cluster_centers_
    print('Kmeans done.')

    np.savetxt(os.path.join(
        cluster_dir, file_kmeans_centroid), clusters)

    weight = np.matmul(emb_matrix, clusters.transpose())
    weight = np.array(weight)
    np.savetxt(os.path.join(cluster_dir, file_asp_wt), weight)

    if not os.path.exists(nats_results):
        os.mkdir(nats_results)
    fout = open(os.path.join(
        nats_results, file_asp_map), 'w')
    for k in range(args.n_clusters):
        fout.write('{} {}\n'.format(k+1, 'nomap'))
    fout.close()

    id2vocab = {vocab2id[wd]: wd for wd in vocab2id}
    output = [[] for _ in range(args.n_clusters)]
    for k, lb in enumerate(km.labels_):
        output[lb].append(id2vocab[k])
    fout = open(os.path.join(cluster_dir, file_keywords), 'w')
    for itm in output:
        fout.write('{}\n'.format(' '.join(itm)))
    fout.close()


def get_cluster_keywords(args):
    '''
    Get keywords for kmeans
    '''
    cluster_dir = '../cluster_results'
    file_vocab = 'vocab.txt'
    file_asp_wt = 'aspect_weight.txt'

    vocab_ = []
    fp = open(os.path.join(cluster_dir, file_vocab), 'r')
    for line in fp:
        wd = line.split()[0]
        vocab_.append(wd)
    fp.close()

    weight_ = np.loadtxt(os.path.join(
        cluster_dir, file_asp_wt), dtype=float)

    print('-'*50)
    for k in range(weight_.shape[1]):
        idx = np.argsort(weight_[:, k])[::-1][:args.n_keywords]
        kw = ' '.join([vocab_[j] for j in idx])
        print('{}: {}'.format(k+1, kw))
