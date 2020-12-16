'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def run_kmeans(args):
    '''
    Run kmeans.
    '''
    emb_matrix = np.load(os.path.join(
        args.work_dir, args.file_train))

    print('run kmeans...')
    km = KMeans(n_clusters=args.n_clusters, random_state=args.cluster_seeds)
    km.fit(emb_matrix)
    clusters = km.cluster_centers_

    print('save weights')
    # norm = np.linalg.norm(emb_matrix, ord=2, axis=1, keepdims=True)
    # emb_matrix = emb_matrix/(norm+1e-20)
    # norm = np.linalg.norm(clusters, ord=2, axis=1, keepdims=True)
    # clusters = clusters/(norm+1e-20)
    weight = np.matmul(emb_matrix, clusters.transpose())
    weight = np.array(weight)

    np.savetxt(os.path.join(args.work_dir, args.file_aspect), weight)
    np.savetxt(os.path.join(args.work_dir, args.file_centroid), clusters)
