'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def run_tfidf_seeds(args):
    '''
    tfidf for seed words.
    '''
    data = []
    corpus = []
    labels_arr = []
    fp = open(os.path.join(args.data_dir, args.file_test), 'r')
    for line in fp:
        itm = json.loads(line)
        data.append(itm)
        corpus.append(itm['text_uae'])
        for wd in itm['label'].lower().split(','):
            labels_arr.append(wd)
    fp.close()
    labels_arr = sorted(list(set(labels_arr)))
    labels = {wd: k for k, wd in enumerate(labels_arr)}

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    vocab_arr = vectorizer.get_feature_names()
    vocab = {wd: k for k, wd in enumerate(vocab_arr)}

    x_words = np.sum(X, axis=0)/np.sum(X)

    score_mat = np.zeros([len(labels), len(vocab)])
    for k, itm in tqdm(enumerate(data)):
        for lb in itm['label'].split(','):
            lb = labels[lb.lower()]
            score_mat[lb] += X[k]
    score_mat = score_mat / np.linalg.norm(
        score_mat, ord=1, axis=-1, keepdims=True)
    score_mat = np.multiply(
        score_mat, np.log(score_mat/x_words + 1e-20))

    for k in range(score_mat.shape[0]):
        vec = score_mat[k]
        idx = np.argsort(vec).tolist()[0][::-1]
        print([vocab_arr[i] for i in idx][:args.n_keywords])
