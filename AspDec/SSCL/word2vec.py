'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os
import random

import gensim
import numpy as np
from tqdm import tqdm


def run_word2vec(args):
    '''
    Run word2vec.
    '''
    cluster_dir = '../cluster_results'
    if not os.path.exists(cluster_dir):
        os.mkdir(cluster_dir)
    if not os.path.exists('../nats_results'):
        os.mkdir('../nats_results')

    fp = open(os.path.join(args.data_dir, args.file_train_w2v), 'r')
    sentences = []
    for line in tqdm(fp):
        itm = json.loads(line)
        sentences.append(itm['text_uae'].split())
    fp.close()
    random.shuffle(sentences)
    print('-'*50)
    print('Number of sentences: {}'.format(len(sentences)))
    print('Begin to train word2vec...')
    model = gensim.models.Word2Vec(
        sentences,
        size=args.emb_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers)
    model.save(os.path.join(cluster_dir, 'w2v_embedding'))
    print('Taining Done.')
    print('-'*50)


def convert_vectors(args):
    '''
    convert vectors and vocab.
    '''
    cluster_dir = '../cluster_results'
    file_vocab = 'vocab.txt'
    file_wordvec = 'vectors_w2v'

    model = gensim.models.Word2Vec.load(
        os.path.join(cluster_dir, 'w2v_embedding'))

    lexicon = {}
    for word in model.wv.vocab:
        if word.strip() == '':
            continue
        lexicon[word] = model.wv[word]
    vocab = []
    for wd in lexicon:
        vocab.append(wd)
    vocab = sorted(vocab)

    vec = np.zeros([len(lexicon), args.emb_size])
    for k, wd in enumerate(vocab):
        vec[k] = lexicon[wd]

    print('Vocabulary size: {}'.format(vec.shape[0]))

    np.save(os.path.join(cluster_dir, file_wordvec), vec)
    fout = open(os.path.join(cluster_dir, file_vocab), 'w')
    for k, itm in enumerate(vocab):
        itm = [itm, str(k)]
        fout.write(' '.join(itm) + '\n')
    fout.close()
