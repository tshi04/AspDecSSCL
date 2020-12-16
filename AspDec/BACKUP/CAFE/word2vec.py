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

    fp = open(os.path.join(args.data_dir, args.file_train), 'r')
    sentences = []
    for line in tqdm(fp):
        itm = json.loads(line)
        sentences.append(itm['text_fine'])
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
    # model.train(sentences, total_examples=len(sentences), epochs=4)
    model.save(os.path.join(cluster_dir, 'w2v_embedding'))
    print('Taining Done.')
    print('-'*50)


def convert_vectors(args):
    '''
    convert vectors and vocab.
    '''
    cluster_dir = '../cluster_results'
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

    vec = np.zeros([len(lexicon), 200])
    for k, wd in enumerate(vocab):
        vec[k] = lexicon[wd]

    print('Vocabulary size: {}'.format(vec.shape[0]))

    np.save(os.path.join(cluster_dir, 'word2vec_vec'), vec)
    fout = open(os.path.join(cluster_dir, 'vocab'), 'w')
    for k, itm in enumerate(vocab):
        itm = [itm, str(k)]
        fout.write(' '.join(itm) + '\n')
    fout.close()
