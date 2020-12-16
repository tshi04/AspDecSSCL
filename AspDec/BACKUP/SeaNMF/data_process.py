'''
Visualize Topics
'''
import argparse
import json
import os

from tqdm import tqdm

from LeafNATS.utils.stopwords import get_stopwords_en

stopwords = get_stopwords_en()

def data_process(args):
    
    out_dir = '../cluster_results'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # create vocabulary
    print('create vocab')
    vocab = {}
    fp = open(
        os.path.join(args.data_dir, args.file_corpus), 'r')
    for line in tqdm(fp):
        itm = json.loads(line)['text_fine']
        for wd in itm:
            if wd in stopwords or len(wd) < 2:
                continue
            try:
                vocab[wd] += 1
            except:
                vocab[wd] = 1
    fp.close()
    vocab_arr = [[wd, vocab[wd]]
                for wd in vocab if vocab[wd] > args.vocab_min_count]
    vocab_arr = sorted(vocab_arr, key=lambda k: k[1])[::-1]
    vocab_arr = vocab_arr[:args.vocab_max_size]
    vocab_arr = sorted(vocab_arr)
    print('vocabulary size = {}.'.format(len(vocab_arr)))

    fout = open(os.path.join(out_dir, args.file_vocab), 'w')
    for itm in vocab_arr:
        itm[1] = str(itm[1])
        fout.write(' '.join(itm)+'\n')
    fout.close()

    # vocabulary to id
    vocab2id = {itm[1][0]: itm[0] for itm in enumerate(vocab_arr)}
    print('create document term matrix')
    data_arr = []
    fp = open(os.path.join(args.data_dir, args.file_corpus), 'r')
    fout = open(os.path.join(out_dir, args.file_term_doc), 'w')
    cnt = 0
    for line in tqdm(fp):
        itm = json.loads(line)['text_fine']
        itm = [str(vocab2id[wd]) for wd in itm if wd in vocab2id]
        if len(itm) < args.doc_min_len or len(itm) > args.doc_max_len:
            continue
        itm = ' '.join(itm)
        fout.write(itm+'\n')
        cnt += 1
    fp.close()
    fout.close()
    print('Number of documents = {}.'.format(cnt))
