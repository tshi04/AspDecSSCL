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
    
    # vocabulary
    print('Vocabulary')
    fp = open(os.path.join(args.work_dir, args.file_vocab), 'r')
    fout = open(os.path.join(args.work_dir, args.file_vocab_out), 'w')
    vocab2id = {}
    for line in fp:
        itm = line.split()
        vocab2id[itm[0]] = itm[1]
        fout.write(line)
    fout.close()
    fp.close()

    print('create document term matrix')
    data_arr = []
    fp = open(os.path.join(args.data_dir, args.file_corpus), 'r')
    fout = open(os.path.join(args.work_dir, args.file_term_doc), 'w')
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
