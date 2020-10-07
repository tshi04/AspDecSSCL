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

def run_bow_classfication(args):
    '''
    BOW classifier
    '''
    asp2id = []
    fp = open(os.path.join('../nats_results', 'aspect_mapping.txt'), 'r')
    for k, line in enumerate(fp):
        itm = line.split()
        asp2id.append(itm[1])
    fp.close()

    word2asp = {}
    fp = open(os.path.join('../nats_results', args.file_keywords_dump), 'r')
    for k, line in enumerate(fp):
        itm = line.split()
        for wd in itm:
            try:
                word2asp[wd].append(k)
            except:
                word2asp[wd] = [k]
    fp.close()
    
    test = []
    fp = open(os.path.join(args.data_dir, args.file_test), 'r')
    for line in tqdm(fp):
        itm = json.loads(line)
        text = itm['text_uae'].split()
        text2asp = [0 for _ in range(len(asp2id))]
        for wd in text:
            try:
                for asp in word2asp[wd]:
                    text2asp[asp] += 1
            except:
                continue
        print(text2asp)
        out = {}
        out['text'] = text
        out['aspect_weight'] = text2asp
        out['gold_label'] = itm['label'].lower()
        test.append(out)
    fp.close()

    fout = open(os.path.join('../nats_results', args.file_output), 'w')
    for itm in test:
        json.dump(itm, fout)
        fout.write('\n')
    fout.close()


    

