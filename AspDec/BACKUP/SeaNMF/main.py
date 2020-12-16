'''
SeaNMF Training
'''
import argparse
import os
import time

parser = argparse.ArgumentParser()

parser.add_argument('--task', default='train',
                    help='train | viz')

parser.add_argument('--data_dir', default='../data/restaurant/',
                    help='directory that store the data.')
parser.add_argument('--file_corpus', default='train.txt',
                    help='input text file')
parser.add_argument('--file_term_doc', default='doc_term_mat.txt',
                    help='term document matrix file')
parser.add_argument('--file_vocab', default='vocab.txt',
                    help='vocab file')

parser.add_argument('--vocab_max_size', type=int, default=10000,
                    help='maximum vocabulary size')
parser.add_argument('--vocab_min_count', type=int, default=10,
                    help='minimum frequency of the words')
parser.add_argument('--doc_min_len', type=int, default=3,
                    help='minimum frequency of the words')
parser.add_argument('--doc_max_len', type=int, default=50,
                    help='minimum frequency of the words')

parser.add_argument('--model', default='seanmf', help='nmf | seanmf')
parser.add_argument('--max_iter', type=int, default=500,
                    help='max number of iterations')
parser.add_argument('--n_topics', type=int, default=14,
                    help='number of topics')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
parser.add_argument('--beta', type=float, default=0.01, help='beta')
parser.add_argument('--max_err', type=float, default=0.1,
                    help='stop criterion')
parser.add_argument('--fix_seed', type=bool, default=True,
                    help='set random seed 0')
parser.add_argument('--n_keywords', type=int, default=10,
                    help='number of keywords in each topic.')
args = parser.parse_args()

if args.task == 'train':
    from .data_process import data_process
    from .train import train
    from .vis_topic import visualization

    data_process(args)
    train(args)
    visualization(args)

if args.task == 'viz':
    from .vis_topic import visualization
    visualization(args)
