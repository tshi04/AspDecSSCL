'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import argparse

from LeafNATS.utils.utils import str2bool

parser = argparse.ArgumentParser()
'''
Use in the framework and cannot remove.
'''
parser.add_argument('--task', default='train',
                    help='train | ')

parser.add_argument('--data_dir', default='../data/restaurant/',
                    help='directory that store the data.')
parser.add_argument('--file_train', default='train.txt',
                    help='Training')

parser.add_argument('--emb_size', type=int, default=200,
                    help='embedding size.')
parser.add_argument('--window', type=int, default=10,
                    help='window size')
parser.add_argument('--min_count', type=int, default=5,
                    help='words min count')
parser.add_argument('--workers', type=int, default=8,
                    help='number of workers')

args = parser.parse_args()

if args.task == 'train':
    from .model import run_word2vec
    from .model import convert_vectors

    run_word2vec(args)
    convert_vectors(args)
