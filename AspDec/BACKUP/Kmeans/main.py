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

parser.add_argument('--data_dir', default='../data/restaurant',
                    help='data directory.')
parser.add_argument('--work_dir', default='../cluster_results',
                    help='work directory.')
parser.add_argument('--file_corpus', default='train.txt',
                    help='file used for corpus')
parser.add_argument('--file_train', default='vectors_w2v.npy',
                    help='file word vectors')
parser.add_argument('--file_aspect', default='aspect_weight.txt',
                    help='file aspect weights')
parser.add_argument('--file_aspect_label', default='aspect_label.txt',
                    help='file aspect weights')
parser.add_argument('--file_centroid', default='aspect_centroid.txt',
                    help='file aspect centroid')
parser.add_argument('--file_vocab', default='vocab_w2v',
                    help='file vocabulary')
parser.add_argument('--file_vocab_out', default='vocab.txt',
                    help='file vocab output')
parser.add_argument('--file_term_doc', default='doc_term_mat.txt',
                    help='term document matrix file')

parser.add_argument('--cluster_seeds', type=int, default=2,
                    help='seeds cluster.')
parser.add_argument('--n_clusters', type=int, default=14,
                    help='number of clusters.')
parser.add_argument('--n_keywords', type=int, default=10,
                    help='number of keywords in each topic.')

args = parser.parse_args()

if args.task == 'train':
    from .model import run_kmeans
    from .vis_topic import visualization
    from .data_process import data_process

    data_process(args)
    run_kmeans(args)
    visualization(args)

if args.task == 'viz':
    from .vis_topic import visualization
    visualization(args)
