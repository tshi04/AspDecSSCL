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
                    help='train | test | evaluate | word2vec \
                        | kmeans | kmeans_eval | term_doc_mat')

parser.add_argument('--data_dir', default='../data/restaurant',
                    help='directory that store the data.')
parser.add_argument('--file_train', default='train.txt', help='train')
parser.add_argument('--file_dev', default='dev.txt', help='development')
parser.add_argument('--file_test', default='test.txt', help='test')
parser.add_argument('--file_output', default='test_output.json',
                    help='test output file')

parser.add_argument('--base_model_dir', default='../nats_results/sscl_models',
                    help='base model dir')

parser.add_argument('--n_epoch', type=int, default=1,
                    help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size.')
parser.add_argument('--checkpoint', type=int, default=300,
                    help='How often you want to save model?')

parser.add_argument('--continue_training', type=str2bool,
                    default=True, help='Do you want to continue?')
parser.add_argument('--train_base_model', type=str2bool, default=False,
                    help='True: Use Pretrained Param | False: Transfer Learning')
parser.add_argument('--use_move_avg', type=str2bool, default=False,
                    help='move average')
parser.add_argument('--use_optimal_model', type=str2bool, default=True,
                    help='Do you want to use the best model?')
parser.add_argument('--model_optimal_key', default='0,0', help='epoch,batch')

parser.add_argument('--is_lower', type=str2bool, default=True,
                    help='convert all tokens to lower case?')
# learning rate and scheduler
parser.add_argument('--lr_schedule', default='warm-up',
                    help='Schedule learning rate. | build-in | warm-up | None')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0,
                    help='clip the gradient norm.')
parser.add_argument('--step_size', type=int, default=2,
                    help='step size')
parser.add_argument('--step_decay', type=float, default=0.8,
                    help='learning rate decay')
parser.add_argument('--warmup_step', type=int, default=2000,
                    help='warmup step size')
parser.add_argument('--model_size', type=int, default=100000,
                    help='model size')
'''
User specified parameters.
'''
parser.add_argument('--device', default="cuda:0", help='device')
# sscl
parser.add_argument('--distance', default='cosine',
                    help='cosine')
parser.add_argument('--emb_size', type=int, default=128,
                    help='embedding size')
parser.add_argument('--max_seq_len', type=int, default=30,
                    help='sentence max length.')
parser.add_argument('--min_seq_len', type=int, default=3,
                    help='sentence min length.')
parser.add_argument('--smooth_factor', type=float, default=0.5,
                    help='smooth factor.')
# word2vec
parser.add_argument('--file_train_w2v', default='train_w2v.txt',
                    help='Training')
parser.add_argument('--window', type=int, default=5,
                    help='window size')
parser.add_argument('--min_count', type=int, default=10,
                    help='words min count')
parser.add_argument('--workers', type=int, default=8,
                    help='number of workers')
# kmeans
parser.add_argument('--kmeans_init', default='vanilla',
                    help='initial centroids. vanilla | kmeans_init.txt')
parser.add_argument('--kmeans_seeds', type=int, default=0,
                    help='kmeans seeds.')
parser.add_argument('--n_clusters', type=int, default=30,
                    help='number of clusters.')
parser.add_argument('--n_keywords', type=int, default=10,
                    help='number of keywords.')
# doc term matrix
parser.add_argument('--file_train_doc_term', default='train_w2v.txt',
                    help='Training')
# bert simple teacher student
parser.add_argument('--pretrained_model', default='bert',
                    help='Use pretrained model name')
parser.add_argument('--thresh_aspect', type=float, default=1.4,
                    help='aspect threshold')
parser.add_argument('--thresh_general', type=float, default=0.7,
                    help='general threshold')
parser.add_argument('--none_type', type=str2bool, default=False,
                    help='consider none type in evaluation.')

args = parser.parse_args()


if args.task == 'word2vec':
    from .word2vec import run_word2vec
    from .word2vec import convert_vectors
    run_word2vec(args)
    convert_vectors(args)

if args.task[:6] == 'kmeans':
    if args.task == 'kmeans':
        from .kmeans import run_kmeans
        from .kmeans import get_cluster_keywords
        run_kmeans(args)
        get_cluster_keywords(args)
    if args.task == 'kmeans_keywords':
        from .kmeans import get_cluster_keywords
        get_cluster_keywords(args)
    if args.task == 'kmeans_evaluate':
        from .evaluation import eval_aspect_coherence
        eval_aspect_coherence(n_keywords=args.n_keywords)

if args.task[:4] == 'sscl':
    import torch
    args.device = torch.device(args.device)
    from .model_sscl import modelSSCL
    model = modelSSCL(args)
    if args.task == "sscl-train":
        model.train()
    if args.task == "sscl-validate":
        args.optimal_model_dir = '../nats_results/sscl_models'
        model.validate()
    if args.task == "sscl-test":
        args.file_output = 'test_sscl_output.json'
        args.optimal_model_dir = '../nats_results/sscl_models'
        model.test()
    if args.task == "sscl-teacher":
        args.file_test = args.file_train
        args.file_output = 'train_sscl_output.json'
        args.optimal_model_dir = '../nats_results/sscl_models'
        model.test()
    if args.task == 'sscl-evaluate':
        from .evaluation import evaluate_sscl_classification
        args.file_output = 'test_sscl_output.json'
        evaluate_sscl_classification(args)
    if args.task == 'sscl-clean':
        from glob import glob
        import os
        import shutil

        files_ = glob('../nats_results/*.model')
        out_dir = '../nats_results/sscl_train_models'
        if os.path.exists(out_dir) and len(files_) > 0:
            shutil.rmtree(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for fl_ in files_:
            shutil.move(fl_, fl_.replace('nats_results',
                                         'nats_results/sscl_train_models'))
        shutil.copy(
            '../nats_results/args.pickled',
            '../nats_results/sscl_train_models/args.pickled')
        shutil.copy(
            '../nats_results/sscl_aspect_keywords.txt',
            '../nats_results/sscl_train_models/sscl_aspect_keywords.txt')
        shutil.copy(
            '../nats_results/aspect_mapping.txt',
            '../nats_results/sscl_train_models/aspect_mapping.txt')
        files_ = glob('../nats_results/batch_*')
        for fl in files_:
            shutil.rmtree(fl)

if args.task[:7] == 'student':
    import torch
    args.device = torch.device(args.device)
    from .model_student import modelKD
    model = modelKD(args)
    if args.task == "student-train":
        args.data_dir = '../nats_results'
        args.file_train = 'train_sscl_output.json'
        model.train()
    if args.task == "student-validate":
        args.optimal_model_dir = '../nats_results/ts_models'
        model.validate()
    if args.task == "student-test":
        args.file_output = 'test_ts_output.json'
        args.optimal_model_dir = '../nats_results/ts_models'
        model.test()
    if args.task == "student-evaluate":
        args.file_output = 'test_ts_output.json'
        from .evaluation import evaluate_ts_classification
        evaluate_ts_classification(args)
    if args.task == 'student-clean':
        from glob import glob
        import os
        import shutil

        files_ = glob('../nats_results/*.model')
        out_dir = '../nats_results/ts_train_models'
        if os.path.exists(out_dir) and len(files_) > 0:
            shutil.rmtree(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for fl_ in files_:
            shutil.move(fl_, fl_.replace('nats_results',
                                         'nats_results/ts_train_models'))
        shutil.copy(
            '../nats_results/args.pickled',
            '../nats_results/ts_train_models/args.pickled')
        files_ = glob('../nats_results/batch_*')
        for fl in files_:
            shutil.rmtree(fl)
