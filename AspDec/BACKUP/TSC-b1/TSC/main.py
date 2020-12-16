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

parser.add_argument('--data_dir', default='../data/tv/',
                    help='directory that store the data.')
parser.add_argument('--file_train', default='train.txt', help='Training')
parser.add_argument('--file_dev', default='dev.txt', help='development data')
parser.add_argument('--file_test', default='test.txt', help='test data')
parser.add_argument('--file_abae_output', default='test_abae_output.json',
                    help='test output file')
parser.add_argument('--file_ts_output', default='test_ts_output.json',
                    help='test output file')
parser.add_argument('--base_model_dir', default='../nats_results/model',
                    help='directory for base models.')

parser.add_argument('--n_epoch', type=int, default=1,
                    help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size.')
parser.add_argument('--checkpoint', type=int, default=500,
                    help='How often you want to save model?')
parser.add_argument('--val_num_batch', type=int, default=400,
                    help='how many batches')
parser.add_argument('--nbestmodel', type=int, default=10,
                    help='How many models you want to keep?')
parser.add_argument('--validate_once', type=str2bool, default=True,
                    help='Only validate once.')

parser.add_argument('--continue_training', type=str2bool,
                    default=True, help='Do you want to continue?')
parser.add_argument('--continue_decoding', type=str2bool,
                    default=False, help='Do you want to continue?')
parser.add_argument('--train_base_model', type=str2bool, default=False,
                    help='True: Use Pretrained Param | False: Transfer Learning')
parser.add_argument('--use_move_avg', type=str2bool, default=False,
                    help='move average')
parser.add_argument('--use_optimal_model', type=str2bool, default=False,
                    help='Do you want to use the best model?')
parser.add_argument('--model_optimal_key', default='0,0', help='epoch,batch')
parser.add_argument('--is_lower', type=str2bool, default=True,
                    help='convert all tokens to lower case?')
# learning rate and scheduler
parser.add_argument('--lr_schedule', default='warm-up',
                    help='Schedule learning rate. | build-in | warm-up | None')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0,
                    help='clip the gradient norm.')
parser.add_argument('--step_size', type=int, default=2,
                    help='step size')
parser.add_argument('--step_decay', type=float, default=0.8,
                    help='learning rate decay')
parser.add_argument('--warmup_step', type=int, default=1000,
                    help='warmup step size')
parser.add_argument('--model_size', type=int, default=2000,
                    help='model size')
'''
User specified parameters.
'''
parser.add_argument('--device', default="cuda:0", help='device')
# abae
parser.add_argument('--distance', default='cosine',
                    help='cosine')
parser.add_argument('--emb_size', type=int, default=128,
                    help='embedding size')
parser.add_argument('--max_seq_len', type=int, default=30,
                    help='sentence length.')
parser.add_argument('--lambda_', type=float, default=0.5,
                    help='attention weights reg.')
# teacher student

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
# evaluate
parser.add_argument('--evaluate_coherence', type=str2bool, default=False,
                    help='Evaluate coherence.')

args = parser.parse_args()

if args.task == 'word2vec':
    from .word2vec import run_word2vec
    from .word2vec import convert_vectors
    run_word2vec(args)
    convert_vectors(args)

if args.task == 'doc_term_mat':
    from .kmeans import create_doc_term_matrix
    create_doc_term_matrix(args)

if args.task[:6] == 'kmeans':
    if args.task == 'kmeans':
        from .kmeans import run_kmeans
        from .kmeans import get_cluster_keywords
        run_kmeans(args)
        get_cluster_keywords(args)

    if args.task == 'kmeans_keywords':
        from .kmeans import get_cluster_keywords
        get_cluster_keywords(args)

    if args.task == 'kmeans_eval':
        from .evaluation import eval_aspect_coherence
        eval_aspect_coherence(
            args.cluster_dir, args.file_term_doc,
            args.file_vocab, args.n_keywords,
            file_aspect_weight='aspect_weight.txt')

if args.task == 'tfidf-seeds':
    from .tfidf_seeds import run_tfidf_seeds
    run_tfidf_seeds(args)

if args.task[:4] == 'abae':
    import torch
    args.device = torch.device(args.device)
    from .model_ABAE import modelABAE
    model = modelABAE(args)
    if args.task == "abae-train":
        model.train()
    if args.task == "abae-test":
        model.test()
    if args.task == 'abae-evaluate':
        from .evaluation import evaluate_classification
        evaluate_classification(args)

if args.task[:4] == 'mate':
    import torch
    args.device = torch.device(args.device)
    from .model_MATE import modelMATE
    model = modelMATE(args)
    if args.task == "mate-train":
        model.train()
    if args.task == "mate-test":
        model.test()

if args.task[:2] == 'ts':
    if args.task[-5:] == 'train' or args.task[-4:] == 'test':
        import torch
        args.device = torch.device(args.device)
        from .model_TS import modelTS
        model = modelTS(args)
        if args.task == "ts-train":
            model.train()
        if args.task == "ts-test":
            model.test()
    if args.task == "ts-evaluate":
        from .evaluation import evaluate_ts_classification
        evaluate_ts_classification(args)

if args.task == 'bow-test':
    from .model_bow import run_bow_classfication
    run_bow_classfication(args)



# if args.task[:8] == 'bert-w2v':
#     import torch
#     args.device = torch.device(args.device)
#     from .model_bert_w2v import modelBERTEmb
#     model = modelBERTEmb(args)
#     if args.task == 'bert-w2v':
#         model.train()
#     if args.task == 'bert-w2v-dump':
#         model.dump_embeddings()

# if args.task[:3] == 'e2e':
#     import torch
#     args.device = torch.device(args.device)
#     from .model_E2E import modelE2E
#     model = modelE2E(args)
#     if args.task == "e2e-train":
#         model.train()
#     if args.task == "e2e-test":
#         model.test()
