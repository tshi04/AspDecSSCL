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
                    help='train | validate | test | evaluate')

parser.add_argument('--data_dir', default='../data/restaurant/',
                    help='directory that store the data.')
parser.add_argument('--file_train', default='train.txt', help='Training')
parser.add_argument('--file_val', default='test.txt', help='validation')
parser.add_argument('--file_test', default='test.txt', help='test data')
parser.add_argument('--file_output', default='test_output.json',
                    help='test output file')

parser.add_argument('--n_epoch', type=int, default=10,
                    help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=64, 
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
parser.add_argument('--train_base_model', type=str2bool, default=False,
                    help='True: Use Pretrained Param | False: Transfer Learning')
parser.add_argument('--use_move_avg', type=str2bool, default=False, 
                    help='move average')
parser.add_argument('--use_optimal_model', type=str2bool, default=False, 
                    help='Do you want to use the best model?')
parser.add_argument('--model_optimal_key', default='9,0', help='epoch,batch')
parser.add_argument('--is_lower', type=str2bool, default=True,
                    help='convert all tokens to lower case?')
# scheduler
parser.add_argument('--learning_rate', type=float, default=0.0005, 
                    help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0,
                    help='clip the gradient norm.')

parser.add_argument('--lr_schedule', default='None',
                    help='Schedule learning rate. | build-in | warm-up | None')
parser.add_argument('--step_size', type=int, default=2,
                    help='step size')
parser.add_argument('--step_decay', type=float, default=0.8,
                    help='learning rate decay')
parser.add_argument('--warmup_step', type=int, default=2000,
                    help='warmup step size')
parser.add_argument('--model_size', type=int, default=3000,
                    help='model size')
'''
User specified parameters.
'''
parser.add_argument('--device', default="cuda:0", help='device')
# vocabulary
parser.add_argument('--file_pretrain_vocab', default='vocab_glove_42B_300d',
                    help='file store pretrain vocabulary.')
parser.add_argument('--file_pretrain_vec', default='glove_42B_300d.npy',
                    help='file store pretrain vec.')
parser.add_argument('--file_aspect', default='aspect.txt', 
                    help='aspect labels')

parser.add_argument('--emb_size', type=int, default=300,
                    help='encoder hidden size')
parser.add_argument('--hidden_size', type=int, default=300,
                    help='encoder hidden size')

parser.add_argument('--max_seq_len', type=int, default=50,
                    help='sentence length.')
parser.add_argument('--n_aspects', type=int, default=20,
                    help='number of aspects')
                    
args = parser.parse_args()

if args.task == 'train' or args.task == 'validate' or args.task == 'test':
    import torch
    args.device = torch.device(args.device)
    from .model import modelUAE
    model = modelUAE(args)
if args.task == "train":
    model.train()
if args.task == "validate":
    model.validate()
if args.task == "test":
    model.test()
