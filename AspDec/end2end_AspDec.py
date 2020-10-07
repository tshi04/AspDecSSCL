'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import glob
import json
import os
import pickle
import re
import shutil
import time
from pprint import pprint

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.data.utils import create_batch_file
from LeafNATS.engines.end2end_large import End2EndBase
from LeafNATS.utils.utils import show_progress


class End2EndAspDecBase(End2EndBase):
    '''
    End2End Aspect Detection
    '''

    def __init__(self, args=None):
        '''
        Initialize
        '''
        super().__init__(args=args)

        self.test_data = []

    def evaluate_worker(self, input_):
        '''
        Used for evaluation
        '''
        raise NotImplementedError

    def validate(self):
        '''
        Validation here.
        Don't overwrite.
        '''
        self.build_vocabulary()
        self.build_models()
        pprint(self.base_models)
        pprint(self.train_models)
        self.init_base_model_params()

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()
        with torch.no_grad():
            model_para_files = []
            model_para_files = glob.glob(os.path.join(
                '../nats_results', sorted(list(self.train_models))[0]+'*.model'))
            for j in range(len(model_para_files)):
                arr = re.split(r'\_|\.', model_para_files[j])
                arr = [int(arr[-3]), int(arr[-2]), model_para_files[j]]
                model_para_files[j] = arr
            model_para_files = sorted(model_para_files)

            if not os.path.exists(self.args.optimal_model_dir):
                os.mkdir(self.args.optimal_model_dir)
            best_f1 = 0
            for fl_ in model_para_files:
                print('Validate *_{}_{}.model'.format(fl_[0], fl_[1]))
                try:
                    for model_name in self.train_models:
                        fl_tmp = os.path.join(
                            '../nats_results',
                            model_name+'_'+str(fl_[0])+'_'+str(fl_[1])+'.model')
                        self.train_models[model_name].load_state_dict(
                            torch.load(fl_tmp, map_location=lambda storage, loc: storage))
                except:
                    print('Models cannot be load!!!')
                    continue
                val_batch = create_batch_file(
                    path_data=self.args.data_dir,
                    path_work='../nats_results',
                    is_shuffle=False,
                    fkey_=self.args.task,
                    file_=self.args.file_dev,
                    batch_size=self.args.batch_size
                )
                print('The number of batches (Dev): {}'.format(val_batch))

                val_results = []
                for batch_id in range(val_batch):
                    start_time = time.time()
                    self.build_batch(batch_id)
                    self.test_worker()
                    val_results += self.test_data
                    self.test_data = []
                    end_time = time.time()
                    show_progress(batch_id+1, val_batch, str(
                        (end_time-start_time))[:8]+"s")
                print()

                f1 = self.evaluate_worker(val_results)
                print('Best f1: {}; Current f1: {}.'.format(best_f1, f1))
                
                if f1 > best_f1:
                    for model_name in self.train_models:
                        fmodel = open(os.path.join(
                            self.args.optimal_model_dir, 
                            '{}.model'.format(model_name)), 'wb')
                        torch.save(
                            self.train_models[model_name].state_dict(), fmodel)
                        fmodel.close()
                    best_f1 = f1

    def aspect_worker(self):
        '''
        Used to extract aspect keywords.
        '''
        raise NotImplementedError

    def test(self):
        '''
        testing
        Don't overwrite.
        '''
        self.build_vocabulary()
        self.build_models()
        pprint(self.base_models)
        pprint(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()

        _nbatch = create_batch_file(
            path_data=self.args.data_dir,
            path_work=os.path.join('..', 'nats_results'),
            is_shuffle=False,
            fkey_=self.args.task,
            file_=self.args.file_test,
            batch_size=self.args.batch_size
        )
        print('The number of samples (test): {}'.format(_nbatch))

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()
        with torch.no_grad():
            if self.args.use_optimal_model:
                for model_name in self.train_models:
                    fl_ = os.path.join(
                        self.args.optimal_model_dir, '{}.model'.format(model_name))
                    self.train_models[model_name].load_state_dict(
                        torch.load(fl_, map_location=lambda storage, loc: storage))
            else:
                arr = re.split(r'\D', self.args.model_optimal_key)
                model_optimal_key = ''.join(
                    ['_', arr[0], '_', arr[1], '.model'])
                print("You choose to use *{} for decoding.".format(model_optimal_key))

                for model_name in self.train_models:
                    model_optimal_file = os.path.join(
                        '../nats_results', model_name+model_optimal_key)
                    self.train_models[model_name].load_state_dict(torch.load(
                        model_optimal_file, map_location=lambda storage, loc: storage))

            start_time = time.time()
            output_file = os.path.join(
                '../nats_results', self.args.file_output)
            
            fout = open(output_file, 'w')
            self.aspect_worker()
            for batch_id in range(_nbatch):
                self.build_batch(batch_id)
                self.test_worker()
                for itm in self.test_data:
                    json.dump(itm, fout)
                    fout.write('\n')
                self.test_data = []
                end_time = time.time()
                show_progress(batch_id+1, _nbatch, str(
                    (end_time-start_time)/3600)[:8]+"h")
            fout.close()
            print()
