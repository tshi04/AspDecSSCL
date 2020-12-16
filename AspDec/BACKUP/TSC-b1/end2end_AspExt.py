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


class End2EndAspExtBase(End2EndBase):
    '''
    End2End training classification.
    Not suitable for language generation task.
    Light weight. Data should be relevatively small.
    '''

    def __init__(self, args=None):
        '''
        Initialize
        '''
        super().__init__(args=args)

        self.test_data = []

    def test_worker(self):
        '''
        Used in decoding.
        Users can define their own decoding process.
        You do not have to worry about path and prepare input.
        '''
        raise NotImplementedError

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
            arr = re.split(r'\D', self.args.model_optimal_key)
            model_optimal_key = ''.join(
                ['_', arr[0], '_', arr[1], '.model'])
            print("You choose to use *{} for decoding.".format(model_optimal_key))

            for model_name in self.train_models:
                model_optimal_file = os.path.join(
                    '..', 'nats_results', model_name+model_optimal_key)
                self.train_models[model_name].load_state_dict(torch.load(
                    model_optimal_file, map_location=lambda storage, loc: storage))

            start_time = time.time()
            output_file = os.path.join('..', 'nats_results',
                                       self.args.file_output)
            data_check = []
            try:
                self.args.continue_decoding
            except:
                self.args.continue_decoding = True
            if os.path.exists(output_file) and self.args.continue_decoding:
                fchk = open(output_file, 'r')
                for line in fchk:
                    data_check.append(line)
                fchk.close()
                cchk = int(len(data_check)/self.args.batch_size)
                cchk = (cchk-1) * self.args.batch_size
                if cchk < 0:
                    cchk == 0
                data_check = data_check[:cchk]
                fchk = open(output_file, 'w')
                for line in data_check:
                    fchk.write(line)
                fchk.close()
            else:
                fout = open(output_file, 'w')
                fout.close()
            try:
                fout = open(output_file, 'a')
            except:
                fout = open(output_file, 'w')
            self.aspect_worker()
            for batch_id in range(_nbatch):
                if batch_id*self.args.batch_size < len(data_check):
                    continue
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
