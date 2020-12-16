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
from LeafNATS.utils.utils import show_progress

from .end2end_AspExt import End2EndAspExtBase


class End2EndTSBase(End2EndAspExtBase):
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

        best_arr = []
        val_file = os.path.join('..', 'nats_results', 'model_validate.txt')
        if os.path.exists(val_file):
            fp = open(val_file, 'r')
            for line in fp:
                arr = re.split(r'\s', line[:-1])
                best_arr.append(
                    [arr[0], arr[1], arr[2], float(arr[3]), float(arr[4])])
            fp.close()

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()
        with torch.no_grad():
            while 1:
                model_para_files = []
                model_para_files = glob.glob(os.path.join(
                    '..', 'nats_results', sorted(list(self.train_models))[0]+'*.model'))
                for j in range(len(model_para_files)):
                    arr = re.split(r'\_|\.', model_para_files[j])
                    arr = [int(arr[-3]), int(arr[-2]), model_para_files[j]]
                    model_para_files[j] = arr
                model_para_files = sorted(model_para_files)

                for fl_ in model_para_files:
                    best_model = {itm[0]: itm[3] for itm in best_arr}
                    if fl_[-1] in best_model:
                        continue
                    print('Validate *_{}_{}.model'.format(fl_[0], fl_[1]))

                    losses = []
                    start_time = time.time()
                    if os.path.exists(fl_[-1]):
                        time.sleep(3)
                        try:
                            for model_name in self.train_models:
                                fl_tmp = os.path.join(
                                    '..', 'nats_results',
                                    model_name+'_'+str(fl_[0])+'_'+str(fl_[1])+'.model')
                                self.train_models[model_name].load_state_dict(
                                    torch.load(fl_tmp, map_location=lambda storage, loc: storage))
                        except:
                            print('Models cannot be load!!!')
                            continue
                    else:
                        continue
                    val_batch = create_batch_file(
                        path_data=self.args.data_dir,
                        path_work=os.path.join('..', 'nats_results'),
                        is_shuffle=True,
                        fkey_=self.args.task,
                        file_=self.args.file_val,
                        batch_size=self.args.batch_size
                    )
                    print('The number of batches (Dev): {}'.format(val_batch))
                    if self.args.val_num_batch > val_batch:
                        self.args.val_num_batch = val_batch
                    for batch_id in range(self.args.val_num_batch):

                        self.build_batch(batch_id)
                        loss = self.build_pipelines()

                        losses.append(loss.data.cpu().numpy())
                        show_progress(batch_id+1, self.args.val_num_batch)
                    print()
                    losses = np.array(losses)
                    end_time = time.time()
                    if self.args.use_move_avg:
                        try:
                            losses_out = 0.9*losses_out + \
                                0.1*np.average(losses)
                        except:
                            losses_out = np.average(losses)
                    else:
                        losses_out = np.average(losses)
                    best_arr.append(
                        [fl_[2], fl_[0], fl_[1], losses_out, end_time-start_time])
                    best_arr = sorted(best_arr, key=lambda bb: bb[3])
                    if best_arr[0][0] == fl_[2]:
                        out_dir = os.path.join('..', 'nats_results', 'model')
                        try:
                            shutil.rmtree(out_dir)
                        except:
                            pass
                        os.mkdir(out_dir)
                        for model_name in self.base_models:
                            fmodel = open(os.path.join(
                                out_dir, model_name+'.model'), 'wb')
                            torch.save(
                                self.base_models[model_name].state_dict(), fmodel)
                            fmodel.close()
                        for model_name in self.train_models:
                            fmodel = open(os.path.join(
                                out_dir, model_name+'.model'), 'wb')
                            torch.save(
                                self.train_models[model_name].state_dict(), fmodel)
                            fmodel.close()
                        try:
                            shutil.copy2(os.path.join(
                                self.args.data_dir, self.args.file_vocab), out_dir)
                        except:
                            pass
                    for itm in best_arr[:self.args.nbestmodel]:
                        print('model={}_{}, loss={}, time={}'.format(
                            itm[1], itm[2], np.round(float(itm[3]), 4),
                            np.round(float(itm[4]), 4)))

                    for itm in best_arr[self.args.nbestmodel:]:
                        tarr = re.split(r'_|\.', itm[0])
                        if tarr[-2] == '0':
                            continue
                        if os.path.exists(itm[0]):
                            for model_name in self.train_models:
                                fl_tmp = os.path.join(
                                    '..', 'nats_results',
                                    model_name+'_'+str(itm[1])+'_'+str(itm[2])+'.model')
                                os.unlink(fl_tmp)
                    fout = open(val_file, 'w')
                    for itm in best_arr:
                        if len(itm) == 0:
                            continue
                        fout.write(' '.join([itm[0], str(itm[1]), str(
                            itm[2]), str(itm[3]), str(itm[4])])+'\n')
                    fout.close()
                if self.args.validate_once:
                    break

    def aspect_worker(self):
        '''
        Aspect keywords
        '''
        return
