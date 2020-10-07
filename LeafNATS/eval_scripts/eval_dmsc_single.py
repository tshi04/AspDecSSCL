'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def evaluation(args):
    '''
    Evaluation Metrics
    -- Accuracy, MSE
    Best model is selected based on accuracy.
    '''
    file_ = os.path.join(
        '../nats_results', args.test_output_dir,
        'test_pred_{}.txt'.format(args.best_epoch))
    pred_data = np.loadtxt(file_)
    file_ = os.path.join(
        '../nats_results', args.test_output_dir,
        'test_true_{}.txt'.format(args.best_epoch))
    true_data = np.loadtxt(file_)

    label_pred = []
    label_true = []
    for k in range(args.n_tasks):
        predlb = [rt for idx, rt in enumerate(
            pred_data[:, k].tolist()) if true_data[idx, k] != 0 and pred_data[idx, k] != 0]
        truelb = [rt for idx, rt in enumerate(
            true_data[:, k].tolist()) if true_data[idx, k] != 0 and pred_data[idx, k] != 0]
        label_pred += predlb
        label_true += truelb

    accu = accuracy_score(label_true, label_pred)
    mse = mean_squared_error(label_true, label_pred)

    accu = round(accu.tolist(), 4)
    mse = round(mse.tolist(), 4)

    print('Accuracy={}, MSE={}'.format(accu, mse))
