'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import numpy as np
import json
from glob import glob


def evaluation(input_dir):
    '''
    Evaluation Metrics
    '''
    files = glob('{}/*'.format(input_dir))
    for file_ in files:
        print('Validate file: {}'.format(file_))
        fp = open(file_, 'r')
        data = []
        for line in fp:
            try:
                itm = json.loads(line)
            except:
                continue
            data.append(itm)

        aa = aat = bb = bbt = 0
        for itm in data:
            for wd in itm['pred']:
                if wd in itm['gold']:
                    aa += 1
                aat += 1
            for wd in itm['gold']:
                if wd in itm['pred']:
                    bb += 1
                bbt += 1
        precision = aa/aat
        recall = bb/bbt
        fscore = 2.0*precision*recall/(precision+recall)
        print('micro precision={}, recall={}, f-score={}'.format(
            precision, recall, fscore))

        aa = aat = 0
        for itm in data:
            if itm['pred'] == itm['gold']:
                aa += 1
            aat += 1
        accuracy = aa/aat
        print('Accuracy={}'.format(accuracy))

        precision = []
        recall = []
        fscore = []
        for itm in data:
            aa = aat = bb = bbt = 0
            for wd in itm['pred']:
                if wd in itm['gold']:
                    aa += 1
                aat += 1
            for wd in itm['gold']:
                if wd in itm['pred']:
                    bb += 1
                bbt += 1
            pp = aa/aat + 1e-10
            rr = bb/bbt + 1e-10
            precision.append(pp)
            recall.append(rr)
            fscore.append(2.0*pp*rr/(pp+rr))
        accuracy = aa/aat
        print('macro precision={}, recall={}, f-score={}'.format(
            np.mean(precision), np.mean(recall), np.mean(fscore)))

        fp.close()
