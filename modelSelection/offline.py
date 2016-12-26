'''
Created on Dec 25, 2016

@author: loapui
'''

import time
from os.path import dirname, join
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from featureEngineer import get_all_feature
from modelSelection.cascaded import cascadedClassifier

_OFFLINE_RESULT = join(dirname(dirname(dirname(__file__))), "offline_result")

def _compute_macro_F1(pred_studentID_label, truth_studentID_label):
    
    pred_studentID_label.columns = ['id', 'pred']
    truth_studentID_label.columns = ['id', 'truth']
    
    studentID_pred_truth = pd.merge(pred_studentID_label, truth_studentID_label, on='id')
    pred = studentID_pred_truth['pred'].values
    truth = truth_studentID_label['truth'].values
    hit = [_p for _p, _t in zip(pred, truth) if _p == _t]
    
    pred_0vsRest = [int(p > 0) for p in pred]
    truth_0vsRest = [int(t > 0) for t in truth]
    hit_0vsRest = [_p for _p, _t in zip(pred_0vsRest, truth_0vsRest) if _p == _t]
    
    f1_set = []
    for _label in range(2):
        nPred = Counter(pred_0vsRest)[_label]
        nTruth = Counter(truth_0vsRest)[_label]
        nHit = Counter(hit_0vsRest)[_label]
                
        precision = nHit / float(nPred)
        recall = nHit / float(nTruth)
        f1 = 2 * precision * recall / (precision + recall)
        
        f1_set.append(f1)
    
    sum_f1 = 0.0
    for _label in [1000, 1500, 2000]:
        nPred = Counter(pred)[_label]
        nTruth = Counter(truth)[_label]
        nHit = Counter(hit)[_label]
                
        precision = nHit / float(nPred)
        recall = nHit / float(nTruth)
        f1 = 2 * precision * recall / (precision + recall)
        
        f1_set.append(f1)
        sum_f1 += nTruth * f1
    
    macro_F1 = sum_f1 / len(truth)
    f1_set.append(macro_F1)
    
    return f1_set


if __name__ == "__main__":
    start = time.clock()
    
    train_feature = get_all_feature('train')
    # fill na
    train_feature = train_feature.fillna(0)
    
    y = train_feature['money']
    
    K = 3
    nFold = 2
    f1_list = []
    for ik in range(K):
        skf = StratifiedKFold(n_splits=nFold, shuffle=True, random_state=ik)
        iF = -1
        for train_index, val_index in skf.split(np.array(train_feature), np.array(y)):
            iF += 1
            trainX = train_feature.iloc[train_index, :]
            valX = train_feature.iloc[val_index, :]
            studentID_pred = cascadedClassifier(trainX, valX)
            studentID_truth = valX[['id', 'money']]
            f1 = _compute_macro_F1(studentID_pred, studentID_truth)
            with open(_OFFLINE_RESULT, 'a+') as f:
                print >> f, "fold %d: (label0, %f), (label123, %f), (label1, %f), (label2, %f), (label3, %f), (macroF1, %f)"\
                            %(ik * nFold + iF + 1, f1[0], f1[1], f1[2], f1[3], f1[4], f1[5])
            f1_list.append(f1)
            
    end = time.clock()
    f1_list = np.array(f1_list).mean(axis = 0).tolist()
    with open(_OFFLINE_RESULT, 'a+') as f:
        print >> f, "fold %s: (label0, %f), (label123, %f), (label1, %f), (label2, %f), (label3, %f), (macroF1, %f)"\
                    %("mean", f1_list[0], f1_list[1], f1_list[2], f1_list[3], f1_list[4], f1_list[5])
        print >> f, "run time: ", end - start
        print >> f, datetime.now()
        print >> f, "#########################################FINISHED####################################################"