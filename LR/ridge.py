#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:04:31 2017

@author: zhaoyu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import ols

def ridge(X, Y, alpha):
    samp_size, feat_size = X.shape
    gram = np.matmul(X.T, X) + np.mat(np.eye(feat_size)) * alpha
    beta = np.matmul(gram.I, np.matmul(X.T, Y))
    return beta

if __name__ == '__main__':
    ps = pd.read_table('./prostate.txt', float_precision='high')
    
    full_data = np.mat(ps.iloc[:, 1:-1])
    full_prdt = full_data[:, :-1]
    full_resp = full_data[:, -1]
    full_prdt = ols.scale(full_prdt)
    
    samp_size, feat_size = full_prdt.shape
    print('{0} features, {1} samples'.format(feat_size, samp_size))    
    
    train_label = ps.loc[ps.train=='T'].iloc[:, 0]-1
    test_label = ps.loc[ps.train=='F'].iloc[:, 0]-1

    train_size = len(train_label)
    test_size = len(test_label)

    if test_size+train_size != samp_size:
        raise AssertionError
    
    trainX = full_prdt[train_label, :]
    trainY = full_resp[train_label, 0]

    testX = full_prdt[test_label, :]
    testY = full_resp[test_label, 0]

    rows_random = np.arange(train_size)
    np.random.shuffle(rows_random)
    trainX = trainX[rows_random.tolist(), :]
    trainY = trainY[rows_random.tolist(), :]

    print('Training samples: {0}\nTest samples: {1}'.format(train_size, test_size))
    
    cor_mat = np.corrcoef(trainX.T)
#    print(cor_mat)
    
    trainX = np.concatenate((np.mat(np.ones([train_size, 1])), trainX), axis=1)
#    Add the intercept to the training data
    
#    beta_hat = ols.ols(trainX, trainY)
#    print(beta_hat)

    kf = KFold(n_splits=10)
#    k_mpe = []
    trainY_hat = np.mat(np.zeros(train_size)).reshape(train_size, 1)
    
    for t_idx, v_idx in kf.split(trainX):
#        print(t_idx, v_idx)
        t_X = trainX[t_idx, :]
        t_Y = trainY[t_idx, :]
        v_X = trainX[v_idx, :]
        v_Y = trainY[v_idx, :]
        beta_hat = ridge(t_X, t_Y, 0)
        v_Y_hat= np.matmul(v_X, beta_hat)
        for idx, val, r_val in zip(v_idx, v_Y_hat, v_Y):
            print(idx, val, r_val)
            trainY_hat[idx] = val
    
    error = trainY - trainY_hat
    mpe = 1/train_size * np.matmul(error.T, error)
    std_err = np.sqrt(np.var(error))
    print(mpe, std_err)
    
#        mpe = ols.eval_mpe(v_X, v_Y, beta_hat)
#        k_mpe.append(mpe)

#    print(k_mpe)
#    print(np.var(np.array(k_mpe)))