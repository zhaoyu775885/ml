#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:24:24 2017

@author: zhaoyu
"""

import numpy as np
import pandas as pd
import copy
import ols

def permutation(idx, perm, label):
    """
    The BackTrack Algorithm Skeleton
    """
    if idx==label.size:
        print(perm)
    else:
        for i in range(label.size):
            if label[i] == 0:
                perm[idx] = i+1
                label[i] = 1
                permutation(idx+1, perm, label)
                label[i] = 0

def combination(idx, comb, label):
    """
    The BackTrack Algorithm Skeleton
    Needed for the subset selection
    """
    if idx == comb.size:
        print(comb)
    else:
        for i in range(0 if idx==0 else comb[idx-1], label.size):
            if label[i] == 0:
                comb[idx] = i+1
                label[i] = 1
                combination(idx+1, comb, label)
                label[i] = 0
                
def best_subset(idx, comb, label):
    global min_mpe, best_set
    global trainX, trainY
    global testX, testY
    global train_size, test_size
    global beta, test_error
    
    if idx == comb.size:
#        print(comb)
        sub_trainX = np.concatenate((np.mat(np.ones(train_size)).T, trainX[:, comb]), axis=1)
        beta_hat = ols.ols(sub_trainX, trainY)
        mpe = ols.eval_mpe(sub_trainX, trainY, beta_hat)

        if mpe < min_mpe:
            min_mpe = mpe
            best_set = copy.deepcopy(comb)
            beta = copy.deepcopy(beta_hat)
#            testY_hat = np.matmul(np.concatenate((np.mat(np.ones(test_size)).T, testX[:, comb]), axis=1), beta_hat)
#            error = testY - testY_hat
#            test_error = 1/test_size * np.dot(error.T, error)[0, 0] 
    else:
        for i in range(0 if idx==0 else comb[idx-1], label.size):
            if label[i] == 0:
                comb[idx] = i
                label[i] = 1
                best_subset(idx+1, comb, label)
                label[i] = 0

def forward_stepwise(X, Y, feat_sel_num):
    train_size, feat_size = X.shape
    tX = np.mat(np.ones(train_size)).T
    feat_set = []
    selected = 0
    
    for i in range(feat_sel_num):
        tmp_idx = -1
        min_mse = 1e10
        tX = np.concatenate((tX, np.mat(np.zeros(train_size)).T), axis=1)
        for idx in range(feat_size):
            if idx in feat_set:
                continue
            else:
                tX[:, -1] = X[:, idx]
                Y_hat = np.matmul(tX, ols.ols(tX, Y))
                error = Y-Y_hat
                mse = 1/test_size * np.dot(error.T, error)[0, 0]
                if mse<min_mse:
                    tmp_idx = idx
                    min_mse = mse
            
        tX[:, -1] = X[:, tmp_idx]
        feat_set.append(tmp_idx)
            
    return feat_set
    
def backward_stepwise(X, Y, feat_sel_num):
    sample_size, feat_size = X.shape
    idx_set = [i for i in range(feat_size)]
    
    while len(idx_set) > feat_sel_num:
        tX = np.concatenate((np.mat(np.ones(sample_size)).T, X[:, idx_set]), axis=1)
        beta_hat = ols.ols(tX, Y)
        beta_std_dev = ols.std_dev(tX, 1)
        Z_score = np.true_divide(np.array(beta_hat), np.array(beta_std_dev)).tolist()
        print(Z_score)
        print()
        idx = Z_score.index(min(Z_score)) - 1
        del idx_set[idx]

    return idx_set
        
        
#def forward_stage():
    
    

if __name__ == '__main__':
 
    upper = 5
    label = np.zeros(upper, dtype=int)
    perm = np.zeros(upper, dtype=int)
    permutation(0, perm, label)
    K = 3
    comb = np.zeros(K, dtype=int)
    combination(0, comb, label)
    
    """
    ps = pd.read_table('./prostate.txt')
    
    full_data = np.mat(ps.iloc[:, 1:-2])
    full_data = ols.scale(full_data)
    full_size, feat_size = full_data.shape
    
    trainX = full_data[ps.loc[ps.train=='T'].iloc[:, 0]-1, :]
    trainY = np.mat(ps.loc[ps.train=='T'].iloc[:, -2]).T
    train_size, feat_size = trainX.shape

    testX = full_data[ps.loc[ps.train!='T'].iloc[:, 0]-1, :]
    testY = np.mat(ps.loc[ps.train!='T'].iloc[:, -2]).T
    test_size, feat_size = testX.shape
    
 
    num_sub_feat = 2


#    Best-Subset Selection
    label = np.zeros(feat_size, dtype=int)
    comb = np.zeros(num_sub_feat, dtype=int)
    
    min_mpe = 100
    best_set = []
    beta = np.mat(np.zeros(3)).I
    test_error = 0
    
    best_subset(0, comb, label)
    
    kf = KFold(n_splits=10)
    k_mpe = []
    
    for t_idx, v_idx in kf.split(trainX):
#        print(t_idx, v_idx)
        t_X = trainX[t_idx, :]
        t_Y = trainY[t_idx, :]
        v_X = trainX[v_idx, :]
        v_Y = trainY[v_idx, :]
        beta_hat = ols.ols(t_X, t_Y)
        best_subset()
        mpe = ols.eval_mpe(v_X, v_Y, beta_hat)
        k_mpe.append(mpe)

    print(np.var(np.array(k_mpe)))    
    
    print(min_mpe)
    print(ols.eval_mpe(np.concatenate((np.mat(np.ones(test_size)).T, testX[:, best_set]), axis=1), testY, beta))
    print(best_set)
    print(beta)


    '''
#    Forward-Stepwise Selection
    forward_set = forward_stepwise(trainX, trainY, num_sub_feat)
    forward_trainX = np.concatenate((np.mat(np.ones(train_size)).T, trainX[:, forward_set]), axis=1)
    forward_beta_hat = ols.ols(forward_trainX, trainY)
    forward_testX = np.concatenate((np.mat(np.ones(test_size)).T, testX[:, forward_set]), axis=1)
    forward_mpe = ols.eval_mpe(forward_testX, testY, forward_beta_hat)
    print(forward_mpe)
    print(forward_set)
    print(forward_beta_hat)
    '''
    
    '''
#    Backward-Stepwise Selection
    backward_set = backward_stepwise(trainX, trainY, num_sub_feat)
    backward_trainX = np.concatenate((np.mat(np.ones(train_size)).T, trainX[:, backward_set]), axis=1)
    backward_beta_hat = ols.ols(backward_trainX, trainY)
    backward_testX = np.concatenate((np.mat(np.ones(test_size)).T, testX[:, backward_set]), axis=1)
    backward_mpe = ols.eval_mpe(backward_testX, testY, backward_beta_hat)
    print(backward_mpe)
    print(backward_set)
    print(backward_beta_hat)    
    '''
       """