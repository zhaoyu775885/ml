#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def scale(X):
    samp_size, feat_size = X.shape
    for jt in range(feat_size):
        mean = np.mean(X[:, jt])
        std_dev = np.sqrt(np.var(X[:, jt], ddof=1))
        X[:, jt] = (X[:, jt]-mean) / std_dev
    return X

def ols(X, Y):
    gram = np.matmul(X.T, X)
    beta = np.matmul(gram.I, np.matmul(X.T, Y))
    return beta

def std_dev(X, var):
    samp_size, feat_size = X.shape
    beta_var_mat = var*np.matmul(X.T, X).I
    return np.mat(np.sqrt(np.diag(beta_var_mat))).T
    
    
def eval_mpe(X, Y, beta_hat):
    samp_size, feat_size = X.shape
    Y_hat = np.matmul(X, beta_hat)
    error = Y_hat - Y
    return 1/samp_size * np.matmul(error.T, error)[0, 0]

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
    full_prdt = scale(full_prdt)
    
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

#    randomly rearange the rows
    rows_random = np.arange(train_size)
    np.random.shuffle(rows_random)
    trainX = trainX[rows_random.tolist(), :]
    trainY = trainY[rows_random.tolist(), :]
                    

    print('Training samples: {0}\nTest samples: {1}'.format(train_size, test_size))
    
    cor_mat = np.corrcoef(trainX.T)
#    print(cor_mat)
    
    trainX = np.concatenate((np.mat(np.ones([train_size, 1])), trainX), axis=1)
#    Add the intercept to the training data
    
#    beta_hat = ols(trainX, trainY)
#    print(beta_hat)

    kf = KFold(n_splits=10)
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
    
#        mpe = eval_mpe(v_X, v_Y, beta_hat)
#        k_mpe.append(mpe)

#    print(k_mpe)
#    print(np.var(np.array(k_mpe)))