#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def scale(X):
    samp_size, feat_size = X.shape
    for jt in range(feat_size):
        mean = np.mean(X[:, jt])
        std_dev = np.sqrt(np.var(X[:, jt], ddof=1))
        X[:, jt] = (X[:, jt]-mean) / std_dev
    return X

def ols(X, Y):
    gram = X.T*X
    beta = gram.I * X.T * Y
    return beta

def std_dev(X, var):
    samp_size, feat_size = X.shape
    beta_var_mat = var*(X.T*X).I
    return np.mat(np.sqrt(np.diag(beta_var_mat))).T
    
    
def eval_mpe(X, Y, beta_hat):
    samp_size, feat_size = X.shape
    Y_hat = X * beta_hat
    error = Y_hat - Y
    error_2 = np.array(error)**2
#    print('tst: ', np.mean(error_2), np.std(error_2, ddof=1))    
    
#    pf = pd.DataFrame({'1':np.array(Y_hat)[:, 0].tolist(),
#                       '2':np.array(Y)[:, 0].tolist(),
#                       '3':np.array(error)[:, 0].tolist()})
#    print(pf)
#    print("here is the mean(err)", np.sqrt(1/(samp_size-1)*(tmp-np.mean(tmp)).T*(tmp-np.mean(tmp))))
    return 1/samp_size * (error.T*error)[0, 0]
    
    
if __name__ == '__main__':
    ps = pd.read_table('./prostate.txt', float_precision='high')
    
    full_data = np.mat(ps.iloc[:, 1:-1])
    full_prdt = full_data[:, :-1]
    full_prdt = scale(full_prdt)
    full_resp = full_data[:, -1]

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

    print('Training samples: {0}\nTest samples: {1}'.format(train_size, test_size))
    
    cor_mat = np.corrcoef(trainX.T)
    print(trainX.shape)
    print(cor_mat)
    
    trainX = np.concatenate((np.mat(np.ones([train_size, 1])), trainX), axis=1)
#    Add the intercept to the training data
    
    beta_hat = ols(trainX, trainY)
#    print(beta_hat)

    trainY_hat = trainX*beta_hat
    error = trainY - trainY_hat
    est_var = 1/(train_size-feat_size-1) * (error.T*error)[0, 0]
#    print(np.sqrt(est_var))

    beta_std_dev = std_dev(trainX, est_var)
#    print(beta_dev)

    if beta_hat.shape == beta_std_dev.shape:
        Z_score = np.divide(beta_hat, beta_std_dev)
#        print(Z_score)
    else:
        raise AssertionError
        
    labels = list(ps)[1:-2]
    labels.insert(0, 'intercept')
    index = ['Term', 'Coefficient', 'Std. Error', 'Z Score']
    df = pd.DataFrame({index[0]:labels,
                       index[1]:beta_hat.reshape(1,feat_size+1).tolist()[0],
                       index[2]:beta_std_dev.reshape(1, feat_size+1).tolist()[0],
                       index[3]:Z_score.reshape(1, feat_size+1).tolist()[0],})
    df = df[index].round({index[1]:2, index[2]:2, index[3]:2})
    print(df)
    
    test_mpe = eval_mpe(trainX, trainY, beta_hat)
#    test_mpe = eval_mpe(np.concatenate((np.mat(np.ones([test_size, 1])), testX), axis=1), testY, beta_hat)
    print('Test Error : {0:0.3f}'.format(test_mpe))
    
    print('Still have not got the meaning of "Std Error" on ESL Page 63.')
    
    
'''
    cavol= trainX[:, 1]
    Vcavol = 1/(train_size-1) * np.dot(cavol.T-np.mean(cavol), cavol-np.mean(cavol))
    Vage = 1/(train_size-1) * np.dot((age.T-np.mean(age)), (age-np.mean(age)))
    Cov_w_c = 1/(train_size-1)*np.dot(age.T-np.mean(age), cavol-np.mean(cavol))
    cor = Cov_w_c/np.sqrt(Vage*Vcavol)
'''
    