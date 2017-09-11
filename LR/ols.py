#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def ols(X, Y):
    gram = np.matmul(X.T, X)
    beta = np.matmul(gram.I, np.matmul(X.T, Y))
    return beta

def std_pdt(train):
    num_sample, feat_size = train.shape
    print(train.shape)    
    for it in range(1, feat_size):
        mean = np.mean(train[:, it])
        std = np.sqrt(np.var(train[:, it], ddof=1))
        train[:, it] = (train[:, it]-mean)/std
    return train

if __name__ == '__main__':
    ps = pd.read_table('./prostate.txt', float_precision='high')

    train_size, feat_size = ps.loc[ps.train=='T'].iloc[:, 1:-2].shape
    print(train_size, feat_size)
    
    trainX = np.mat(ps.loc[ps.train=='T'].iloc[:, 1:-2])
#    trainX = np.mat(ps.iloc[:67, 1:-2])
    trainX = np.concatenate((np.mat(np.ones(train_size)).T, trainX), axis=1)
    trainY = np.mat(ps.loc[ps.train=='T'].iloc[:, -2]).T
    trainX = std_pdt(trainX)
    
    beta_hat = ols(trainX, trainY)
    print(beta_hat)
#    
#    trainY_hat = np.matmul(trainX, beta_hat)
#    error = trainY - trainY_hat
#    print(error)
    

#cavol= trainX[:, 1]
#Vcavol = 1/(train_size-1) * np.dot(cavol.T-np.mean(cavol), cavol-np.mean(cavol))
#Vage = 1/(train_size-1) * np.dot((age.T-np.mean(age)), (age-np.mean(age)))
#Cov_w_c = 1/(train_size-1)*np.dot(age.T-np.mean(age), cavol-np.mean(cavol))
#cor = Cov_w_c/np.sqrt(Vage*Vcavol)