#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:53:31 2018

@author: zhaoyu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def logit(X, beta):
    return 1.0/(1.0+np.exp(-1.0*X*beta))

def logistic_regression(df):
    samp_sz, feat_sz = df.shape
    
    X = np.mat(df.iloc[:, :-1])
    X = np.concatenate((np.mat(np.ones([samp_sz, 1])), X), axis=1)
    y = np.mat(df.iloc[:, -1]).T
               
    beta_old = np.mat(np.zeros([X.shape[1], 1]))
    p = logit(X, beta_old)
    W = np.mat(np.diag(np.ravel(p)))
    while True:
        beta_new = beta_old + (X.T*W*X).I*X.T*(y-p)
        if np.linalg.norm(beta_new-beta_old)<1e-6*np.linalg.norm(beta_old):
            break
        beta_old = beta_new
        p = logit(X, beta_old)
        W = np.mat(np.diag(np.ravel(p)))
        print(beta_new.T)
    

def plot_scatter(data):
    pos_data = data.loc[data['chd']==1]
    neg_data = data.loc[data['chd']==0]
    plt.scatter(pos_data.iloc[:, 1], pos_data.iloc[:, -1], marker='o', color='r')
    plt.scatter(neg_data.iloc[:, 1], neg_data.iloc[:, -1], marker='x', color='g')

if __name__ == '__main__':
    df = pd.read_csv('./chd.txt')
    del df['row.names'], df['typea'], df['adiposity']
    
    df['famhist'] = [1 if df.loc[i, 'famhist']=='Present' 
       else 0 for i in range(len(df))]
    
    cor_mat = np.corrcoef(np.mat(df).T)
    
    logistic_regression(df)