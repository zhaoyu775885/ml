#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:08:57 2018

@author: zhaoyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def svm(x, y, alpha):
    '''
    Sequential Minimal Optimization (SMO) solution for SVM
    '''
    samp_sz = len(x)
    x_mat = np.asmatrix(x)
    KMat =  x_mat * x_mat.T
    
    idx1 = 0
    idx2 = 2
    
    C = alpha[idx1]*y[idx1] + alpha[idx2]*y[idx2]
    
    eta = KMat[idx1, idx1] - 2*KMat[idx1, idx2] + KMat[idx2, idx2]
    g1 = -y[idx1]
    g2 = -y[idx2]
    
    for it in range(samp_sz):
        g1 += alpha[it]*y[it]*KMat[idx1, it]
        g2 += alpha[it]*y[it]*KMat[idx2, it]
    
    print(g1, g2)

    alpha[idx1] -= y[idx1] * (g1-g2) / eta
    alpha[idx2] = (C-alpha[idx1]*y[idx1])*y[idx2]

    print(alpha)

def plot_scatter(data):
    '''
    The parameter data is of pandas data frame type
    '''
    data_pos = data[data.iloc[:, -1]==1]
    data_neg = data[data.iloc[:, -1]==-1]
    plt.scatter(data_pos.iloc[:, 0], data_pos.iloc[:, 1], marker='o', color='r')
    plt.scatter(data_neg.iloc[:, 0], data_neg.iloc[:, 1], marker='d', color='b')


if __name__ == '__main__':
    ps = pd.read_table('testSet.txt', header=None)
#    plot_scatter(ps)
    
    samp_sz, feat_sz = ps.iloc[:, :-1].shape
    
    data = np.mat(ps)
    x = np.array(ps.iloc[:, :-1])
    y = np.array(ps.iloc[:, -1])

    alpha = np.zeros(samp_sz)
    
    svm(x, y, alpha)
   
    