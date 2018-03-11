#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:08:57 2018

@author: zhaoyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def get_beta(alpha, y, x):
    '''
    beta = \sum_{i}^{N}alpha[i]*y[i]*x[:, i]
    '''
    samp_sz, feat_sz = x.shape
    
    beta = np.zeros([feat_sz], dtype='float64')
    for idx in range(samp_sz):
        beta += alpha[idx]*y[idx]*x[idx, :]

    return beta
    
    
def selectJrand(i, m):
    j = i
    while j==i:
        j = int(np.random.uniform(0, m))
    return j
    

def svm(x, y, alpha):
    '''
    Sequential Minimal Optimization (SMO) solution for SVM
    '''
    
    # Define basic varialbes
    samp_sz = len(x)
    x_mat = np.asmatrix(x)
    KMat =  x_mat * x_mat.T
    
    k = 0
    
    beta = get_beta(alpha, y, x)
    beta0 = 0
    
    while k<10000:
        k += 1
    
        for idx1 in range(samp_sz):
            g1 = beta.dot(x[idx1,:]) + beta0
            e1 = g1 - y[idx1]
            if g1*y[idx1] < 1-1e-4 or (alpha[idx1]>0 and g1*y[idx1]>1.0001):
    
                idx2 = selectJrand(idx1, samp_sz)
    
                g2 = beta.dot(x[idx2,:]) + beta0
                e2 = g2 - y[idx2]
                
                if abs(e2 - e1) < 1e-4:
                    continue
                else:
                    alpha1 = alpha[idx1].copy()
                    alpha2 = alpha[idx2].copy()
                    
#                    print('Selected indices: ', idx1, idx2)
                    
                    C = alpha[idx1]*y[idx1] + alpha[idx2]*y[idx2]
                    eta = KMat[idx1, idx1] - 2*KMat[idx1, idx2] + KMat[idx2, idx2]
                    
                    alpha[idx2] += y[idx2] * (e1-e2) / eta
                    
                    L = 0
                    H = 0
                    if y[idx2]!=y[idx1]:
                        L = max(0, alpha2-alpha1)
                        alpha[idx2] = max(L, alpha[idx2])
                    else:
                        H = alpha1+alpha2
                        alpha[idx2] = min(max(0, alpha[idx2]), H)
                        
                    if abs(alpha[idx2]-alpha2)<1e-4:
                        continue
                    
    #                print('why: ', L, H, alpha[idx2])
                        
                    alpha[idx1] = (C-alpha[idx2]*y[idx2])*y[idx1]
                    print('alpha: ', alpha[idx1], alpha[idx2])
    
                    beta = get_beta(alpha, y, x)
            
                    b1 = y[idx1] - beta.dot(x[idx1, :])
                    b2 = y[idx2] - beta.dot(x[idx2, :])
                    
                    if alpha[idx2]>0:
                        beta0 = b1
                    elif alpha[idx1]>0:
                        beta0 = b2
                    else:
                        print(alpha[idx1], alpha[idx2])
                        print("impossible")          
                        
                    print(beta, beta0, '\n')
                
        
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
    plot_scatter(ps)
    
    samp_sz, feat_sz = ps.iloc[:, :-1].shape
    
    data = np.mat(ps)
    x = np.array(ps.iloc[:, :-1])
    y = np.array(ps.iloc[:, -1])

    alpha = np.zeros(samp_sz)
    
    svm(x, y, alpha)
   
    