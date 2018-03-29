#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:25:42 2018

Practice K-means and Expectation-Maximization Algorithms

@author: zhaoyu
"""
import numpy as np
#import numpy.random as rd
import matplotlib.pyplot as plt

if '__main__' == __name__:
    mean1 = [1, 1]
    cov1 = np.mat([[1, 0.2], [0.2, 1]])
    mean2 = [4, 4]
    cov2 = np.mat([[2, 0.1], [0.1, 2]])
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 80).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 50).T
#    plt.scatter(x1, y1, marker='o', color='r')
#    plt.scatter(x2, y2, marker='x', color='b')
    
    all_x = np.concatenate((x1, x2), axis=0)
    all_y = np.concatenate((y1, y2), axis=0)
#    plt.scatter(all_x, all_y)
    pt = np.mat([all_x, all_y]).T
    labels = np.mat(np.zeros([pt.shape[0], 1], dtype=int))
    
    sz = pt.shape[0]
    
    K = 2
    guess1 = np.random.normal(10, 1, 2)
    guess2 = np.random.normal(1000, 1, 2)
    guess = np.mat([guess1, guess2])
    cnt = [0, 0]
    
    while True:
        tmp_guess = guess.copy()
        for i in range(2):
            it = pt[i]
            dist = 1e10
            label = -1
            for j in range(K):
                tmp_dist = np.linalg.norm(it - tmp_guess[j])
                if tmp_dist < dist:
                    dist = tmp_dist
                    label = j
            labels[i] = label
            
            print(label)
            tmp_guess[label] = (cnt[label]*tmp_guess[label]+it) / (cnt[label]+1)
            print(guess, '\n', tmp_guess)
            
            cnt[label] += 1
        guess = tmp_guess.copy()
        if np.linalg.norm(tmp_guess-guess) < 1e-5:
            break
        
        print(guess)