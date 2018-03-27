# -*- coding: utf-8 -*-

import numpy as np

def kNN_basic(mat, label, vec, k):
    dist = []
    sz, col = mat.shape
    for it in range(sz):
        ori = mat[it, :]
        tmp = ori - vec
        dist.append(np.linalg.norm(tmp))
        
    min_record = -1
    kNN_label = []
    for i in range(k):
        min_dist = 1e9
        record = -1
        for j in range(sz):
            if dist[j] <= min_record:
                continue
            if dist[j] < min_dist:
                min_dist = dist[j]
                record = j
        min_record = min_dist
        kNN_label.append(label[record])
    
    existed_label = set(kNN_label)
    cnt = {x:0 for x in existed_label}
    for it in kNN_label:
        cnt[it] += 1
    
    freq = 0
    flabel = -1
    for it in existed_label:
        if cnt[it] > freq:
            freq = cnt[it]
            flabel = it
            
    return flabel

def kNN(train_mat, train_label, test_mat, k):
    test_label = []
    row, col = test_mat.shape
    for it in range(row):
        vec = test_mat[it, :]
        label = kNN_basic(train_mat, train_label, vec, k)
        test_label.append(label)
        
    return test_label