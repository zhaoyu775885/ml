# -*- coding: utf-8 -*-

import numpy as np

def lda(train, train_label, test):
    row, col = train.shape
    sz = len(train_label)
    cnt_label = set(train_label)
    num_class = len(cnt_label)
    dict_label = {x:0 for x in cnt_label}
    
    center_mat = np.mat(np.zeros([num_class, col]))
    mean_vec = np.mat(np.zeros([col, 1]))
    for i in range(sz):
        dict_label[train_label[i]] += 1
        center_mat[train_label[i], :] += train[i]
    
    for it in cnt_label:
        mean_vec += center_mat[it, :].T    
        center_mat[it] *= 1.0/dict_label[it]

    mean_vec *= 1.0/sz
    
    Sb = np.mat(np.zeros([col, col]))
    Sw = np.mat(np.zeros([col, col]))
    
    for i in range(sz):
        tmp = train[i, :].T - center_mat[train_label[i], :].T
        Sw += tmp * tmp.T
    for i in range(num_class):
        tmp = center_mat[i].T - mean_vec
        Sb += tmp * tmp.T
        
    Fmat = Sw.I * Sb
    eig_val, eig_vec = np.linalg.eig(Fmat)
    eig_vec = eig_vec[:, :num_class]

#    针对方案1
    test_label = []
    test = test * eig_vec    
    center_mat = center_mat * eig_vec
    test_n, col = test.shape
    for i in range(test_n):
        dist = 1e9
        record = -1
        for j in range(num_class):
            tmp = np.linalg.norm(test[i, :]-center_mat[j])
            if tmp < dist:
                record = j
                dist = tmp
        test_label.append(record)
    return test_label

#    针对方案2    
#    train = train * eig_vec
#    test = test * eig_vec
#    return train, test, center_mat

