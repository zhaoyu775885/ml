# -*- coding: utf-8 -*-

import numpy as np
import copy as cp

def trunc_svd(mat, num_class):
    m, n = mat.shape
    tmp_mat = np.mat(np.zeros(mat.shape))
    for j in range(n):
        tmp_mat[:, j] = mat[:, j] - mat[:, j].mean()*np.mat(np.ones([m, 1]))
    sz = min(m, n) - num_class
    u, s, v = np.linalg.svd(tmp_mat)
    u = u[:, :sz]
    s = s[:sz]
    v = v[:, :sz]
    return u, s, v

def pca(train_mat, test_mat, num_class):
    u, s, v = trunc_svd(train_mat, num_class)
    train_mat = train_mat*v
    test_mat = test_mat*v
    return train_mat, test_mat