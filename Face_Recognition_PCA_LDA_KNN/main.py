# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from PIL import Image as im
from PCA import *
from kNN import *
from LDA import *


if __name__ == '__main__':
    data_path_base = './att_faces/'
    
    sz = 40
    each_sample_sz = 10
    train_test_split = 5
    
    face_data = []
    face_label = []
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    
    for it in range(sz):
        data_path = data_path_base + 's' + str(it+1) + '/'
        for jt in range(each_sample_sz):
            file_path = data_path + str(jt+1) + '.pgm'
#            print(file_path)
            face = im.open(file_path)
            face_data.append(list(face.getdata()))
            face_label.append(it)
            if jt < train_test_split:
                train_data.append(list(face.getdata()))
                train_label.append(it)
            else:
                test_data.append(list(face.getdata()))
                test_label.append(it)

    train_mat = np.mat(train_data, dtype='float64')
    test_mat = np.mat(test_data, dtype='float64')
    

    train_mat, test_mat = pca(train_mat, test_mat, sz)
#    PCA降维，这是由于Predictor个数p大于样本数N。
#    1. 对于高维问题，kNN分类器有困难
#    2. 无法进行后续的LDA操作，因为无法找到合适的Covariance Matrix

    """
    方案1: 单纯LDA
    """
    prdt_label = lda(train_mat, train_label, test_mat)
    
    """
    方案2: LDA降维+kNN分类
    """
#    train_mat, test_mat, center_mat = lda(train_mat, train_label, test_mat)
##    LDA特征脸提取
##    1. 非必须的步骤
##    2. 在PCA中，数据维度只是降到了和样本数N一样的程度，而实际上LDA可以做到只有K-1个数据维度，K为种类数
#    
##    kNN分类器
#    k = 5
##    vec = test_mat[0, :]
#    prdt_label = kNN(train_mat, train_label, test_mat, k)
    
#    统计预测精度    
    cnt = 0
    for i in range(len(test_label)):
        if prdt_label[i] == test_label[i]:
            cnt += 1
    print(cnt*1.0/len(test_label))
    