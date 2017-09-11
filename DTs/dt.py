#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:01:22 2017

@author: zhaoyu
"""

import numpy as np
from sklearn.datasets import load_iris

def createDataSet():
    data_set = [[1, 1, 'y'],
                [1, 1, 'y'],
                [1, 0, 'n'],
                [0, 1, 'n'],
                [0, 1, 'n']]
    label_set = ['no surface', 'flippers']
    return data_set, label_set
    
def calShannonEnt(data_set):
    class_label_set = {}
    data_number = 0
    for item in data_set:
        data_number += 1
        class_label = item[-1]
        if class_label not in class_label_set:
            class_label_set[class_label] = 1
        else:
            class_label_set[class_label] += 1
    ShannonEnt = 0
    for key in class_label_set:
        p_key = class_label_set[key] / data_number
        ShannonEnt += -p_key * np.log2(p_key)
    
#    print(data_number, ShannonEnt)
#    print(class_label_set)
        
    return ShannonEnt

def splitDdataSet(data_set, index, value):
    retDataSet = []
    for featVec in data_set[:]:
        if featVec[index] == value:
            newVec = featVec[:index]
            newVec.extend(featVec[index+1:])
            retDataSet.append(newVec)
            
    return retDataSet
    
def chooseBestFeat(data_set):
    feat_number = len(data_set[0]) - 1
#    print(feat_number)
    
    original_ent = calShannonEnt(data_set)
    
    best_info_gain = 0
    best_feat_index = -1
    
    for it in range(feat_number):
        feat_set = [sample[it] for sample in data_set]
        feat_uni_set = set(feat_set)
#        print(feat_uni_set)
        new_ent = 0
        for feat in feat_uni_set:
            sub_feat_set = splitDdataSet(data_set, it, feat)
            p_sub = len(sub_feat_set) / len(data_set)
            new_ent += p_sub*calShannonEnt(sub_feat_set)

#        print(new_ent, original_ent)
        info_gain = original_ent - new_ent
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feat_index = it        
#    print(best_feat_index)
    
    return best_feat_index
    
def createTree(data_set, label_set):
    classList = [it[-1] for it in data_set]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    elif len(classList) == 1:
        return classList[0]
    else:
        bestFeat = chooseBestFeat(data_set)
        bestFeatLabel = label_set[bestFeat]
        cur_tree = {bestFeatLabel:{}}
        del(label_set[bestFeatLabel])
        feat
    
    
if __name__ == '__main__':
    data_set, label_set = createDataSet()
    
#    sub_set_1 = splitDdataSet(data_set, 0, 0)
#    print(sub_set_1, calShannonEnt(sub_set_1))
#    sub_set_2 = splitDdataSet(data_set, 0, 1)
#    print(sub_set_2, calShannonEnt(sub_set_2))
    
    
    best_feat = chooseBestFeat(data_set)
    
    print(best_feat)
    
    
    