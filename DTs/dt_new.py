#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:33:39 2017

@author: zhaoyu
"""

import numpy as np
import pandas as pd

def cal_ent(df):
    # find class types and corresponding number of cases
    case_num = len(df)
    types = list(set(df.iloc[:, -1]))
    num_each_type = {}
    for each_type in types[:]:
        num_each_type[each_type] = len(df[df.iloc[:, -1]==each_type])
        
    ent = 0
    for each_type in types[:]:
        prob = num_each_type[each_type]/case_num
#        print(prob)
        ent += -prob * np.log2(prob)
        
    return ent

def best_feat(df):
    feat_set = list(df)
    del(feat_set[-1])
    
    base_ent = cal_ent(df)
    best_ent_gain = 0
    best_feat = 0
#    son_dfs = []
    for feat in feat_set[:]:
        feat_types = list(set(df[feat]))
        tmp_ent = 0
        for each_type in feat_types[:]:
            tmp_df = df[df[feat]==each_type]
            prob = len(tmp_df) / len(df)
            tmp_ent += cal_ent(tmp_df)*prob
#            son_dfs.append(tmp_df)
#        print(son_dfs)
#        son_dfs.clear()
#        print('---------------------')
        if base_ent-tmp_ent>=best_ent_gain:
            best_ent_gain = base_ent - tmp_ent
            best_feat = feat
            
#    print(best_feat, max_ent)
    return best_feat

def leaf_node_judge(df):
    if len(list(df))!=2:
        return True
    elif len(list(set(df.iloc[:, -1])))==1:
        return True
    else:
        return 0

def voting(df):
    if len(list(df))!=1:
        raise Exception('Voting Use Error!')
    else:
        types = list(set(df.iloc[:, -1]))
        major_num = 0
        major_type = 0
        for each_type in types[:]:
            tmp_type_num = len(df[df.iloc[:, -1]==each_type])
            if tmp_type_num>major_num:
                major_num = tmp_type_num
                major_type = each_type
            
    return major_type
    
def decision_tree(df, tree):
    if len(list(set(df.iloc[:, -1])))==1:
        tree.append(list(set(df.iloc[:, -1]))[0])
    elif len(list(df))==1:
        tree.append(voting(df))
    else:
        feat_sets = list(df)
        print('1', feat_sets)
#        if len(feat_sets)==2:
#            print('-------------')
#            print(df)
#            print('-------------')
        
        feat = best_feat(df)
        print('2', feat)
        
        feat_sets.remove(feat)
#        print('3', feat_sets)
        
        one_out_df = df[feat_sets]
        
        tree.append(feat)
        
        feat_types = list(set(df[feat]))
        
        child_tree = {}
        for each_type in feat_types[:]:
            child_tree[each_type] = []
            child_df = one_out_df[df[feat]==each_type]
#            print(child_df)
            decision_tree(child_df, child_tree[each_type])
            
        tree.append(child_tree)
    
train = pd.read_csv('lenses.txt', delim_whitespace=True)
#ent = cal_ent(train)
#print(ent)
#best_feat(train)

main_dt = []
decision_tree(train, main_dt)