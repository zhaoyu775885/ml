#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:11:05 2017

@author: zhaoyu
"""
 
import numpy as np
import pandas as pd

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

A = np.mat(np.ones([4, 4]))
b = np.mat(np.ones([4, 2]))