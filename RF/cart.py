#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:43:37 2018

@author: zhaoyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_scatter(ps):
    plt.scatter(ps.iloc[:, 0], ps.iloc[:, 1])

if '__main__' == __name__:
    ps = pd.read_table('./ex00.txt', header=None)
    
    plot_scatter(ps)