
# coding: utf-8

import numpy as np
import os 
import sys
sys.path.append('../')
import mlx as ml 
import warnings
import operator
import pickle

with open('../../data/tree.pkl','rb') as f:
    TR = pickle.load(f)
TR.class_pred_[1]={'A':36.0,'C':138.0,'G':25.0}
TR.feature[4]='SOX2'



print ml.sampleTree(TR,{'RBM34':'C','SOX2': 'C'},sample='random',DIST=True)


#to do:

# input probability distributions instead of deterministic constraints
# example: {'RBM34':{'C':0.2, 'T':.8},'SOX2': {'A':.6,'G':.3,'T':.2}}
# simple approach: draw samples on init conditions.. calculate output dist, and average.

