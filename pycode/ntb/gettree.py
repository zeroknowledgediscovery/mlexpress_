#!/usr/bin/python

import numpy as np
import os
import sys
sys.path.append('../')
import mlx as ml 
import warnings
import operator
import pickle
import glob
import pylab as plt
import pandas as pd
from tqdm import tqdm
DEBUG=False


def loadTree(PATH_TO_TREES):
    TREE={}
    TREES=glob.glob(PATH_TO_TREES)
    for filename in tqdm(TREES):
        with open(filename,'rb') as f:
            TR = pickle.load(f)
            #TREE[filename]=TR
        f.close()
        index=os.path.splitext(os.path.basename(filename))[0].split('_')[-1]
        TREE[index]=TR
    return TREE
PATH_TO_TREES='../../cchf/cchfl_trees/*pkl'


TREE=loadTree(PATH_TO_TREES)

for I,TR in TREE.iteritems():
    TR=TREE[I]
    if I=='P419':
        ml.tree_export(TR,outfilename=I+'.dot',
                       LIGHT=0.45,
                       legend=False,
                       LABELCOL='black',
                       TEXTCOL='white',
                       TYPE='ortho',
                       leaves_parallel=False)
