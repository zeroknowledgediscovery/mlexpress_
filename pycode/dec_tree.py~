#!/usr/bin/python

import numpy as np
import pandas as pd
import subprocess
import os
import sys

import mlx as ml 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stat 
import argparse
import sparkline
import warnings
import tempfile

warnings.filterwarnings("ignore")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description='Example with non-optional arguments')
 
parser.add_argument('--response', dest='RESPONSE', action="store", type=str,
                    default='SPECIES',help="Response Variable")
parser.add_argument('--filea', dest='FILE', action="store", type=str,
                    default='../../data/database_may30_2017/AA_Human_20022003')
parser.add_argument('--fileb', dest='FILE2', action="store", type=str,
                    default='../../data/database_may30_2017/AA_Swine_20022003')
parser.add_argument('--fileax', dest='FILEx', action="store", type=str,
                    default='../../data/database_may30_2017/AA_Human_20042005')
parser.add_argument('--filebx', dest='FILEx2', action="store", type=str,
                    default='../../data/database_may30_2017/AA_Swine_20042005')
parser.add_argument('--spa', dest='SP1', action="store", type=str,default='Human',help="Class Name 1")
parser.add_argument('--spb', dest='SP2', action="store", type=str,default='Swine',help="Class Name 2")
parser.add_argument('--ntree', dest='NUMTREE', action="store", type=int,default=300,help="Number of trees in rndom forest")
parser.add_argument('--cores', dest='CORES', action="store", type=int,default=10,help="Number of cores to use in rndom forest")
parser.add_argument("--plot", type=str2bool, nargs='?',dest='PLOT_',
                        const=True, default=False,
                        help="Show plot")
parser.add_argument("--varimp", type=str2bool, nargs='?',dest='VARIMP',
                        const=True, default=False,
                        help="Feature importance (experimental")

parser.add_argument('--del', dest='DELETE', action="store", type=str,nargs='+', default='',help="Deleted features")
parser.add_argument('--inc', dest='INCLUDE', action="store", type=str,nargs='+', default='',help="Included features")
parser.add_argument("--verbose", type=str2bool, nargs='?',dest='VERBOSE',
                        const=True, default=False,
                        help="Verbose")
parser.add_argument('--treename', dest='TREENAME', action="store", type=str,
                    default='tmp')



results=parser.parse_args()
RESPONSE=results.RESPONSE
FILE=results.FILE
FILE2=results.FILE2
FILEx=results.FILEx
FILEx2=results.FILEx2
SP1=results.SP1
SP2=results.SP2
VERBOSE=results.VERBOSE
NUMTREE=results.NUMTREE
CORES=results.CORES
VARIMP=results.VARIMP
PLOT=results.PLOT_
DELETE=results.DELETE
INCLUDE=results.INCLUDE
TREENAME=results.TREENAME



#------------------------------
sys.stdout.write(ml.BLUE)
print "Response variable: ",RESPONSE
print FILE
print FILE2
print FILEx
print FILEx2
print "Species names: ",SP1,SP2
print "Deleted variables: ", DELETE
print "Included variables: ", INCLUDE
sys.stdout.write(ml.RESET)
#------------------------------

datatrain=ml.setdataframe(FILE,FILE2,
                          SP1=SP1,SP2=SP2,delete_=DELETE,include_=INCLUDE)
datatest=ml.setdataframe(FILEx,FILEx2,
                         SP1=SP1,SP2=SP2,delete_=DELETE,include_=INCLUDE)

CT,Pr,ACC,CF,Prx,ACCx,CFx,TR=ml.Xctree(RESPONSE__=RESPONSE,
                                       datatrain__=datatrain,
                                       datatest__=datatest,
                                       VERBOSE=VERBOSE,TREE_EXPORT=False)

#---------  PRINT -------------
sys.stdout.write(ml.RED)
print
print "DECISION TREE ACCURACY "
print "ACC (in  sample): ",ACC
print "ACC (out sample): ",ACCx

sys.stdout.write(ml.CYAN)
print "IN SAMPLE CONFUSION MATRIX:"
print CF
if CFx is not None:
    print
    print "OUT SAMPLE CONFUSION MATRIX:"
    print CFx
    sys.stdout.write(ml.RESET)
sys.stdout.write(ml.RESET)

sys.stdout.write(ml.CYAN)
if VERBOSE:
    print Pr.head()
    print Prx.head()

#TR=ml.visTree(CT,Pr,PLOT,True)
if TR is not None:
    ml.tree_export(TR,TYPE='polyline',
                   outfilename=TREENAME+"_.dot",LIGHT=1.5,BGCOLOR='gray27')
    if VARIMP:
        print "Feature Importance:"
        ml.pp.pprint(TR.significant_feature_weight_)
else:
    print 'EMPTY RULES RETURNED BY CTREE'


sys.stdout.write(ml.RESET)
#------------ EOF


