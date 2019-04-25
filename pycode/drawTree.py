#!/usr/bin/python

import pickle
import mlx
import argparse

TREEPATH='../cchf/cchfl_trees/75_cchfl_P1909.pkl'
ofile='tmp'





parser = argparse.ArgumentParser(description='Example with non-optional arguments')

parser.add_argument('-treepath', dest='TREEPATH', action="store", type=str,
                    default='./tree.pkl',help=" path to tree")
parser.add_argument('-ofile', dest='ofile', action="store", type=str,
                    default='tmp',help=" outfile")

args_=parser.parse_args()


with open(args_.TREEPATH, 'rb') as f:
    TR=pickle.load(f)

mlx.tree_export(TR,outfilename=args_.ofile+'.dot',
                leaves_parallel=False,
                rounded=True,
                filled=True,
                TYPE='straight',
                BGCOLOR='transparent',
                legend=False,
                LIGHT=1,
                LABELCOL='white',
                TEXTCOL='black',
                EDGECOLOR='white',
                EXEC=True)
