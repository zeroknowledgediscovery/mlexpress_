#!/usr/bin/python
import pandas as pd
import os
import sys

fnamea='../out_n.txt'
fnameb='../out_t.txt'
cols=-1

total = len(sys.argv)
if total > 1:
    fnamea = sys.argv[1]
if total > 2:
    fnameb = sys.argv[2]
if total > 3:
    cols = int(sys.argv[3])


dfn=pd.read_csv(fnamea)
dft=pd.read_csv(fnameb)

coln=dfn.columns
colt=dft.columns
col=[col for col in coln if col in colt]

dfn=dfn[col]
dft=dft[col]

dfn.to_csv(fnamea.replace('.txt','').replace('.csv','')+'matched.csv',index=None)
dft.to_csv(fnameb.replace('.txt','').replace('.csv','')+'matched.csv',index=None)

