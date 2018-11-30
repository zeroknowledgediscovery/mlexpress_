#!/usr/bin/python
import pandas as pd

fnamea='../out_n.txt'
fnameb='../out_t.txt'

dfn=pd.read_csv(fnamea)
dft=pd.read_csv(fnameb)

coln=dfn.columns
colt=dft.columns
col=[col for col in coln if col in colt]

dfn=dfn[col]
dft=dft[col]

dfn.to_csv(fnamea.replace('.txt','')+'matched.csv',index=None)
dft.to_csv(fnameb.replace('.txt','')+'matched.csv',index=None)

