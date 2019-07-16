#!/bin/bash

FEATURES="P157 P158 P159 P205 P207 P208 P290 P291 P292 P240 P241 P242 P77 \
P78 P137 P157 P400"

for f in `echo $FEATURES`
do
    for YR in {2006..2018}
    do
	RES=`./rfx.py --response $f --file ./ntb/ntb/h3n2"$YR".csv --filex ./ntb/ntb/h3n2"$YR".csv --ntr 100 --cor 11 --varimp True --verbose True --inconly P157 P158 P159 P205 P207 P208 P290 P291 P292 P240 P241 P242 P77 P78 P137 P157 host P400`
	H=`echo $RES | grep host | awk '{print $NF}'`
	echo $YR $f $H
    done
done
	
