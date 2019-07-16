#!/bin/bash

FEATURES="P157 P158 P159 P205 P207 P208 P290 P291 P292 P240 P241 P242 P77 \
P78 P137 P157 P400"

PREF=$1

for YR in {2000..2018}
do
    RES=`./rfx.py --response host --file "$1"_"$YR"_"$((YR+1))".csv --filex "$1"_"$YR"_"$((YR+1))".csv --ntr 100 --cor 11 --varimp True --verbose False `
    #--inconly P157 P158 P159 P205 P207 P208 P290 P291 P292 P240 P241 P242 P77 P78 P137 P157 host P400
    #H=`echo $RES | grep host | awk '{print $NF}'`
    echo $YR  $RES
done

