#!/bin/bash

FILEA=$1
FILEB=$2
p=$3

./code/bin/makesample2 -f $FILEA -o ${FILEA/.txt/.csv} -M 0 -R '_n' -p $p
./code/bin/makesample2 -f $FILEB -o ${FILEB/.txt/.csv} -M 0 -R '_t' -p $p

./code/ntb/matchcolumns.py ${FILEA/.txt/.csv} ${FILEB/.txt/.csv}

FA=${FILEA/.txt/.csv}
FB=${FILEB/.txt/.csv}

cut -d "," -f 1-2000 $FA > ${FA/.csv/matched2000.csv}
cut -d "," -f 1-2000 $FB > ${FB/.csv/matched2000.csv}

