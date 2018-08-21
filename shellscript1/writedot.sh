#!/bin/bash


FILE=$1
RES='tmp.png'

if [ $# -gt 1 ] ; then
    RES=$2
fi 
PREF=/home/ishanu/ZED/Research/mlexpress_/data/dotpref

tgt=`sed -n 1p $FILE | awk '{print $2}'`


cat $PREF > tmp.dot
echo $tgt "[shape=circle, style=filled, fillcolor=red];" >> tmp.dot


awk 'NF==3{print $1, "->", $2,";"}' $FILE >> tmp.dot

echo "} " >> tmp.dot

dot -Tpdf tmp.dot -o $RES

