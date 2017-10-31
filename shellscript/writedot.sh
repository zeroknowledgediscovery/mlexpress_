#!/bin/bash


FILE=$1
PREF=/home/ishanu/ZED/Research/mlexpress_/data/dotpref



cat $PREF > tmp.dot
awk 'NF==3{print $1, "->", $2,";"}' $FILE >> tmp.dot

echo "} " >> tmp.dot

dot -Tpng tmp.dot -o tmp.png

