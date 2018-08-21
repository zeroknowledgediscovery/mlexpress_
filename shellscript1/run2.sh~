#!/bin/bash

FEATURE=$1
RES='res1.dat'
PROG=" /home/ishanu/ZED/Research/mlexpress_/pycode/dec_tree_2.py "
FILE=/home/ishanu/ZED/Research/mlexpress_/data/qdat11.dat

if [ $# -lt 1 ] ; then 
    echo './prog <feature> <res> <file>'
    echo './prog DDR1 res1.dat (def) qdat11.dat (def)'
    exit
fi


if [ $# -gt 1 ] ; then 
    RES=$2
fi
if [ $# -gt 2 ] ; then 
    FILE=$3
fi

if [ $# -gt 3 ] ; then 
    DEL=$4
fi


STOP=0

while [ $STOP -eq 0 ]
do
   echo $FEATURE 
   $PROG  --file $FILE --filex $FILE --varimp True --response $FEATURE --zerodel B --del CELL $DEL >> $RES
   awk '{print $1}' $RES | sed '/^$/d' | grep ^[A-Z] | sort | uniq > tmpsrc
   awk '{print $2}' $RES | sed '/^$/d' | grep ^[A-Z] | sort | uniq > tmptgt
   REM=`comm -23  tmpsrc tmptgt`
   
   if [ -z "$REM" ] ; then
       STOP=1
   else
       FEATURE=`echo $REM | awk '{print $1}'`
   fi 
done






