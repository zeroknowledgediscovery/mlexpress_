#!/bin/bash

FEATURE=$1
THRESHOLD=0.5

RES='res1.dat'
PROG=" /home/ishanu/ZED/Research/mlexpress_/pycode/dec_tree_R.py "
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

if [ $# -gt 4 ] ; then 
    THRESHOLD=$5
fi


STOP=0

while [ $STOP -eq 0 ]
do
    echo $FEATURE

    
    A=`   $PROG  --file $FILE --filex $FILE --varimp True --response $FEATURE --zerodel B --del CELL $DEL --importance_threshold $THRESHOLD`

    echo $A

    exit
    
    if [ "$A" == "" ] ; then
	grep -v $FEATURE tmpsrc > tmp
	mv tmp tmpsrc
    else 
	echo $A | awk 'BEGIN{ORS=""}{for(i=1;i<=NF;i++){print $i,""; if (i%3==0){print "\n"}}}' >> $RES
	awk '{print $1}' $RES | sed '/^$/d' | grep ^[A-Z] | sort | uniq > tmpsrc
	awk '{print $2}' $RES | sed '/^$/d' | grep ^[A-Z] | sort | uniq > tmptgt
    fi
    REM=`comm -23  tmpsrc tmptgt`
    
    if [ "$REM" == "" ] ; then
	STOP=1
    else
	FEATURE=`echo $REM | awk '{print $1}'`
    fi 
done






