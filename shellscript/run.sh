#!/bin/bash

FEATURE=$1
RES=$2
RESET=1
FILE=../qdat11.dat

if [ $# -gt 2 ] ; then 
    RESET=$3
fi

if [ $# -gt 3 ] ; then 
    FILE=$4
fi

if [ $RESET -eq 1 ] ; then 
    echo -n '' > $RES
fi
STOP=0

while [ $STOP -eq 0 ]
do
    PROG='./dec_tree_2.py --file '"$FILE"' --filex '"$FILE"' --varimp True --response '"$FEATURE"' --zerodel B --del CELL'

    tmp0=`$PROG`
    tmp=`echo $tmp0 | awk '{print $1,$2,$3}'`
    if [ "$tmp" != "" ] ; then
	SRC=`echo $tmp | awk '{print $1}'`
	echo $tmp
	
	if [[ ! -z `awk '{print $1}' $RES | grep $SRC` ]] ; then
	    STOP=1
	fi

	z=`grep "$tmp" $RES`
	if [[ -z $z ]] ; then 
	    echo $tmp >> $RES
	fi
	FEATURE=$SRC
    else
	STOP=1
    fi
done

