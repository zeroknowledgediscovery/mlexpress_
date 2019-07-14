#!/bin/bash

MCFILE=$1
OUTFILE=$2


a=`cat  $MCFILE`

f=`cat front.html`
b=`cat back.html`

echo -n $f > $OUTFILE
echo -n $a >> $OUTFILE
echo -n $b >> $OUTFILE
