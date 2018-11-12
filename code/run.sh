#!/bin/tcsh

set PWD = `pwd`
set ID = $1

if($ID == "") then
echo "--> ERROR ID IS EMPTY!"
exit 0
endif

csh $PWD/_run.sh $ID > & /dev/null &


