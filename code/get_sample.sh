#!/bin/tcsh


set num = $1

cat ./test0.csv |head -n $num  >! ./test1.csv
cat ./train0.csv |head -n $num >! ./train1.csv

