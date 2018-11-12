#!/bin/tcsh


set PWD = `pwd`

cd $PWD/data
##kaggle competitions download -c ga-customer-revenue-prediction

echo "unzip test"
unzip test_v2.csv.zip &
unzip train_v2.csv.zip &

chmod 777 test_v2.csv
chmod 777 train_v2.csv

