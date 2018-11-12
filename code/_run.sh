#!/bin/tcsh



set PWD = `pwd`
set ID = $1
set MAX = 5000

if( $ID == "" ) then
echo "---> ID is NULL"
exit 0
endif
echo "---> Working on ID: $ID"

set log = $PWD/LOG/log.${ID}

###########################################################################
# Define how many records that we want to use

cd ../data
echo "---> Cut out the first $MAX line for both train and test" >! $log
csh get_sample.sh $MAX  >> & $log

###########################################################################
# Run for feature processing
cd $PWD/feature
python3 run_feature.py $ID >> & $log 


###########################################################################
# Run for model training and prediction
cd $PWD/model
python3 run_model.py $ID >> & $log 
