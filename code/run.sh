#!/bin/tcsh



set PWD = `pwd`
set ID = $1

if( $ID == "" ) then
echo "---> ID is NULL"
exit 0
endif


set log = $PWD/LOG/log.${ID}

###########################################################################
# Define how many records that we want to use
set MAX = 5000
cd ../data
echo "---> Cut out the first $MAX line for both train and test" >! $log
csh get_sample.sh $MAX 

###########################################################################
# Run for feature processing
cd $PWD/feature
python3 run_feature.py $ID >> & $log 


###########################################################################
# Run for model training and prediction
cd $PWD/model
python3 run_model.py $ID >> & $log 
