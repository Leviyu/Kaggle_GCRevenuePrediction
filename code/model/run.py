#!/home/ubuntu/anaconda3/bin//python3
import os
import sys
import model_train
import pandas as pd
from model_train import lets_train

sys.path.append('.')

work_id = sys.argv[1]
print("--> Work on ID: ",work_id)



train_df = pd.read_hdf("../data/train_df.h5",key="train_df")
test_df = pd.read_hdf("../data/test_df.h5",key="test_df")


my_train = lets_train(train_df,test_df,'totals.transactionRevenue',work_id)
my_train.run()
