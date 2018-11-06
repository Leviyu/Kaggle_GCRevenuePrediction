#!/home/ubuntu/anaconda3/bin//python3
import os
import sys
import model_train
import pandas as pd
from model_train import lets_train
from tool import *


# sys.path.append('.')

work_id = sys.argv[1]
print("--> Model training Work on ID: ",work_id)
train_file = "../../data/train_df."+work_id+".h5"
test_file = "../../data/test_df."+work_id+".h5"


train_df = pd.read_hdf(train_file,key="train_df")
test_df = pd.read_hdf(test_file,key="test_df")

print("----> train_df shape is:",train_df.shape)
# # print(train_df.columns)
my_train = lets_train(train_df,test_df,'totals.transactionRevenue',work_id)
my_train.run()

# lgb_session(train_df,test_df)

# print(train_df.shape)
