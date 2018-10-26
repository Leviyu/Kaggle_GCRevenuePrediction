







import os
import sys
import model_train
from model_train import lets_train

sys.path.append('.')


my_train = lets_train(train_df,test_df,'totals.transactionRevenue')
my_train.run()