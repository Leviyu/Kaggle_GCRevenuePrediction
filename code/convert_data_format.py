

import pandas as pd
from feature.kit import *


max_row = 50000


train_file = "../data/train.csv"
test_file = "../data/test.csv"


train = load_df(train_file,nrows=max_row)
test = load_df(test_file,nrows=max_row)


out_train = "../data/train0.csv"
out_test = "../data/test0.csv"

train.to_csv(out_train,index=False)
test.to_csv(out_test,index=False)
