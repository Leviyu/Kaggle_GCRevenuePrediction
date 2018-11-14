#!/home/ubuntu/anaconda3/bin//python3
# GA CUSTOMER REVENUE COMPETITION
# Updated kernel (11/11) with v2 files
# Read and preprocess all columns, except hits.

import gc
import os
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
import time
import sys
from ast import literal_eval
pd.set_option('display.max_columns', 500)


def load_df(file_name = 'train_v2.csv', nrows = None,skiprows=None):
    """Read csv and convert json columns."""
    
    USE_COLUMNS = [
        'channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork',
        'socialEngagementType', 'totals', 'trafficSource', 'visitId',
        'visitNumber', 'visitStartTime', 'customDimensions',
        'hits'
    ]

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv('../data/{}'.format(file_name),
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, nrows=nrows, usecols=USE_COLUMNS,skiprows=skiprows)
    print("--> reading csv finished")
    for column in JSON_COLUMNS:
        print("-------> Working on column:",column)
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
    print("----> convert json finished")
    # Normalize customDimensions
    df['customDimensions']=df['customDimensions'].apply(literal_eval)
    df['customDimensions']=df['customDimensions'].str[0]
    df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)

    column_as_df = json_normalize(df['customDimensions'])
    column_as_df.columns = [f"customDimensions_{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop('customDimensions', axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("customDimensions conversiont finished")
    # Normalize hits
    feat = 'hits'
    df[feat]=df[feat].apply(literal_eval)
    df[feat]=df[feat].str[0]
    ##print(df[feat])
    df[feat]=df[feat].apply(lambda x: {'index':np.NaN} if pd.isnull(x) else x)

    column_as_df = json_normalize(df[feat])
    column_as_df.columns = [f"hits_{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop('hits', axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("---> hits conversion finished")


    # Normalize hits
    feat = 'hits_promotion'
    df[feat]=df[feat].str[0]
    ##//df[feat]=df[feat].apply(literal_eval)
    ##print(df[feat])
    df[feat]=df[feat].apply(lambda x: {'index':np.NaN} if pd.isnull(x) else x)

    column_as_df = json_normalize(df[feat])
    column_as_df.columns = [f"hits_promotion_{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop(feat, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("---> hits_promotion conversion finished")


    # Normalize hits
    feat = 'hits_product'
    df[feat]=df[feat].str[0]
    ##//df[feat]=df[feat].apply(literal_eval)
    ##print(df[feat])
    df[feat]=df[feat].apply(lambda x: {'index':np.NaN} if pd.isnull(x) else x)

    column_as_df = json_normalize(df[feat])
    column_as_df.columns = [f"hits_product_{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop(feat, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("---> hits_promotion conversion finished")







    return df

    
def pipeline():
    timer = time.time()
    train = load_df('train_v2.csv')
    # Drop constant columns in train and test
    const_cols = [c for c in train.columns if train[c].nunique(dropna=False) < 2]
    const_cols.append('customDimensions_index')  # Also not usefull
    train.drop(const_cols, axis=1, inplace=True)
    # Drop campaignCode (has only 1 example that is not NaN) - only on train set
    train.drop('trafficSource_campaignCode', axis=1, inplace=True)
    # Save as pickle file (could be hdf5 or feather too)
    train.to_pickle('train_all.pkl')
    print("Train shape", train.shape)
    del train; gc.collect()
    
    test = load_df('test_v2.csv')
    # Drop constant columns in train
    test.drop(const_cols, axis=1, inplace=True)
    # Save as pickle file (could be hdf5 or feather too)
    test.to_pickle('test_all.pkl')
    print("Test shape", test.shape)
    print("Pipeline completed in {}s".format(time.time() - timer))
    

##train_df = load_df("train_v2.csv",nrows=)
job = sys.argv[1]

print("job is {}".format(job))
job = int(job)
job_per = sys.argv[2]
job_per = int(job_per)
job_beg = job * job_per 
job_end = (job +1) * job_per
totoal_min = 1
total_max = 2000000
skip_row = [x for x in range(totoal_min,total_max) if x not in range(job_beg,job_end)]
train_df = load_df('train_v2.csv',nrows=job_per,skiprows=skip_row)
train_df.to_csv("../data/train.tmp.{}".format(job),index=False)
