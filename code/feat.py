#!/home/ubuntu/anaconda3/bin//python3
'''
EDA


'''
__author__ = 'Hongyu Lai'

##########################################################################################
# Import library
import sys
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
sys.path.append('.')
from kit import *

##########################################################################################


##run_id = sys.argv[1]

run_id = "T1"

print("----> Working on experiment: ",run_id)

train_df = load_df('../data/train.csv')
test_df = load_df('../data/test.csv')

to_drop = [
    'trafficSource.adContent',
    'trafficSource.adwordsClickInfo.adNetworkType',
    'trafficSource.adwordsClickInfo.gclId',
    'trafficSource.adwordsClickInfo.isVideoAd',
    'trafficSource.adwordsClickInfo.page',
    'trafficSource.adwordsClickInfo.slot',
    'trafficSource.campaignCode',
    'trafficSource.referralPath'
]

drop_col(train_df,to_drop)
drop_col(test_df,to_drop)


##########################################################################################
# fillna for some columns
for df in (train_df,test_df):
#     df['totals.bounces'] = df['totals.bounces'].fillna(0).astype(int)
    feat = 'totals.newVisits'
    df[feat] = df[feat].fillna(0).astype(int)
    feat = 'totals.pageviews'
    df[feat] = df[feat].fillna(0).astype(int)
    feat = 'trafficSource.isTrueDirect'
    df[feat] = df[feat].fillna(True).astype(bool)

# just for train
feat = 'totals.transactionRevenue'
train_df[feat] = train_df[feat].fillna(0.0).astype(float)



##########################################################################################
# reduce feature dimension
feat = 'trafficSource.keyword'        
train_df[feat] = train_df[feat].apply(lambda x: clean_keys(x))
test_df[feat] = test_df[feat].apply(lambda x: clean_keys(x))




def clean_keys(df):
    df['trafficSource.keyword'] =  df['trafficSource.keyword'].fillna('(not provided)')
    
clean_keys(train_df)
clean_keys(test_df)


##########################################################################################
# Deal with visitStartTime
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime

for df in (train_df,test_df):
    df['year'] = df['visitStartTime'].apply(lambda x : 
                                            datetime.utcfromtimestamp(x).strftime("%Y"))
    df['month'] = df['visitStartTime'].apply(lambda x : 
                                            datetime.utcfromtimestamp(x).strftime("%m"))
    df['day'] = df['visitStartTime'].apply(lambda x : 
                                            datetime.utcfromtimestamp(x).strftime("%d"))
    df['hour'] = df['visitStartTime'].apply(lambda x : 
                                            datetime.utcfromtimestamp(x).strftime("%H"))
    df['day_week'] = df['visitStartTime'].apply(lambda x : 
                                            datetime.utcfromtimestamp(x).strftime("%w"))
    df['am_pm'] = df['visitStartTime'].apply(lambda x : 
                                            datetime.utcfromtimestamp(x).strftime("%p"))





