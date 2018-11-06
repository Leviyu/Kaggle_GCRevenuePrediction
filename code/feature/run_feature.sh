#!/home/ubuntu/anaconda3/bin//python3
'''
Feature Exploration


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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

##########################################################################################
##run_id = sys.argv[1]

run_id = "T1"
print("----> Working on experiment: ",run_id)
target = 'totals.transactionRevenue'

# train_df = load_df('../data/train.csv')
# test_df = load_df('../data/test.csv')
train_df = load_df('../../data/train1.csv')
test_df = load_df('../../data/test1.csv')

##########################################################################################
# Clean target
train_df[target].fillna(0,inplace=True)
train_df[target] = train_df[target].astype('float')
train_df[target] = np.log1p(train_df[target])
train_df['fullVisitorId'] = train_df['fullVisitorId'].astype('str')
test_df['fullVisitorId'] = test_df['fullVisitorId'].astype('str')
# train_df[target].unique()


##########################################################################################
sing_value_drop_list = ['socialEngagementType', 'device.browserSize',
       'device.browserVersion', 'device.flashVersion', 'device.language',
       'device.mobileDeviceBranding', 'device.mobileDeviceInfo',
       'device.mobileDeviceMarketingName', 'device.mobileDeviceModel',
       'device.mobileInputSelector', 'device.operatingSystemVersion',
       'device.screenColors', 'device.screenResolution',
       'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude',
       'geoNetwork.networkLocation', 'totals.visits',
       'trafficSource.adwordsClickInfo.criteriaParameters']

# sing_value_drop_list
drop_col(train_df,sing_value_drop_list)
drop_col(test_df,sing_value_drop_list)


##########################################################################################
# Catergory different kind of features
to_drop = []
category_feature = []
numerical_feature = []

##########################################################################################
# Deal with visitStartTime
feat = 'visitStartTime'
clean_start_time(train_df)
clean_start_time(test_df)
to_drop.append(feat)
# os._exit(1)
##########################################################################################
# Drop col in train
to_drop.append('trafficSource.campaignCode')
##########################################################################################
category_feature.append('channelGrouping')
to_drop.append('date')
feat = 'sessionId'
to_drop.append(feat)
feat = 'visitId'
to_drop.append(feat)


##########################################################################################
print("--> add extra columns based on fullVisitorId")
# 1. some visitor visited more then once, and they are more likely to buy items
# add another column, the number of times the this visitor visited the store
check_visit_id(train_df)
check_visit_id(test_df)
# 2. for visitor who bought items, it is more likely to buy again
# add another column, 1 if visitor bough anything, 0 if not
check_visit_revenue(train_df,train_df)
check_visit_revenue(test_df,train_df)

##########################################################################################
feat = 'visitNumber'
clearRare(feat,train_df,test_df,tag=16)
check_me(feat)
numerical_feature.append(feat)

##########################################################################################
feat = 'visitFreq'
clearRare(feat,train_df,test_df,tag=19)
check_me(feat)
numerical_feature.append(feat)
##########################################################################################
to_drop.append('visitStartTime')


##########################################################################################
feat = 'device.browser'
missing_value = 'other'
clearRare(feat,train_df,test_df,tag=missing_value)
check_me(feat)
# numerical_feature.append(feat)
category_feature.append(feat)
##########################################################################################
feat = 'device.deviceCategory'
check_me(feat)
category_feature.append(feat)
##########################################################################################
feat = 'device.isMobile'
# clearRare(feat)
# for df in (train_df,test_df):
#     mask = ( df[feat] == 'other')
#     df.loc[mask,feat] = 19
check_me(feat)
# numerical_feature.append(feat)
category_feature.append(feat)



cat_list1 = [
    'device.operatingSystem',
    'geoNetwork.city',
    'geoNetwork.region',
    'geoNetwork.region',
    'geoNetwork.subContinent',
]



for feat in cat_list1:
    missing_value = 'other'
    clearRare(feat,train_df,test_df,tag=missing_value)
    check_me(feat)
    # numerical_feature.append(feat)
    category_feature.append(feat)

num_list1 = [
    'totals.hits',
    'totals.bounces',
    'totals.newVisits',
    

]
for feat in num_list1:
    missing_value = '999'
    clearRare(feat,train_df,test_df,tag=missing_value)
    check_me(feat)
    numerical_feature.append(feat)
    # category_feature.append(feat)    





##########################################################################################
# replace dot with space in feature geonetwork
clean_networkdomain(train_df,test_df)

##########################################################################################
feat = 'geoNetwork.metro'
missing_value = 'other'
clearRare(feat,train_df,test_df,tag=missing_value)
category_feature.append(feat)
# numerical_feature.append(feat)
check_me(feat)








os._exit(1)






##########################################################################################
# reduce feature dimension
feat = 'trafficSource.keyword'        
print("--> Reduce Dimension for:",feat)
train_df[feat] = train_df[feat].apply(lambda x: clean_keys(x))
test_df[feat] = test_df[feat].apply(lambda x: clean_keys(x))

feat = 'trafficSource.source'
train_df[feat] = train_df[feat].apply(lambda x: clean_keys(x))
test_df[feat] = test_df[feat].apply(lambda x: clean_keys(x))




def clean_keys(df):
    df['trafficSource.keyword'] =  df['trafficSource.keyword'].fillna('(not provided)')
    
clean_keys(train_df)
clean_keys(test_df)


##########################################################################################
# Deal with visitStartTime
print("---> Convert visitStartTime into multiple features")
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





# check_me('visitFreq',train_df)





##########################################################################################
print("----> Deal with networkDomain feature")
feat = 'geoNetwork.networkDomain'
for df in (test_df,train_df):
    print("--->")
    df[feat] = df[feat].apply(lambda x:x.replace('.',' '))

# Use tfidfVectorizer to extract feature
Tvect = TfidfVectorizer(ngram_range=(1,2),max_features=20000)
vect = Tvect.fit(train_df[feat])
train_vect = vect.transform(train_df[feat])
test_vect = vect.transform(test_df[feat])
svd = TruncatedSVD(n_components=10)
vect_cols = ['vect'+str(x) for x in range(1,11)]
df_train_vect = pd.DataFrame(svd.fit_transform(train_vect),columns=vect_cols)
df_test_vect = pd.DataFrame(svd.fit_transform(test_vect),columns=vect_cols)
train_df = pd.concat([train_df,df_train_vect],axis=1)
test_df = pd.concat([test_df,df_test_vect],axis=1)

##########################################################################################
print("--> Search for features with items that have very low frequency \
    and clean them into others ")
##########################################################################################
combine_small_features_list = [
    'geoNetwork.city',
    'geoNetwork.region',
    'trafficSource.keyword',
    'trafficSource.source'
]
for col in combine_small_features_list:
    clearRare(train_df,test_df,columnname=col, limit = 1000)




##########################################################################################


# print(train_df['totals.hits'])

for col in cat_list:
    print("-----> Encoding for: ", col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform( list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform( list( test_df[col].values.astype('str')))
    



##########################################################################################
print("---> drop more features that we dont need")
to_drop = [
    'geoNetwork.networkDomain'
]

##########################################################################################


# output to hdf format
output_col = [x for x in train_df.columns if x not in to_drop]
train_out = train_df[output_col]
output_col = [x for x in test_df.columns if x not in to_drop]
test_out = test_df[output_col]
print("---> output features into hdf format")
train_df.to_hdf('../../data/train_df.h5',key='train_df',format='table')
test_df.to_hdf('../../data/test_df.h5',key='test_df',format='table')





##########################################################################################




##########################################################################################






##########################################################################################







##########################################################################################






##########################################################################################

























