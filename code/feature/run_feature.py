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
run_id = sys.argv[1]

print(run_id)
##run_id = "T1"
print("----> Working on experiment: ",run_id)
target = 'totals.transactionRevenue'


# train_df = load_df('../data/train.csv')
# test_df = load_df('../data/test.csv')
print("----> Loading dataframe, takes about 2min")
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
# # 1. some visitor visited more then once, and they are more likely to buy items
# # add another column, the number of times the this visitor visited the store
# check_visit_id(train_df)
# check_visit_id(test_df)
# # 2. for visitor who bought items, it is more likely to buy again
# # add another column, 1 if visitor bough anything, 0 if not
# check_visit_revenue(train_df,train_df)
# check_visit_revenue(test_df,train_df)

##########################################################################################
# feat = 'visitFreq'
# feat_type = 'numerical'
# #     feat_type = 'categorial'
# missing_fill = 'NULL'
# combine_value = '19'   ## it is set to 'other' by default
# clean_current_feature(feat,feat_type,missing_fill,combine_value,
#     train_df,test_df,numerical_feature,category_feature)



# ##########################################################################################
feat = 'visitNumber'
feat_type = 'numerical'
#     feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = '16'   ## it is set to 'other' by default
# combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)


##########################################################################################
feat = 'channelGrouping'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)
##########################################################################################
feat = 'device.browser'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)
##########################################################################################
feat = 'device.deviceCategory'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)
##########################################################################################
feat = 'device.isMobile'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)
##########################################################################################
feat = 'device.operatingSystem'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)
##########################################################################################
feat = 'geoNetwork.city'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'other'   ## it is set to 'other' by default
# combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'geoNetwork.metro'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'other'   ## it is set to 'other' by default
# combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'geoNetwork.country'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'other'   ## it is set to 'other' by default
# combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'geoNetwork.continent'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
# clean geoNetwork.networkDomain
feat = 'geoNetwork.networkDomain'
# train_df, test_df = clean_networkdomain(train_df,test_df,numerical_feature)
to_drop.append(feat)
# category_feature.append(feat)
# print(numerical_feature)
# print(train_df['vect2'])
# print(train_df.columns)
# os._exit(0)

##########################################################################################
feat = 'geoNetwork.region'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
# combine_value = 'other'   ## it is set to 'other' by default
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'geoNetwork.subContinent'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'totals.bounces'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)
# os._exit(0)

##########################################################################################
feat = 'totals.hits'
feat_type = 'numerical'
# feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = '35'   ## it is set to 'other' by default
# combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'totals.newVisits'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = '0'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'totals.pageviews'
feat_type = 'numerical'
# feat_type = 'categorial'
missing_fill = '0'
combine_value = '29'   ## it is set to 'other' by default
# combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'trafficSource.adContent'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'other'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'trafficSource.adwordsClickInfo.adNetworkType'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'other'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'trafficSource.adwordsClickInfo.gclId'
to_drop.append(feat)

##########################################################################################
feat = 'trafficSource.adwordsClickInfo.isVideoAd'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = True
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'trafficSource.adwordsClickInfo.page'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = '0'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)
##########################################################################################
feat = 'trafficSource.adwordsClickInfo.slot'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'other'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'trafficSource.campaign'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'trafficSource.isTrueDirect'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = False
combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)


##########################################################################################
feat = 'trafficSource.keyword'
train_df[feat] = train_df[feat].apply(lambda x: clean_keys(x))
test_df[feat] = test_df[feat].apply(lambda x: clean_keys(x))
category_feature.append(feat)

# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'Not Provided'
combine_value = 'Combined'   ## it is set to 'other' by default
# combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'trafficSource.medium'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'Not Provided'
combine_value = 'Combined'   ## it is set to 'other' by default
# combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

##########################################################################################
feat = 'trafficSource.referralPath'
to_drop.append(feat)


##########################################################################################
feat = 'trafficSource.source'
# feat_type = 'numerical'
feat_type = 'categorial'
missing_fill = 'NULL'
combine_value = 'Combined'   ## it is set to 'other' by default
# combine_value = 'NULL'   ## it is set to 'other' by default
clean_current_feature(feat,feat_type,missing_fill,combine_value,
    train_df,test_df,numerical_feature,category_feature)

# print(train_df[feat])

##########################################################################################
category_feature.append('year')
category_feature.append('month')
category_feature.append('day')
category_feature.append('hour')
category_feature.append('day_week')
category_feature.append('am_pm')
##########################################################################################

# print(numerical_feature)
numerical_feature = set(numerical_feature)
category_feature = set(category_feature)
to_drop = set(to_drop)

print("---> numerical feature num: ",len(numerical_feature))
print(numerical_feature)
print("---> Category feature num: ",len(category_feature))
print(category_feature)
print("---> Drop feature num: ",len(to_drop))
print(to_drop)
print("---> Totoal feature num: ",len(to_drop)+len(numerical_feature)
    +len(category_feature))

double_check_feature_type(train_df,test_df,numerical_feature,'float')
double_check_feature_type(train_df,test_df,category_feature,'str')




##########################################################################################
# Encode categorial features
for col in category_feature:
    print("-----> Encoding for: ", col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform( list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform( list( test_df[col].values.astype('str')))
    

##########################################################################################
for col in to_drop:
    print("--> Dropping column for:", col)
    for df in train_df,test_df:
        if col in df.columns:
            df.drop(col,axis=1,inplace=True)


##########################################################################################

output_hdf(train_df,test_df,run_id)
# output to hdf format






