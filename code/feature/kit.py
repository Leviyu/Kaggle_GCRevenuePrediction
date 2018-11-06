import pandas as pd
import json
from pandas.io.json import json_normalize
import os
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
  




target = 'totals.transactionRevenue'
global train_df
global test_df

def load_df(csv_path='../data/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
    #   column_as_df = df[column].apply(lambda x: pd.Series(x))
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
def drop_col(df,to_drop):
    for col in to_drop:
        if col in df.columns:
            print("--> now dropping: ",col)
            df.drop(col,axis=1,inplace=True)
def clean_keys(tag):
  '''
  # clean source
  # reduce cluter for some features
  '''
  tag = str(tag).lower()
  simple_keys = {'google':'google',
                 'youtube':'youtube',
                 'tub':'youtube',
                 'you':'youtube',
                 'goog':'google',
                 'yu':'youtube',
                 'content targeting':'content targeting',
                 'shirt':'tshirt',
                 'android':'android',
                 'bottle':'bottle',
                 'ube':'youtube',
                 'gle':'google',
                 '谷歌':'google',
                 'yahoo':'yahoo',
                 'redd':'reddit',
                 'twit':'twitter',
                 'face':'facebook',
                 'tumb':'tumblr',
                 'what':'whatsapp',
                 'pint':'pinterest',
                 'meet':'meetup',
                 'baidu':'baidu',
                 'quora':'quora',
                 'bing':'bing'
                }
  for key,value in simple_keys.items():
      if key in tag:
          return value
  return tag
def check_visit_id(df):
    feat = 'fullVisitorId'
    visit_freq = df.groupby(feat)[feat].count()
    def fun1(x):
        return visit_freq[x]

    df['visitFreq'] = df[feat].apply(lambda x:fun1(x))
def check_visit_revenue(df,train_df):
    feat = 'fullVisitorId'
    visitor_revenue = train_df.groupby(feat)[target].sum()
    def fun2(x):
        if x not in visitor_revenue.index:
            return 0
        revenue = visitor_revenue[x]
        if revenue == 0:
            return 0
        else:
            return 1
    df['RichGuys'] = df[feat].apply(lambda x: fun2(x))
def check_me(feat,train_df):
    target='totals.transactionRevenue'
    #     print(train_df[feat]
    # two plots needed for non-zero revenue cases
    # 1. plot of relationship between feature and target, we only show the top 10
    # 2. plot the frequency of each dimension of current feature, based on the same 10 
    # categories we show in #1
    no_zero_df = train_df[train_df[target] > 0]
    fig,(ax1,ax2) = plt.subplots(2,1)
    no_zero_df.groupby(feat)[target].sum().plot(x=feat,y=target,kind='bar',logy=True,figsize=(12,5)
                                   ,ax=ax1 )
    no_zero_df.groupby(feat)[feat].count().plot(ax=ax2)

    print(" -----> totoal size of current feature is:",no_zero_df.groupby(feat)[feat].size().shape[0])
    plt.show()   
def check_me2(feat,train_df):
    target='totals.transactionRevenue'
    # two plots needed for non-zero revenue cases
    # 1. plot of relationship between feature and target, we only show the top 10
    # 2. plot the frequency of each dimension of current feature, based on the same 10 
    # categories we show in #1

    no_zero_df = train_df[train_df[target] > 0]
    unique_num = no_zero_df.groupby(feat)[feat].count().shape[0]
    unique_num_all = train_df.groupby(feat)[feat].count().shape[0]
    
    ## print features with frequency > 0.001
    freq = train_df.groupby(feat)[feat].count() / train_df.shape[0]
    freq_feat = freq[freq.values > 0.001].index
    print("------> freq > 0.001 feature is: ",freq_feat.T)
    
    
    print(" -----> Unique Feature with Revenue/All:",unique_num,unique_num_all)
    meta = check_meta_df(train_df,feat)
    print(meta[meta.index==feat].T)
    
    if unique_num_all > 400:
        print("Current feature have more then 100 dimensions, break")
        return
    #     print(no_zero_df.groupby(feat)[target].sum())
    fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1)
    fig.subplots_adjust(hspace=0.6)
    plt.title(feat)
    no_zero_df.groupby(feat)[target].sum().plot(x=feat,y=target,kind='bar',figsize=(12,10)
                                   ,ax=ax1)
    no_zero_df.groupby(feat)[feat].count().plot(ax=ax2,kind='bar')
    
    train_df.groupby(feat)[target].sum().plot(x=feat,y=target,kind='bar'
                                   ,ax=ax3)
    train_df.groupby(feat)[feat].count().plot(ax=ax4,kind='bar') 
def clearRare(columnname,train,test, per_limit = 0.001,tag='other'):
    # you may search for rare categories in train, train&test, or just test
    #vc = pd.concat([train[columnname], test[columnname]], sort=False).value_counts()
    vc = test[columnname].value_counts()

    dim = train.shape[0]
    limit = dim * per_limit
    common = vc > limit
    common = set(common.index[common].values)
    print("Set", sum(vc <= limit), columnname, "categories to 'other';", end=" ")
    
    train.loc[train[columnname].map(lambda x: x not in common), columnname] = tag
    test.loc[test[columnname].map(lambda x: x not in common), columnname] = tag
    print("now there are", train[columnname].nunique(), "categories in train")
def clean_current_feature(feat, feat_type, missing_fill,combine_value,
  train_df,test_df,numerical_feature,category_feature):
  print("----> Clean feature for:",feat)
  if missing_fill == "NULL":
    # print("NULL passed")
    pass
  else:
    # print("---> input missing value",missing_fill)
    train_df[feat].fillna(missing_fill,inplace=True)
    test_df[feat].fillna(missing_fill,inplace=True)

  if combine_value == "NULL":
      pass
  else:
      clearRare(feat,train_df,test_df,tag=combine_value)
      
  if feat_type == 'numerical':
      numerical_feature.append(feat)
      # print(feat)
      # print(train_df[feat].isnull().sum())
      # print(train_df[feat])
      train_df[feat] = train_df[feat].astype('float')
      test_df[feat] = test_df[feat].astype('float')
  else:
      category_feature.append(feat)
      train_df[feat] = train_df[feat].astype('str')
      test_df[feat] = test_df[feat].astype('str')
        
    # check_me(feat)
def check_meta_df(df):
    target='totals.transactionRevenue'
    # this function prints the meta info that I would like to see for each feature
    meta_feat = ['type','uniqueCount','q01count','q05count','min','max','mean']
    meta = pd.DataFrame(columns=meta_feat)
    for index,col_name in enumerate(df.columns):
        col = df[col_name]
        #         print('--> on: ',col_name)
        meta.at[col_name,'type'] = col.dtypes
        meta.at[col_name,'uniqueCount'] = col.unique().shape[0]

        freq = df.groupby(col_name)[col_name].count() / df.shape[0]
        meta.at[col_name,'q01count'] = (freq < 0.001).sum() / freq.shape[0]
        meta.at[col_name,'q05count'] = (freq < 0.005).sum() / freq.shape[0]
        if 'float' in str(df[col_name].dtype) or 'int' in str(df[col_name].dtype):
            meta.at[col_name,'min'] = col.min()
            meta.at[col_name,'max'] = col.max()
            meta.at[col_name,'mean'] = col.mean()
            meta.at[col_name,'skewness'] = col.skew()        
    print(meta)
def get_unique_col(df):
  out = pd.DataFrame()
  for col in df.columns:
    num_uniq = df.groupby(col)[col].count().shape[0]
    out.loc[col,'num'] = num_uniq

  return out
def clean_start_time(df):
  feat = 'visitStartTime'
  # Deal with visitStartTime
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
def check_me(feat):
  pass
  return
def clean_networkdomain(train_df,test_df,numerical_feature):
  print("----> Clean networkDomain")
  # replace dot with space in feature geonetwork
  feat = 'geoNetwork.networkDomain'
  for df in (train_df,test_df):
    df[feat] = df[feat].apply(lambda x:x.replace('.',' '))

  # Use tfidfVectorizer to extract feature
  Tvect = TfidfVectorizer(ngram_range=(1,2),max_features=20000)
  vect = Tvect.fit(train_df[feat])

  train_vect = vect.transform(train_df[feat])
  test_vect = vect.transform(test_df[feat])

  # dimension reduction on extracted feature
  svd = TruncatedSVD(n_components=10)
  vect_cols = ['vect'+str(x) for x in range(1,11)]
  df_train_vect = pd.DataFrame(svd.fit_transform(train_vect),columns=vect_cols)
  df_test_vect = pd.DataFrame(svd.fit_transform(test_vect),columns=vect_cols)
  train_df = pd.concat([train_df,df_train_vect],axis=1)
  # print(train_df.columns)
  test_df = pd.concat([test_df,df_test_vect],axis=1)
  for vv in vect_cols:
    numerical_feature.append(vv)
  return train_df,test_df
def output_hdf(train_df,test_df,run_id):
  # output to hdf format
  train_out = train_df
  test_out = test_df

  train_file = "../../data/train_df."+run_id+".h5"
  test_file = "../../data/test_df."+run_id+".h5"

  train_df.to_hdf(train_file,key='train_df',format='table')
  test_df.to_hdf(test_file,key='test_df',format='table')
  print("---> Write to hdf done")





def double_check_feature_type(train_df,test_df,feat_list,feat_type):
  for feat in feat_list:
    print("---> Checking feature type for: ", feat)
    for df in train_df,test_df:
      df[feat] = df[feat].astype(feat_type)






