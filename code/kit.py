import pandas as pd
import json
from pandas.io.json import json_normalize











def load_df(csv_path='../data/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
#         column_as_df = df[column].apply(lambda x: pd.Series(x))
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

def drop_col(df,to_drop):
    for col in to_drop:
        if col in df.columns:
            print("--> now dropping: ",col)
            df.drop(col,axis=1,inplace=True)



# clean source
# reduce cluter for some features
def clean_keys(tag):
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



