
from sklearn.model_selection import GroupKFold
import numpy as np 
import lightgbm as lgb
import pandas as pd

def get_folds(df=None,n_splits=5):
	"""
	Return dataframe index correcponding to visitor group fold
	"""

	unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

	# Get folds
	folds = GroupKFold(n_splits=n_splits)
	fold_ids = []

	ids = np.arange(df.shape[0])

	for trn_vis,val_vis in folds.split(X=unique_vis,y=unique_vis,groups=unique_vis):
		fold_ids.append(
			[
				ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
				ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
			]

			)

	return fold_ids

def lgb_session(train,test):
	folds = get_folds(df=train, n_splits=5)

	y_reg = train['totals.transactionRevenue'].fillna(0)

	excluded_features = [
		'fullVisitorId',
		'RichGuys',
		'visitFreq'

	]

	train_features = [_f for _f in test.columns if _f not in excluded_features]

	# train_features = test.columns
	# print(train_features)

	importances = pd.DataFrame()
	oof_reg_preds = np.zeros(train.shape[0])
	sub_reg_preds = np.zeros(test.shape[0])
	for fold_, (trn_, val_) in enumerate(folds):
	    trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
	    val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
	    
	    reg = lgb.LGBMRegressor(
	        num_leaves=31,
	        learning_rate=0.03,
	        n_estimators=1000,
	        subsample=.9,
	        colsample_bytree=.9,
	        random_state=1
	    )
	    reg.fit(
	        trn_x, np.log1p(trn_y),
	        eval_set=[(val_x, np.log1p(val_y))],
	        early_stopping_rounds=50,
	        verbose=100,
	        eval_metric='rmse'
	    )
	    imp_df = pd.DataFrame()
	    imp_df['feature'] = train_features
	    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
	    
	    imp_df['fold'] = fold_ + 1
	    importances = pd.concat([importances, imp_df], axis=0, sort=False)
	    
	    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
	    oof_reg_preds[oof_reg_preds < 0] = 0
	    _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
	    _preds[_preds < 0] = 0
	    sub_reg_preds += np.expm1(_preds) / len(folds)
	    
	mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5