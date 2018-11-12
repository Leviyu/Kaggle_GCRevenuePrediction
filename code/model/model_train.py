import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

## supress warning
import warnings
warnings.filterwarnings("ignore")

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as dtree
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import multiprocessing
import time

##from StackingAveragedModels import StackingAveragedModels


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds    
    def fit(self, X, y):
        # We again fit the data on clones of the original models
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            print("--> Trying to fit model :",i)
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    def predict(self, X):
        #Do the predictions of all base models on the test data and use the averaged predictions as 
        #meta-features for the final prediction which is done by the meta-model
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

class lets_train():
    # This is a class that:
    # 1. defines the models that we want to use
    # 2. including the parameter for each model a
    # 3. trains the model and predict on test set
    # 4. calculate all kinds of metrics of the model
    def __init__(self,train_df,test_df,target,work_id):
        self.ID = work_id
        feat_to_drop = [
                            'fullVisitorId'
                        ]
        
        
        print("---> our model pipline is created")
        self.test_id = test_df['fullVisitorId'].values
        self.train_df = train_df.drop(feat_to_drop,axis=1)
        self.test_df = test_df.drop(feat_to_drop,axis=1)
        self.target = target
        

        self.train_x = self.train_df.drop(self.target,axis=1)
        self.train_y = np.log1p(self.train_df[target])
        
        self.test_x = self.test_df
        
        print("train shape is  ", self.train_df.shape)
        print("test shape is  ", self.test_df.shape)
        # for col in self.train_df.columns:
        #     print("--> col %s unique number is %d \n" % ( col, len(self.train_df[col].unique())))
    def run(self):
        
        # 1. define the modes that we want to use
        self.define_models()
        
        # 2. cross-validate the models that we selected
        self.cv_models()
        
        # 3. use models to train and predict
        self.train_predict()
        
        # 4. Ensemble models
        self.ensemble_models(method="averaging")

        # 4. output submitted result
        self.sub_result()
        
        print(self.MLA)
    def define_models(self):
        MLA_Columns = ["ModelName","CVScoreMean","CVScoreSTD"]
        self.MLA = pd.DataFrame(columns=MLA_Columns)
        self.base_models={
                # "lgb" : lgb.LGBMRegressor(
                #         num_leaves=30,
                #         metric="rmse",
                #         min_child_samples=100,
                #         learning_rate=0.1,
                #         bagging_fraction=0.7,
                #         feature_fraction=0.5,
                #         bagging_seed=2018),
                "lasso":make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)),
                "elasticNet":make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)),
                # ##"KRR":KernelRidge(alpha=0.6 ),
                # ##"KRR":KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
                "gboost":GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                    max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, 
                    loss='huber', random_state =5),
                "xgboost":xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                    learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200,
                    reg_alpha=0.4640, reg_lambda=0.8571,subsample=0.5213, silent=1,
                    random_state =7, nthread = -1),
                }

        self.models1 = {
            "stack": StackingAveragedModels(
                base_models = (
                    ##self.base_models['lgb'],
                    ##self.base_models['gboost'],
                    make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)),
                    make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)),
                    ##self.base_models['lasso'],
                    ##self.base_models['elasticNet']
                    ),
                meta_model = 
                make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)))
                ##, meta_model = self.base_models['lasso'])
            }
        # construct models with 3 differnt models
        # stacked models/ xgboost/ lightgbm and stack them
        self.models2 = {
            "stack": StackingAveragedModels(
                base_models = (
                    ##self.base_models['lgb'],
                    ##self.base_models['gboost'],
                    make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)),
                    make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)),
                    ##self.base_models['lasso'],
                    ##self.base_models['elasticNet']
                    ),
                meta_model = 
                make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))),
                "gboost":GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                    max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, 
                    loss='huber', random_state =5),
                "xgboost":xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                    learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200,
                    reg_alpha=0.4640, reg_lambda=0.8571,subsample=0.5213, silent=1,
                    random_state =7, nthread = -1),
                "lgb" : lgb.LGBMRegressor(
                        num_leaves=30,
                        min_child_samples=100,
                        learning_rate=0.1,
                        bagging_fraction=0.7,
                        feature_fraction=0.5,
                        bagging_frequency=5,
                        bagging_seed=2018)
                }
        self.models3 = {
                "xgboost":xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                    learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200,
                    reg_alpha=0.4640, reg_lambda=0.8571,subsample=0.5213, silent=1,
                    random_state =7, nthread = -1)
                }
        self.models4 = {
                "lasso":make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)),
                }

        self.models = self.base_models;
        ##self.models = self.models4;
    
        index = 0
        for  model,value in self.models.items():
            ##self.MLA.loc[index,'ModelName'] = model.__class__.__name__
            self.MLA.loc[index,'ModelName'] = model
            index+=1

        self.pred = pd.DataFrame(columns=self.MLA['ModelName'])
        
        for model,value in self.models.items():
            ##print("---> models we used include:",model.__class__.__name__)
            print("---> models we used include:",model)
         
        self.predictions = []
    def train_predict(self):
        for model_name,model in self.models.items():
            print(" ---> Work on train&Predict for %s "% model)
            model.fit(self.train_x,self.train_y)
            pred_log = model.predict(self.test_x)
            pred = np.expm1(pred_log)
            self.pred[model_name] = pred
    def cv_models(self):
        nfold = 3
        cv_split = model_selection.ShuffleSplit(n_splits=nfold,test_size=0.3,
                                       train_size=0.7,random_state=43)
        index = 0
        for model_name, model in self.models.items():
            # use muiti processing to speed up this process
            print(" ---> Work on CV for %s "% model_name)
            
            start = time.time()
            rmse = np.sqrt(-cross_val_score(model,self.train_x.values,self.train_y,
                                           scoring='neg_mean_squared_error',
                                           cv = cv_split,n_jobs=5))
                                           ##cv = cv_split))
            end = time.time()
            print("  time spent: ", end-start)

            self.MLA.loc[index,'CVScoreMean'] = rmse.mean()
            self.MLA.loc[index,'CVScoreSTD'] = rmse.std()
            index+=1
    def ensemble_models(self,method):
        if method is "averaging":
            weights = np.empty( len(self.models))
            weights.fill(1)
        elif method is "stack4":
            weights = [0.6,0.1,0.1,0.1]


            dimension = self.test_df.shape[0]
            sum_pred = np.zeros(dimension)
            sum_weight = 0
            self.models['average'] = 'ensemble'
            self.pred['average'] = pd.DataFrame(np.zeros(dimension))
            index = 0
            for model_name,model in self.models.items():
                weight = weights[index]
                self.pred['average'] = self.pred['average'] + self.pred[model_name] * weight
                sum_weight = sum_weight + weight
                index +=1
            self.pred['average'] = self.pred['average'] / sum_weight            
    def sub_result(self):
        for model_name,model in self.models.items():
            print("----> output submission for ", model_name)
            pred_test = self.pred[model_name]
            work_dir = "../RUN/"+self.ID;
            ##os.mkdir(work_dir)
            cmd = "mkdir -p "+work_dir
            os.system(cmd)
            model_out_file = work_dir+"/"+model_name+".csv"
            pred_test[pred_test<0] = 0
            
            sub_df = pd.DataFrame( {'fullVisitorId':self.test_id} )
            sub_df['PredictedLogRevenue'] = pred_test
            sub_df = sub_df.groupby('fullVisitorId')['PredictedLogRevenue'].sum().reset_index()
            sub_df.columns = ['fullVisitorId','PredictedLogRevenue']
            sub_df['PredictedLogRevenue'] = np.log1p(sub_df['PredictedLogRevenue'])
            sub_df.to_csv(model_out_file,index=False)
        ## save meta info into files too
        self.MLA.to_csv(work_dir+"/MLA.csv")
        self.pred.to_csv(work_dir+"/pred.csv")

    

            


