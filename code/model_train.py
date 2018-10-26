
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




class lets_train():
    #
    # This is a class that:
    # 1. defines the models that we want to use
    # 2. including the parameter for each model a
    # 3. trains the model and predict on test set
    # 4. calculate all kinds of metrics of the model
    
    def __init__(self,train_df,test_df,target):
        
        feat_to_drop = ['visitId',
                       
                        'fullVisitorId'
                        ]
        
        
        print("---> our model pipline is created")
        self.train_df = train_df.drop(feat_to_drop,axis=1)
        self.test_df = test_df.drop(feat_to_drop,axis=1)
        self.target = target
        
#         print(self.train_df[self.target])
        self.train_x = self.train_df.drop(self.target,axis=1)
        self.train_y = np.log1p(self.train_df[target])
        
        self.test_x = self.test_df
        
        print("train shape is  ", self.train_df.shape)
        print("test shape is  ", self.test_df.shape)
#         for col in self.train_df.columns:
#             print("--> col %s unique number is %d \n" % ( col, len(self.train_df[col].unique())))
        
        
    
    def run(self):
        
        # 1. define the modes that we want to use
        self.define_models()
        
        # 2. cross-validate the models that we selected
        self.cv_models()
        
        # 3. use models to train and predict
        self.train_predict()
        
        # 4. output submitted result
        self.sub_result()
        
        print(self.MLA)
#         print(self.pred)

    
    def define_models(self):
        MLA_Columns = ["ModelName","CVScoreMean","CVScoreSTD"]
        self.MLA = pd.DataFrame(columns=MLA_Columns)
        self.models=[lgb.LGBMRegressor(
        num_leaves=30,
        min_child_samples=100,
        learning_rate=0.1,
        bagging_fraction=0.7,
        feature_fraction=0.5,
        bagging_frequency=5,
        bagging_seed=2018
        )]
    
    
        for index, model in enumerate(self.models):
            self.MLA.loc[index,'ModelName'] = model.__class__.__name__
        self.pred = pd.DataFrame(columns=self.MLA['ModelName'])
        
        for model in self.models:
            print("---> models we used include:",model.__class__.__name__)
         
        self.predictions = []
    
    def train_predict(self):
        for index, model in enumerate(self.models):
            print(" ---> Work on train&Predict for %s "% model.__class__.__name__)
            model.fit(self.train_x,self.train_y)
            model_name = model.__class__.__name__
            pred = model.predict(self.test_x)
            self.pred[model_name] = pred
    
    def cv_models(self):
        nfold = 3
        cv_split = model_selection.ShuffleSplit(n_splits=nfold,test_size=0.3,
                                       train_size=0.7,random_state=43)
        for index, model in enumerate(self.models):
            print(" ---> Work on CV for %s "% model.__class__.__name__)
            
            rmse = np.sqrt(-cross_val_score(model,self.train_x.values,self.train_y,
                                           scoring='neg_mean_squared_error',
                                           cv = cv_split))
            
    
            self.MLA.loc[index,'CVScoreMean'] = rmse.mean()
            self.MLA.loc[index,'CVScoreSTD'] = rmse.std()
            
    def sub_result(self):
        
        for model in self.models:
            model_name = model.__class__.__name__
            print("----> output submission for ", model_name)
            pred_test = self.pred[model_name]
            model_out_file = "~/aws_out/"+model_name+".csv"
            pred_test[pred_test<0] = 0
            
            sub_df = pd.DataFrame( {'fullVisitorId':test_id} )
            sub_df['PredictedLogRevenue'] = np.expm1(pred_test)
            sub_df = sub_df.groupby('fullVisitorId')['PredictedLogRevenue'].sum().reset_index()
            sub_df.columns = ['fullVisitorId','PredictedLogRevenue']
            sub_df['PredictedLogRevenue'] = np.log1p(sub_df['PredictedLogRevenue'])
            sub_df.to_csv(model_out_file,index=False)
    
            





