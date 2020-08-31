# -*- coding: utf-8 -*-
"""
@author: user

"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

class data_loader:
    
    def load_data(self):
        
        data = pd.read_csv('data/bank-additional-full.csv', sep = ';')
        return data

    """
    
    
    """
    
    def preprocess(self, df):
        
        # relabel columns in the data
        
        jobs = {'housemaid':'A','blue-collar':'A','technician':'A','admin.':'B','management':'B',\
       'retired':'C','student':'C','unknown':'C','entrepreneur':'D','self-employed':'D',\
       'services':'E','unemployed' : 'C'}

        df['job2'] = df['job'].map(jobs)
        
        
        education = {'basic.4y':'E1','basic.6y':'E1','basic.9y':'E1',\
            'high.school':'E1','university.degree':'E2',\
             'professional.course':'E2', 'unknown' : 'E3', 'illiterate' : 'E3'}
        
        df['education2'] = df['education'].map(education)
        
        
        months = {'mar':'Q1','apr' : 'Q1' ,'may' : 'Q1', 'jun' : 'Q2', 'jul' : 'Q2',\
         'aug' : 'Q2', 'sep' : 'Q3', 'oct' : 'Q3', 'nov' : 'Q3', 'dec' : 'Q3'}
            
        df['months_new'] = df['month'].map(months)
        
        
        df['age_cut'] = pd.cut(x=df['age'], bins=[16, 30, 45, 60,100], labels=['20s', '30s', '40s','60s'])
        
        df.loc[df['campaign'] > 15, 'campaign'] = 15
        
        df['pdays'] =  df['pdays'] + 1
        df['pdays'].replace({1000:0}, inplace = True)
        
        df.drop(df[df['marital'] == 'unknown'].index, inplace = True)
        
       
        data = df.drop(['age','duration','job','education','month','day_of_week'], axis = 1)
        
        return data
        
    
    def encode_target(self,df):
        
        label_encoder = LabelEncoder()
        df['y']= label_encoder.fit_transform(df['y'])
        
        
    def scale_columns(self, df):
        
        num = []
        cat = []
    
        for feature in df.columns:
            if df[feature].dtype in ['object',]:
                cat.append(feature)
            if df[feature].dtype in ['float','int','int64','int32']:
                num.append(feature)
                
        X = df.drop('y', axis=1).reset_index(drop = True)

        scaler = StandardScaler()
        
        #pt = PowerTransformer()
        #rt = RobustScaler()

        cat_X = pd.get_dummies(X[cat])

        scaled_X = scaler.fit_transform(X[num])
        #power_X = pt.fit_transform(X[num])
        #robust_X = rt.fit_transform(X[num])


        scaled_df = pd.DataFrame(scaled_X , columns = num)
        #power_df = pd.DataFrame(power_X , columns = num_under)
        #robust_df = pd.DataFrame(robust_X , columns = num_under)
        #log_X = np.log(X[num_under])
        
        
        data = pd.merge(cat_X, scaled_df, how = 'inner', left_index = True, right_index = True)
        
        return data
        
class resample_data(data_loader):
    
    def under_sample(self, df):
        
        no , yes = df['y'].value_counts()
        no_users = df[df['y'] == 0]
        yes_users = df[df['y'] == 1]

        no_sample = no_users.sample(yes)
        data = pd.concat([no_sample,yes_users], axis = 0)
        
        return data
    
    def over_sample(self, df):
        
        X = df.drop('y', axis=1).reset_index(drop = True)
        y = df['y']
        
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        return X_res, y_res
    
class split:
    
    def train_test(df):
        
        X = df.drop('y', axis=1).reset_index(drop = True)
        y = df['y']
        
        sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        for train_index, test_index in sss.split(X, y):
            print("Train:", train_index, "Test:", test_index)
            Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        
        return Xtrain, Xtest, ytrain, ytest