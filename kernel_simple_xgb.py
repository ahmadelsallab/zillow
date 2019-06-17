'''
Created on Sep 22, 2017

@author: aelsalla
'''

# Parameters


import numpy as np
import pandas as pd
import random
import datetime as dt

import gc

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

import xgboost as  xgb







print('\nLoading train, prop and sample data...')
train = pd.read_csv("..\\..\\dat\\train_2016_v2.csv", parse_dates=["transactiondate"])
prop = pd.read_csv('..\\..\\dat\\properties_2016.csv')
sample = pd.read_csv('..\\..\\dat\\sample_submission.csv')
 
print('Fitting Label Encoder on properties...')
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))
        
print('Creating training set...')
df_train = train.merge(prop, how='left', on='parcelid')

df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train["transactiondate_year"] = df_train["transactiondate"].dt.year
df_train["transactiondate_month"] = df_train["transactiondate"].dt.month
df_train['transactiondate_quarter'] = df_train['transactiondate'].dt.quarter
df_train["transactiondate"] = df_train["transactiondate"].dt.day

print('Filling NA/NaN values...' )
df_train.fillna(-1.0)

print('Creating x_train and y_train from df_train...' )
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train["logerror"]

y_mean = np.mean(y_train)
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

print('Creating df_test...')
sample['parcelid'] = sample['ParcelId']

print("Merging Sample with property data...")
df_test = sample.merge(prop, on='parcelid', how='left')

df_test["transactiondate"] = pd.to_datetime('2016-11-15')  # placeholder value for preliminary version
df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
df_test["transactiondate"] = df_test["transactiondate"].dt.day     
x_test = df_test[train_columns]

print('Shape of x_test:', x_test.shape)
print("Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
  

#Implement the Xgboost#

# We can now select the parameters for Xgboost and monitor the progress of results on our validation set. The explanation of the xgboost parameters and what they do can be found on the following link http://xgboost.readthedocs.io/en/latest/parameter.html #

### Cross Validation ###

# We are dividing our datasets into the training and validation sets so that we could monitor and the test the progress of our machine learning algorithm. This would let us know when our model might be over or under fitting on the dataset that we have employed. #

from sklearn.model_selection import train_test_split

X = x_train
y = y_train 

Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.2, random_state=42)


dtrain = xgb.DMatrix(Xtrain, label=ytrain)
dvalid = xgb.DMatrix(Xvalid, label=yvalid)
#dtest = xgb.DMatrix(x_test.values)
dtest = xgb.DMatrix(x_test)

# Try different parameters! 
xgb_params = {'min_child_weight': 5, 'eta': 0.035, 'colsample_bytree': 0.5, 'max_depth': 4,
            'subsample': 0.85, 'lambda': 0.8, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,
            'eval_metric': 'mae', 'objective': 'reg:linear' }           

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

model_xgb = xgb.train(xgb_params, dtrain, 1000, watchlist, early_stopping_rounds=100,
                  maximize=False, verbose_eval=10)

#Predicting the results#

# Let us now predict the target variable for our test dataset. All we have to do now is just fit the already trained model on the test set that we had made merging the sample file with properties dataset #

xgb_pred = model_xgb.predict(dtest)


sample = pd.read_csv('..\\..\\dat\\sample_submission.csv')
test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']

test_columns = ['201610','201611','201612','201710','201711','201712']

#for i in range(len(test_dates)):
for i in range(len(test_columns)):
#    test['transactiondate'] = test_dates[i]
    pred = xgb_pred
    sample[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

print( "\nSample submission predictions:" )
print( sample.head() )



##### WRITE THE RESULTS

from datetime import datetime

print( "\nWriting results to disk ..." )
sample.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")