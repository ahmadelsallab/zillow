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
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split







print('\nLoading train, prop and sample data...')
train = pd.read_csv("..\\..\\dat\\train_2016_v2.csv", parse_dates=["transactiondate"])
prop_in = pd.read_csv('..\\..\\dat\\properties_2016.csv')
sample = pd.read_csv('..\\..\\dat\\sample_submission.csv')

print('Forward selection')
prop_in_ALL = prop_in.drop('parcelid', axis=1)
num_features = len(prop_in_ALL.columns)
print(num_features)
best_features = []
for n in range(num_features):
    best_feature = prop_in_ALL.columns[0]
    min_val_acc = 1000
    print('Select from: ', len(prop_in_ALL.columns), ' features')
    for c in prop_in_ALL.columns:
        print('Testing feature:', c)
    
        prop= pd.DataFrame()
        prop['parcelid'] = prop_in['parcelid']
        prop[c] = prop_in_ALL[c]
        

        if(1):
            print('Fitting Label Encoder on properties...')
            for c in prop.columns:
                prop[c]=prop[c].fillna(-1)
                if prop[c].dtype == 'object':
                    lbl = LabelEncoder()
                    lbl.fit(list(prop[c].values))
                    prop[c] = lbl.transform(list(prop[c].values))
            
             
                    
            print('Creating training set...')
            df_train = train.merge(prop, how='left', on='parcelid')
            '''
            df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
            df_train["transactiondate_year"] = df_train["transactiondate"].dt.year
            df_train["transactiondate_month"] = df_train["transactiondate"].dt.month
            df_train['transactiondate_quarter'] = df_train['transactiondate'].dt.quarter
            df_train["transactiondate"] = df_train["transactiondate"].dt.day
            '''
            print('Filling NA/NaN values...' )
            df_train.fillna(-1.0)
            
            print('Creating x_train and y_train from df_train...' )
            #x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplacecnt', 'fireplaceflag'], axis=1)
            x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
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
            '''
            df_test["transactiondate"] = pd.to_datetime('2016-11-15')  # placeholder value for preliminary version
            df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
            df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
            df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
            df_test["transactiondate"] = df_test["transactiondate"].dt.day
            '''     
            x_test = df_test[train_columns]
            
            print('Shape of x_test:', x_test.shape)
            print("Preparing x_test...")
            for c in x_test.dtypes[x_test.dtypes == object].index.values:
                x_test[c] = (x_test[c] == True)
              
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, x_train, test_size=0.2, random_state=42)
            
            #Implement the NN#
            ## Preprocessing
            print("\nPreprocessing neural network data...")
            imputer= Imputer()
            imputer.fit(x_train.iloc[:, :])
            x_train = imputer.transform(x_train.iloc[:, :])
            imputer.fit(x_test.iloc[:, :])
            x_test = imputer.transform(x_test.iloc[:, :])
            
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)
            
            len_x=int(x_train.shape[1])
            print("len_x is:",len_x)
            
            
            # Neural Network
            print("\nSetting up neural network model...")
            nn = Sequential()
            nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
            nn.add(PReLU())
            nn.add(Dropout(.4))
            nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
            nn.add(PReLU())
            nn.add(BatchNormalization())
            nn.add(Dropout(.63))
            nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
            nn.add(PReLU())
            nn.add(BatchNormalization())
            nn.add(Dropout(.45))
            nn.add(Dense(units = 28, kernel_initializer = 'normal'))
            nn.add(PReLU())
            nn.add(BatchNormalization())
            nn.add(Dropout(.5))
            nn.add(Dense(1, kernel_initializer='normal'))
            nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))
            
            print("\nFitting neural network model...")
            nn.fit(np.array(x_train), np.array(y_train), batch_size = 32, epochs = 70, verbose=2)
            
            print("\nPredicting with neural network model...")
            #print("x_test.shape:",x_test.shape)
            y_pred_ann = nn.predict(x_test)
            
            print( "\nPreparing results for write..." )
            nn_pred = y_pred_ann.flatten()
            print( "Type of nn_pred is ", type(nn_pred) )
            print( "Shape of nn_pred is ", nn_pred.shape )
            
            print( "\nNeural Network predictions:" )
            print( pd.DataFrame(nn_pred).head() )
            
            val_acc = nn.evaluate(x_valid, y_valid)
            # Test feature
            if(val_acc < min_val_acc):
                best_feature = c
                min_val_acc = val_acc
                
            # Cleanup
            del train
            del prop
            del sample
            del x_train
            del x_test
            del df_train
            del df_test
            del y_pred_ann
            gc.collect()

            '''
            
            ########################
            ########################
            ##  Combine and Save  ##
            ########################
            ########################
            
            
            ##### COMBINE PREDICTIONS
            
        
            
            
            sample = pd.read_csv('..\\..\\dat\\sample_submission.csv')
            test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
            
            test_columns = ['201610','201611','201612','201710','201711','201712']
            
            #for i in range(len(test_dates)):
            for i in range(len(test_columns)):
            #    test['transactiondate'] = test_dates[i]
                pred = nn_pred
                sample[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
                print('predict...', i)
            
            print( "\nSample submission predictions:" )
            print( sample.head() )
            
            
            
            ##### WRITE THE RESULTS
            
            from datetime import datetime
            
            print( "\nWriting results to disk ..." )
            sample.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
            
            print( "\nFinished ...")
            '''       
    
    print('Best chosen feature: ', best_feature)
    
    best_features.append(best_feature)
    print('Best features so far: ', best_features)
    
    prop_in_ALL = prop_in_ALL.drop(best_feature, axis=1)
        
best_features = pd.DataFrame(best_features)
best_features.to_pickle('..\\..\\dat\\fwd_selection.pkl')