'''
Created on Sep 25, 2017

@author: aelsalla
'''

#Importing Libraries or Packages that are needed throughout the Program 
import numpy as np 
import pandas as pd 
import xgboost as  xgb 
import random 
import datetime as dt 
import gc
  
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer


 #Load the Datasets #

     # We need to load the datasets that will be needed to train our machine learning algorithms, handle our data and make predictions.
 #Note that these datasets are the ones that are already provided once you enter the competition by accepting terms and conditions
train = pd.read_csv('..\\..\\dat\\train_2016_v2.csv' , parse_dates=["transactiondate"]) 
properties = pd.read_csv('..\\..\\dat\\properties_2016.csv')   
test = pd.read_csv('..\\..\\dat\\sample_submission.csv') 
test= test.rename(columns={'ParcelId': 'parcelid'}) #To make it easier for merging datasets on same column_id later

# Analyse the Dimensions of our Datasets.

print("Training Size:" + str(train.shape))
print("Property Size:" + str(properties.shape))
print("Sample Size:" + str(test.shape))


# Type Converting the DataSet 

# The processing of some of the algorithms can be made quick if data representation is made in int/float32 instead of int/float64. Therefore, in order to make sure that all of our columns types are in float32, we are implementing the following lines of code #


for c, dtype in zip(properties.columns, properties.dtypes):
   if dtype == np.float64:        
       properties[c] = properties[c].astype(np.float32)
   if dtype == np.int64:
       properties[c] = properties[c].astype(np.int32)


for column in test.columns:
    if test[column].dtype == int:
        test[column] = test[column].astype(np.int32)
    if test[column].dtype == float:
        test[column] = test[column].astype(np.float32)

#living area proportions 
properties['living_area_prop'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']
#tax value ratio
properties['value_ratio'] = properties['taxvaluedollarcnt'] / properties['taxamount']
#tax value proportions
properties['value_prop'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']


###Merging the Datasets ###

# We are merging the properties dataset with training and testing dataset for model building and testing prediction #

df_train = train.merge(properties, how='left', on='parcelid') 
df_test = test.merge(properties, how='left', on='parcelid')


### Remove previos variables to keep some memory
del properties, train
gc.collect();


print('Memory usage reduction...')
df_train[['latitude', 'longitude']] /= 1e6
df_test[['latitude', 'longitude']] /= 1e6

df_train['censustractandblock'] /= 1e12
df_test['censustractandblock'] /= 1e12


# Label Encoding For Machine Learning &amp; Filling Missing Values 

# We are now label encoding our datasets. All of the machine learning algorithms employed in scikit learn assume that the data being fed to them is in numerical form. LabelEncoding ensures that all of our categorical variables are in numerical representation. Also note that we are filling the missing values in our dataset with a zero before label encoding them. This is to ensure that label encoder function does not experience any problems while carrying out its operation #

from sklearn.preprocessing import LabelEncoder  

lbl = LabelEncoder()
for c in df_train.columns:
    df_train[c]=df_train[c].fillna(0)
    if df_train[c].dtype == 'object':
        lbl.fit(list(df_train[c].values))
        df_train[c] = lbl.transform(list(df_train[c].values))

for c in df_test.columns:
    df_test[c]=df_test[c].fillna(0)
    if df_test[c].dtype == 'object':
        lbl.fit(list(df_test[c].values))
        df_test[c] = lbl.transform(list(df_test[c].values))     

### Rearranging the DataSets ###
'''
We will now drop the features that serve no useful purpose. 
We will also split our data and divide it into the representation to make it clear which features are to be treated as 
determinants in predicting the outcome for our target feature. Make sure to include the same features in the test set as 
were included in the training set 

'''

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 
                         'propertycountylandusecode', ], axis=1)

x_test = df_test.drop(['parcelid', 'propertyzoningdesc',
                       'propertycountylandusecode', '201610', '201611', 
                       '201612', '201710', '201711', '201712'], axis = 1) 

x_train = x_train.values
y_train = df_train['logerror'].values

## NN ##
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



#Submitting the Results 

#Once again load the file and start submitting the results in each column

sample_file = pd.read_csv('..\\dat\\sample_submission.csv') 
for c in sample_file.columns[sample_file.columns != 'ParcelId']:
    sample_file[c] = nn_pred

print('Preparing the csv file ...')
sample_file.to_csv('nn_predicted_results.csv', index=False, float_format='%.4f')
print("Finished writing the file")