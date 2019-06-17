'''
Created on Sep 20, 2017

@author: aelsalla
'''

import pandas as pd
import csv

data_dict = pd.read_csv("..\\..\\dat\\zillow_data_dictionary.csv", names=['Feature','Description']) 
print(data_dict.shape)
labels = pd.read_csv("..\\..\\dat\\train_2016_v2.csv", parse_dates=["transactiondate"])
print(labels.shape)

#data = pd.read_csv("..\\..\\dat\\properties_2016.csv", parse_dates=["transactiondate"])
data = pd.read_csv("..\\..\\dat\\properties_2016.csv")
print(data.shape)
