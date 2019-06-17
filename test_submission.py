'''
Created on Sep 26, 2017

@author: aelsalla
'''

import csv
submission_file = 'sample_submission.csv'# 'sub20170926_134843_NN_Keras.csv'
#submission_file = 'sub20170926_134843_NN_Keras.csv'
with open(submission_file, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    count = 0
    for row in reader:
        if count == 0:
            print(row) 
        count += 1
        if(len(row) == 0):
            print('Empty row')
    print(count)
    
    