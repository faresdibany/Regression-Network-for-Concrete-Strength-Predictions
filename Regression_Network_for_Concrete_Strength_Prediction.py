# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 23:41:43 2020

@author: Dell
"""
import pandas as pd 
import numpy as np 
import keras 
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv') 
concrete_data.head() 
concrete_data.describe() 
concrete_data.isnull().sum() 
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] 
# all columns except Strength 
target = concrete_data['Strength'] # Strength column 
predictors.head() 
target.head() 
predictors_norm = (predictors - predictors.mean()) / predictors.std() 
predictors_norm.head() 
n_cols = predictors_norm.shape[1] # number of predictors
def regression_model(): # create model 
    model = Sequential() 
    model.add(Dense(10, activation='relu', input_shape=(n_cols,))) 
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error') 
    return model
MSE=np.zeros(50)
for i in range(50):
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(predictors, target, test_size=0.3, random_state=3) 
    print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape)) 
    print('Shape of X testing set {}'.format(X_testset.shape),'&',' Size of Y testing set {}'.format(y_testset.shape))
    model = regression_model() 
    model.fit(predictors_norm, target, validation_split=0.3, epochs=50, verbose=2)
    # evaluate model predictions
    #validation_data=(predictors_test,target_test),epochs=50,verbose=0)
    from sklearn.metrics import mean_squared_error  
    y_pred = model.predict(X_testset)
    X_testset.shape
    #y_pred.shape
    #y_testset.shape
    MSE[i] = mean_squared_error(y_testset, y_pred)
    print("Test set MSE for {} cycle:{}".format(i+1,MSE[i]))
