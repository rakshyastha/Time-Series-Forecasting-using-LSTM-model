# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:36:49 2023

@author: Rakshya Shrestha
"""
#importing libraries
#import math
import os
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
#from pandas.api.types import is_datetime64_any_dtype

#Setting working directory
path = 'C:/Users/14098/OneDrive - Lamar University/Desktop/Machine Learning/Timeseries'
os.chdir(path)

#converting csv to dataframe
df=pd.read_csv("imputeddaysj17.csv")
print('Number of rows and columns:', df.shape)
df.head(5)

#converting date to datetime format
df.index = pd.to_datetime(df['date'])
df['Date'] = df.index 

#splitting the data into testing and training sets
Train = df.iloc[:26416, 2:3]
training_set = df.iloc[:26416, 2:3].values
Test = df.iloc[26416:, 2:3]
test_set = df.iloc[26416:, 2:3].values

# Feature Scaling to normalize the data
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 10 time-steps and 1 output
Y = 10
X_train = []
y_train = []
for i in range(Y, 26416):
    X_train.append(training_set_scaled[i-Y:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Building LSTM model
model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
callbacks = [EarlyStopping(monitor='loss', patience=10 , restore_best_weights=True)]
model.fit(X_train, y_train, epochs = 10, batch_size = 64,callbacks=callbacks)

model.summary()

#preparing the test data
X = 10
dataset_train = df.iloc[:26416, 2:3]
dataset_test = df.iloc[26416:, 2:3]
dataset_test.shape
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - X:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) #normalizing the input
X_test = []
for i in range(X, 6614):
    X_test.append(inputs[i-X:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)


#make prediction using test data
predicted = model.predict(X_test)
predicted.shape
predicted = sc.inverse_transform(predicted) #invert the normalization

#converting prediction to dataframe
Test['Prediction']=predicted
Test.head(5)


#visualize the output
plt.figure(figsize=(16,6))
plt.title('Water level Elevation prediction' , fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.ylabel('WaterLevelElevation' ,fontsize=15)
plt.plot(Train['WaterLevelElevation'],linewidth=3)
plt.plot(Test['WaterLevelElevation'],linewidth=3)
plt.plot(Test["Prediction"],linewidth=3)
plt.legend(['Train','Test','Prediction'])


#visualize the predicted-test part
plt.figure(figsize=(16,6))
plt.title('Water level Elevation prediction' , fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.ylabel('WaterLevelElevation' ,fontsize=15)
plt.plot(Test['WaterLevelElevation'],linewidth=3)
plt.plot(Test["Prediction"],linewidth=3)
plt.legend(['Test','Prediction'])











