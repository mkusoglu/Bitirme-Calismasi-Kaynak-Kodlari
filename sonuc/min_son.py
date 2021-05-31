# -*- coding: utf-8 -*-
"""
Created on Mon May 31 05:18:32 2021

@author: MustafaKuşoğlu
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import CSVLogger

df3=pd.read_csv('merge_with_tufe.csv')
df3=df3.iloc[::-1]

df3=df3.drop(['Unnamed: 0','Unnamed: 0.1','Tarih',],axis=1)
scaler3 = preprocessing.MinMaxScaler()
train3=df3.iloc[:,:6]
test3=df3.iloc[:,6:]
names3=train3.columns
d3 = scaler3.fit_transform(train3)
train3 = pd.DataFrame(d3, columns=names3)
test3=test3.reset_index(drop=True)



x_train3,x_test3,y_train3,y_test3=train_test_split(train3,test3,test_size=0.2,shuffle=False)

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mae',metrics=['mae','mape','mse','accuracy'])
    return model


model=build_model()
csv_logger =  CSVLogger("mae_minson.csv", append=True)
history=model.fit(x_train3,y_train3,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger])

his=pd.read_csv('mae_minson.csv')
tahm=model.predict(x_test3)
tah=pd.DataFrame(tahm)

acc = history.history['mae']
tess = y_test3.head(80)
tahs = tah.head(80)
tess=tess.reset_index(drop=True)

#---------------------------------------

def build_model2():
    model=models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='sgd',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model


model2=build_model2()
csv_logger2 =  CSVLogger("mse_minson.csv", append=True)
history2=model2.fit(x_train3,y_train3,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger2])

his2=pd.read_csv('mse_minson.csv')
tahm2=model2.predict(x_test3)
tah2=pd.DataFrame(tahm2)

acc2 = history2.history['mae']
tess2 = y_test3.head(80)
tahs2 = tah2.head(80)
tess2=tess2.reset_index(drop=True)

#lstmler----------
df=pd.read_csv('merge_with_tufe.csv')
df=df.drop(['Unnamed: 0','Unnamed: 0.1','Tarih',],axis=1)
scaler = preprocessing.MinMaxScaler()
train=df.iloc[:,:6]
test=df.iloc[:,6:]
names=train.columns
d = scaler.fit_transform(train)
train = pd.DataFrame(d, columns=names)
test=test.reset_index(drop=True)



x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.2,shuffle=True)
x_train = np.expand_dims(x_train, 1)

model3 = models.Sequential()
model3.add(LSTM(8))
model3.add(layers.Dense(1,activation='relu'))
model3.compile(optimizer='adam',
              loss='mae',
              metrics=['mae','mse'])
csv_logger3 =  CSVLogger("lstm_mse_son.csv", append=True)

history3=model3.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger3])

#-------------

model4 = models.Sequential()
model4.add(LSTM(8))
model4.add(layers.Dense(1,activation='relu'))
model4.compile(optimizer='adam',
              loss='huber_loss',
              metrics=['mae','mse'])
csv_logger4 =  CSVLogger("lstm_mae_son.csv", append=True)

history4=model4.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger4])
