# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:48:57 2021

@author: MustafaKuşoğlu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import CSVLogger
from sklearn.metrics import accuracy_score



df=pd.read_csv('merge_with_tufe.csv')
df['Tarih']=pd.to_datetime(df['Tarih'])
df=df.sort_values(by='Tarih')


df=df.drop(['Unnamed: 0','Unnamed: 0.1','Tarih',],axis=1)
scaler = preprocessing.MinMaxScaler()
names=df.columns
d = scaler.fit_transform(df)
df = pd.DataFrame(d, columns=names)

train=df.iloc[:,:6]
test=df.iloc[:,6:]


x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.2,shuffle=True)


def build_model():
    model=models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model


num_epoch=1000
model=build_model()
csv_logger4 =  CSVLogger("model_history_log5.csv", append=True)
model.fit(x_train,y_train,
              epochs=num_epoch,batch_size=1,verbose=1,callbacks=[csv_logger4])

his=pd.read_csv('model_history_log5.csv')
tahm=model.predict(x_test)
tah=pd.DataFrame(tahm)
model.score(tah,y_test)
print(model.evaluate(tah, y_test))


my_series = y_test.squeeze()


def build_model1():
    model=models.Sequential()
    model.add(layers.Dense(32,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model


num_epoch1=5000
model1=build_model1()
csv_logger1 =  CSVLogger("model_history_log1.csv", append=True)
model1.fit(x_train,y_train,
              epochs=num_epoch1,batch_size=1,verbose=1,callbacks=[csv_logger1])

his1=pd.read_csv('model_history_log1.csv')

def build_model2():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model


num_epoch2=1000
model2=build_model2()
csv_logger2 =  CSVLogger("model_history_log2.csv", append=True)
model2.fit(x_train,y_train,
              epochs=num_epoch2,batch_size=1,verbose=1,callbacks=[csv_logger2])

his2=pd.read_csv('model_history_log2.csv')

def build_model3():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model


num_epoch3=5000
model3=build_model3()
csv_logger3 =  CSVLogger("model_history_log3.csv", append=True)
model3.fit(x_train,y_train,
              epochs=num_epoch3,batch_size=1,verbose=1,callbacks=[csv_logger3])

his3=pd.read_csv('model_history_log3.csv')