# -*- coding: utf-8 -*-
"""
Created on Sun May  9 01:19:11 2021

@author: MustafaKuşoğlu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers,losses
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
train=df.iloc[:,:6]
test=df.iloc[:,6:]
names=train.columns
d = scaler.fit_transform(train)
train = pd.DataFrame(d, columns=names)
test=test.reset_index(drop=True)



x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.2,shuffle=True)

def build_model2():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model


num_epoch2=5000
model2=build_model2()
csv_logger2 =  CSVLogger("4x4_5000e_opt=adam_loss=mae=2_random.csv", append=True)
model2.fit(x_train,y_train,
              epochs=num_epoch2,batch_size=1,verbose=1,callbacks=[csv_logger2])

his2=pd.read_csv('4x4_5000e_opt=adam_loss=mae=2_random.csv')
tahm2=model2.predict(x_test)
tah2=pd.DataFrame(tahm2)

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='sgd',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model


num_epoch=5000
model=build_model()
csv_logger =  CSVLogger("4x4_5000e_opt=sgd_loss=mse_split=2_random.csv", append=True)
model.fit(x_train,y_train,
              epochs=num_epoch,batch_size=1,verbose=1,callbacks=[csv_logger])

his=pd.read_csv('4x4_5000e_opt=sgd_loss=mse_split=2_random.csv')
tahm=model.predict(x_test)
tah=pd.DataFrame(tahm)

#1
def build_model1():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='huber_loss',metrics=['mae','mape','mse','accuracy'])
    return model


num_epoch1=5000
model1=build_model1()
csv_logger1 =  CSVLogger("4x4_5000e_opt=sgd_loss=huber=2_random.csv", append=True)
model1.fit(x_train,y_train,
              epochs=num_epoch1,batch_size=1,verbose=1,callbacks=[csv_logger1])

his1=pd.read_csv('4x4_5000e_opt=sgd_loss=huber=2_random.csv')
tahm1=model.predict(x_test)
tah1=pd.DataFrame(tahm1)

#2


#3
df2=pd.read_csv('merge_with_tufe.csv')
df2=df2.drop(['Unnamed: 0','Unnamed: 0.1','Tarih',],axis=1)
scaler2 = preprocessing.MinMaxScaler()
train2=df2.iloc[:,:6]
test2=df2.iloc[:,6:]
names2=train2.columns
d2 = scaler2.fit_transform(train2)
train2 = pd.DataFrame(d2, columns=names2)
test2=test2.reset_index(drop=True)



x_train2,x_test2,y_train2,y_test2=train_test_split(train2,test2,test_size=0.2,shuffle=False)

def build_model3():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='sgd',loss='huber_loss',metrics=['mae','mape','mse','accuracy'])
    return model


num_epoch3=5000
model3=build_model3()
csv_logger3 =  CSVLogger("4x4_5000e_opt=sgd_loss=huber_split=2_bk.csv", append=True)
model3.fit(x_train2,y_train2,
              epochs=num_epoch3,batch_size=1,verbose=1,callbacks=[csv_logger3])

his3=pd.read_csv('4x4_5000e_opt=sgd_loss=huber_split=2_bk.csv')
tahm3=model3.predict(x_test2)
tah3=pd.DataFrame(tahm3)
