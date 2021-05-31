# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:54:03 2021

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
df=df.iloc[::-1]

df=df.drop(['Unnamed: 0','Unnamed: 0.1','Tarih',],axis=1)
scaler = preprocessing.MinMaxScaler()
train=df.iloc[:,:6]
test=df.iloc[:,6:]
names=train.columns
d = scaler.fit_transform(train)
train = pd.DataFrame(d, columns=names)
test=test.reset_index(drop=True)



x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.2,shuffle=False)


def build_model7():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mae',metrics=['mae','mape','mse','accuracy'])
    return model


model7=build_model7()
csv_logger7 =  CSVLogger("4_5000e_opt=rms_loss=mae_split=2_kb.csv", append=True)
model7.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger7])

his7=pd.read_csv('4_5000e_opt=rms_loss=mae_split=2_kb.csv')
tahm7=model7.predict(x_test)
tah7=pd.DataFrame(tahm7)


def build_model():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='sgd',loss='huber_loss',metrics=['mae','mape','mse','accuracy'])
    return model


model=build_model()
csv_logger =  CSVLogger("4_5000e_opt=sgd_loss=huber_split=2_kb.csv", append=True)
model.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger])

his=pd.read_csv('4_5000e_opt=sgd_loss=huber_split=2_kb.csv')
tahm=model.predict(x_test)
tah=pd.DataFrame(tahm)

def build_model1():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='sgd',loss='mae',metrics=['mae','mape','mse','accuracy'])
    return model



model1=build_model1()
csv_logger1 =  CSVLogger("4_5000e_opt=sgd_loss=mae_split=2_kb.csv", append=True)
model1.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger1])

his1=pd.read_csv('4_5000e_opt=sgd_loss=mae_split=2_kb.csv')
tahm1=model1.predict(x_test)
tah1=pd.DataFrame(tahm1)


def build_model8():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='sgd',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model



model8=build_model8()
csv_logger8 =  CSVLogger("4_5000e_opt=sgd_loss=mse_split=2_kb.csv", append=True)
model8.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger8])

his8=pd.read_csv('4_5000e_opt=sgd_loss=mse_split=2_kb.csv')
tahm8=model8.predict(x_test)
tah8=pd.DataFrame(tahm8)


def build_model2():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='huber_loss',metrics=['mae','mape','mse','accuracy'])
    return model



model2=build_model2()
csv_logger2 =  CSVLogger("4_5000e_opt=adam_loss=huber_split=2_kb.csv", append=True)
model2.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger2])

his2=pd.read_csv('4_5000e_opt=adam_loss=huber_split=2_kb.csv')
tahm2=model2.predict(x_test)
tah2=pd.DataFrame(tahm2)

def build_model3():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model



model3=build_model3()
csv_logger3 =  CSVLogger("4_5000e_opt=adam_loss=mse_split=2_kb.csv", append=True)
model3.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger3])

his3=pd.read_csv('4_5000e_opt=adam_loss=mse_split=2_kb.csv')
tahm3=model3.predict(x_test)
tah3=pd.DataFrame(tahm3)

def build_model4():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mae',metrics=['mae','mape','mse','accuracy'])
    return model


model4=build_model4()
csv_logger4 =  CSVLogger("4_5000e_opt=adam_loss=mae_split=2_kb.csv", append=True)
model4.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger4])

his4=pd.read_csv('4_5000e_opt=adam_loss=mae_split=2_kb.csv')
tahm4=model4.predict(x_test)
tah4=pd.DataFrame(tahm4)

def build_model5():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='huber_loss',metrics=['mae','mape','mse','accuracy'])
    return model



model5=build_model5()
csv_logger5 =  CSVLogger("4_5000e_opt=rms_loss=huber_split=2_kb.csv", append=True)
model5.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger5])

his5=pd.read_csv('4_5000e_opt=rms_loss=huber_split=2_kb.csv')
tahm5=model5.predict(x_test)
tah5=pd.DataFrame(tahm5)

def build_model6():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model



model6=build_model6()
csv_logger6 =  CSVLogger("4_5000e_opt=rms_loss=mse_split=2_kb.csv", append=True)
model6.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger6])

his6=pd.read_csv('4_5000e_opt=rms_loss=mse_split=2_kb.csv')
tahm6=model6.predict(x_test)
tah6=pd.DataFrame(tahm6)

