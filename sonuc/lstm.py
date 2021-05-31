# -*- coding: utf-8 -*-
"""
Created on Thu May 13 23:11:04 2021

@author: MustafaKuşoğlu
"""
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
model = models.Sequential()
model.add(LSTM(8))
model.add(layers.Dense(1,activation='relu'))
model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae','mse'])
csv_logger =  CSVLogger("5000lstm8_opt=adam_loss=mae_random.csv", append=True)
history=model.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger])


model1 = models.Sequential()
model1.add(LSTM(8))
model1.add(layers.Dense(1,activation='relu'))
model1.compile(optimizer='adam',
              loss='mse',
              metrics=['mae','mse'])
csv_logger1 =  CSVLogger("5000lstm8_opt=adam_loss=mse_random.csv", append=True)

history1=model1.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger1])

model2 = models.Sequential()
model2.add(LSTM(8))
model2.add(layers.Dense(1,activation='relu'))
model2.compile(optimizer='adam',
              loss='huber_loss',
              metrics=['mae','mse'])
csv_logger2 =  CSVLogger("5000lstm8_opt=adam_loss=huber_random.csv", append=True)

history2=model2.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger2])

model3 = models.Sequential()
model3.add(LSTM(8))
model3.add(layers.Dense(1,activation='relu'))
model3.compile(optimizer='sgd',
              loss='mae',
              metrics=['mae','mse'])
csv_logger3 =  CSVLogger("5000lstm8_opt=sgd_loss=mae_random.csv", append=True)

history3=model3.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger3])

model4 = models.Sequential()
model4.add(LSTM(8))
model4.add(layers.Dense(1,activation='relu'))
model4.compile(optimizer='sgd',
              loss='mse',
              metrics=['mae','mse'])
csv_logger4 =  CSVLogger("5000lstm8_opt=sgd_loss=mse_random.csv", append=True)

history4=model4.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger4])

model5 = models.Sequential()
model5.add(LSTM(8))
model5.add(layers.Dense(1,activation='relu'))
model5.compile(optimizer='sgd',
              loss='huber_loss',
              metrics=['mae','mse'])
csv_logger5 =  CSVLogger("5000lstm8_opt=sgd_loss=huber_random.csv", append=True)

history5=model5.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger5])

model11 = models.Sequential()
model11.add(LSTM(4))
model11.add(layers.Dense(1,activation='relu'))
model11.compile(optimizer='adam',
              loss='mae',
              metrics=['mae','mse'])
csv_logger11 =  CSVLogger("5000lstm4_opt=adam_loss=mae_random.csv", append=True)

history11=model11.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger11])

model6 = models.Sequential()
model6.add(LSTM(4))
model6.add(layers.Dense(1,activation='relu'))
model6.compile(optimizer='adam',
              loss='mse',
              metrics=['mae','mse'])
csv_logger6 =  CSVLogger("5000lstm4_opt=adam_loss=mse_random.csv", append=True)

history6=model6.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger6])

model7 = models.Sequential()
model7.add(LSTM(4))
model7.add(layers.Dense(1,activation='relu'))
model7.compile(optimizer='adam',
              loss='huber_loss',
              metrics=['mae','mse'])
csv_logger7 =  CSVLogger("5000lstm4_opt=adam_loss=huber_random.csv", append=True)

history7=model7.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger7])

model8 = models.Sequential()
model8.add(LSTM(4))
model8.add(layers.Dense(1,activation='relu'))
model8.compile(optimizer='sgd',
              loss='mae',
              metrics=['mae','mse'])
csv_logger8 =  CSVLogger("5000lstm4_opt=sgd_loss=mae_random.csv", append=True)

history8=model8.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger8])

model9 = models.Sequential()
model9.add(LSTM(4))
model9.add(layers.Dense(1,activation='relu'))
model9.compile(optimizer='sgd',
              loss='mse',
              metrics=['mae','mse'])
csv_logger9 =  CSVLogger("5000lstm4_opt=sgd_loss=mse_random.csv", append=True)

history9=model9.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger9])

model10 = models.Sequential()
model10.add(LSTM(4))
model10.add(layers.Dense(1,activation='relu'))
model10.compile(optimizer='sgd',
              loss='huber_loss',
              metrics=['mae','mse'])
csv_logger10 =  CSVLogger("5000lstm4_opt=sgd_loss=huber_random.csv", append=True)

history10=model10.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger10])

model12 = models.Sequential()
model12.add(LSTM(8))
model12.add(layers.Dense(1,activation='relu'))
model12.compile(optimizer='rmsprop',
              loss='mae',
              metrics=['mae','mse'])
csv_logger12 =  CSVLogger("5000lstm8_opt=rms_loss=mae_random.csv", append=True)

history12=model12.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger12])

model13 = models.Sequential()
model13.add(LSTM(8))
model13.add(layers.Dense(1,activation='relu'))
model13.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae','mse'])
csv_logger13 =  CSVLogger("5000lstm8_opt=rms_loss=mse_random.csv", append=True)

history13=model13.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger13])

model14 = models.Sequential()
model14.add(LSTM(8))
model14.add(layers.Dense(1,activation='relu'))
model14.compile(optimizer='rmsprop',
              loss='huber_loss',
              metrics=['mae','mse'])
csv_logger14 =  CSVLogger("5000lstm8_opt=rmsprop_loss=huber_random.csv", append=True)

history14=model14.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger14])

model15 = models.Sequential()
model15.add(LSTM(4))
model15.add(layers.Dense(1,activation='relu'))
model15.compile(optimizer='rmsprop',
              loss='mae',
              metrics=['mae','mse'])
csv_logger15 =  CSVLogger("5000lstm4_opt=rmsprop_loss=mae_random.csv", append=True)

history15=model15.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger15])

model16 = models.Sequential()
model16.add(LSTM(4))
model16.add(layers.Dense(1,activation='relu'))
model16.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae','mse'])
csv_logger16 =  CSVLogger("5000lstm4_opt=rmsprop_loss=mse_random.csv", append=True)

history16=model16.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger16])

model17 = models.Sequential()
model17.add(LSTM(4))
model17.add(layers.Dense(1,activation='relu'))
model17.compile(optimizer='rmsprop',
              loss='huber_loss',
              metrics=['mae','mse'])
csv_logger17 =  CSVLogger("5000lstm4_opt=rmsprop_loss=huber_random.csv", append=True)

history17=model17.fit(x_train,y_train,
                  epochs=5000,
                  batch_size=1,
                  callbacks=[csv_logger17])
