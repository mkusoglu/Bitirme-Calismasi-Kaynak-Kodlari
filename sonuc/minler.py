# -*- coding: utf-8 -*-
"""
Created on Sat May 29 03:10:02 2021

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
#mae- 4_5000e_opt=adam_loss=mae_split=2_kb.csv
#mse 16x16_5000e_opt=sgd_loss=mse_split=2_kb.csv

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



x_train,x_test,y_train,y_test=train_test_split(train3,test3,test_size=0.2,shuffle=False)

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(4,activation='relu',input_shape=(6,),))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mae',metrics=['mae','mape','mse','accuracy'])
    return model


model=build_model()
csv_logger =  CSVLogger("mae_min.csv", append=True)
history=model.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger])

his=pd.read_csv('mae_min.csv')
tahm=model.predict(x_test)
tah=pd.DataFrame(tahm)

acc = history.history['mae']
tess = y_test.head(80)
tahs = tah.head(80)
tess=tess.reset_index(drop=True)


def build_model1():
    model=models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(6,),))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='sgd',loss='mse',metrics=['mae','mape','mse','accuracy'])
    return model


model1=build_model1()
csv_logger1 =  CSVLogger("mse_min.csv", append=True)
history1=model1.fit(x_train,y_train,
              epochs=5000,batch_size=1,verbose=1,callbacks=[csv_logger1])

his1=pd.read_csv('mse_min.csv')
tahm1=model1.predict(x_test)
tah1=pd.DataFrame(tahm1)

acc1 = history1.history['mae']

y_test=y_test.reset_index(drop=True)
mse_min_test = y_test.head(80)
mse_min_tah = tah1.head(80)
mse_tail = y_test.tail(80)
mse_tail = mse_tail.reset_index(drop=True)

epochs = range(1,80)
plt.plot(mse_min_test, label='Gercek Degerler')
plt.plot(mse_min_tah, color='red', label='Test Sonucları')
plt.xlabel('2020 Gunleri')
plt.ylabel('Doviz Kuru Degerleri')
plt.legend()
plt.show()

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




x_test = np.expand_dims(x_test, 1)
x_test_tail=x_test.last(80)
lst=model.predict(x_test)
lst_tah = lst[-80:]
plt.plot(mse_tail, label='Gercek Degerler')
plt.plot(lst_tah, color='red', label='Test Sonucları')
plt.xlabel('2021 Gunleri')
plt.ylabel('Doviz Kuru Degerleri')
plt.legend()
plt.show()

#model1
lst=model1.predict(x_test)
lst1_tah = lst1[:80]
plt.plot(mse_min_test, label='Gercek Degerler')
plt.plot(lst1_tah, color='red', label='Test Sonucları')
plt.xlabel('202 Gunleri')
plt.ylabel('Doviz Kuru Degerleri')
plt.legend()
plt.show()
#y_test, lst, c='crimson'
"""
fig, ax = plt.subplots()
ax.scatter(y_test, lst)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

plot.scatter(x_train, y_train, color = 'red')
plot.plot(x_train, linearRegressor.predict(xTrain), color = 'blue')
plot.title('Salary vs Experience (Training set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

acc = history.history['mae']
epochs = range(1,len(acc)+1)
plt.plot(epochs, y_test[0], 'blur', label='Training loss')
plt.plot(epochs, acc, 'red', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()"""
"""
epochs = range(1,80)
plt.plot(tess.values, label='Gercek Degerler')
plt.plot(tahs, color='red', label='Test Sonucları')
plt.xlabel('2020 Gunleri')
plt.ylabel('Doviz Kuru Degerleri')
plt.legend()
plt.show()

plt.plot(epochs, y_test, 'blur', label='Training loss')
plt.plot(epochs, tahm, 'red', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
"""