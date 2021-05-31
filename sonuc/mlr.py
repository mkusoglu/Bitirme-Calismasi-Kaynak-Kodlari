import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error


df=pd.read_csv('merge_with_tufe.csv')
df2=df
df=df2
s=df.dolar.std()

df['dolar']=df.dolar.fillna(df.dolar.mean())
df['petrol']=df.petrol.fillna(df.petrol.mean())
df['tahmin']=df.tahmin.fillna(df.tahmin.mean())
df['ortalama']=df.ortalama.fillna(df.ortalama.mean())

df['Tarih']=pd.to_datetime(df['Tarih'])

train=df.iloc[:,:9]
test=df.iloc[:,9:]

train=train.drop(['Unnamed: 0','Unnamed: 0.1','Tarih',],axis=1)
train=train.drop(['ortalama'],axis=1)
x=train
y=test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,shuffle=False)
regresor=LinearRegression()

regresor.fit(x_train,y_train)

y_pred=regresor.predict(x_test)



mae1=mean_absolute_error(y_pred,y_test)
mse1=mean_squared_error(y_pred,y_test)

epochs = range(1,len(acc)+1)
y_test=y_test.reset_index(drop=True)
mse_min_test = y_test.head(80)
y_pred1 = y_pred[:80]
plt.plot(mse_min_test, label='Gercek Degerler')
plt.plot(y_pred1, color='red', label='Test Sonucları')
plt.xlabel('2013 Gunleri')
plt.ylabel('Doviz Kuru Degerleri')
plt.legend()
plt.show()

rmse1=mean_squared_error(y_pred,y_test,squared=False)
#
df2=df
df2=df2.sort_values(by='Tarih')

train1=df2.iloc[:,:9]
test1=df2.iloc[:,9:]
train1=train1.drop(['Unnamed: 0','Unnamed: 0.1','Tarih'],axis=1)
x_train1,x_test1,y_train1,y_test1=train_test_split(train1,test1,test_size=0.1,shuffle=False)

regresor1=LinearRegression()

regresor1.fit(x_train1,y_train1)

y_pred1=regresor1.predict(x_test1)


mae2=mean_absolute_error(y_pred1,y_test1)
mse2=mean_squared_error(y_pred1,y_test1)
rmse2=mean_squared_error(y_pred1,y_test1,squared=False)

#3
df3=df
df3=df3.sort_values(by='Tarih')

train2=df3.iloc[:,:9]
test2=df3.iloc[:,9:]
train2=train2.drop(['Unnamed: 0','Unnamed: 0.1','Tarih','ortalama'],axis=1)
x_train2,x_test2,y_train2,y_test2=train_test_split(train2,test2,test_size=0.1,shuffle=False)

regresor2=LinearRegression()

regresor2.fit(x_train2,y_train2)

y_pred2=regresor2.predict(x_test2)


mae3=mean_absolute_error(y_pred2,y_test2)
mse3=mean_squared_error(y_pred2,y_test2)
rmse3=mean_squared_error(y_pred2,y_test2,squared=False)

#
df['dolar']=df.dolar.fillna(df.dolar.mean())
df['petrol']=df.petrol.fillna(df.petrol.mean())
df['tahmin1']=df.tahmin1.fillna(df.tahmin1.mean())
df['ortalama']=df.ortalama.fillna(df.ortalama.mean())

df['Tarih']=pd.to_datetime(df['Tarih'])

train=df.iloc[:,:9]
test=df.iloc[:,9:]

train=train.drop(['Unnamed: 0','Unnamed: 0.1','Tarih',],axis=1)
train=train.drop(['ortalama'],axis=1)
x=train
y=test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,shuffle=True)
regresor=LinearRegression()

regresor.fit(x_train,y_train)

y_pred=regresor.predict(x_test)

mae1=mean_absolute_error(y_pred,y_test)
mse1=mean_squared_error(y_pred,y_test)
rmse1=mean_squared_error(y_pred,y_test,squared=False)
