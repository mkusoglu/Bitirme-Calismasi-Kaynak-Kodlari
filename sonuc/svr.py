# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:40:15 2021

@author: MustafaKuşoğlu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.svm import SVC

df=pd.read_csv('merge_with_tufe.csv')

train=df.iloc[:,:9]
test=df.iloc[:,9:]

train=train.drop(['Unnamed: 0','Unnamed: 0.1','Tarih',],axis=1)
train=train.drop(['ortalama'],axis=1)
x=train
y=test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,shuffle=False)
y_train.info()
y_train=y_train.astype('int64')
svclassifier = SVC(kernel='rbf')
svclassifier.fit(x_train, y_train)

y_pred = svclassifier.predict(x_test)
