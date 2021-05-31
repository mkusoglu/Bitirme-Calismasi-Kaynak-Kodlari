# -*- coding: utf-8 -*-
"""
Created on Mon May 31 05:05:12 2021

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

df=pd.read_csv('4_5000e_opt=adam_loss=mae_split=2_kb.csv')
df2=pd.read_csv('16x16_5000e_opt=sgd_loss=mse_split=2_kb.csv')
df3=pd.read_csv('5000lstm8_opt=adam_loss=mae_random.csv')
df4=pd.read_csv('5000lstm8_opt=adam_loss=huber_random.csv')

