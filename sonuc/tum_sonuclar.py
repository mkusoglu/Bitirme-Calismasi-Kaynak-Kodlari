# -*- coding: utf-8 -*-
"""
Created on Sun May 16 22:34:18 2021

@author: MustafaKuşoğlu
"""
import pandas as pd
import numpy as np

df=pd.read_csv('16x16_1000e_opt=adam_loss=huber_split=2_bk.csv')
df1=pd.read_csv('16x16_1000e_opt=adam_loss=huber_split=2_bk.csv')
df2=pd.read_csv('16x16_1000e_opt=adam_loss=huber_split=2_kb.csv')
df3=pd.read_csv('16x16_1000e_opt=adam_loss=huber_split=2_random.csv')
df4=pd.read_csv('16x16_1000e_opt=adam_loss=huber=2_kb_ortsuz.csv')
df5=pd.read_csv('16x16_1000e_opt=adam_loss=mae_split=2_bk.csv')
df6=pd.read_csv('16x16_1000e_opt=adam_loss=mae_split=2_kb.csv')
df7=pd.read_csv('16x16_1000e_opt=adam_loss=mae=2_random.csv')
df8=pd.read_csv('16x16_1000e_opt=adam_loss=mse_split=2_bk.csv')
df9=pd.read_csv('16x16_1000e_opt=adam_loss=mse_split=2_kb.csv')
#df10=pd.read_csv('16x16_1000e_opt=adam_loss=mse=2_random.csv')
df11=pd.read_csv('16x16_1000e_opt=rmsprop_loss=huber_split=2_bk.csv')
df12=pd.read_csv('16x16_1000e_opt=rmsprop_loss=huber_split=2_kb.csv')
df13=pd.read_csv('16x16_1000e_opt=rmsprop_loss=huber_split=2_random.csv')
df14=pd.read_csv('16x16_1000e_opt=rmsprop_loss=mae_split=2_bk.csv')
df15=pd.read_csv('16x16_1000e_opt=rmsprop_loss=mae_split=2_kb.csv')
df16=pd.read_csv('16x16_1000e_opt=rmsprop_loss=mae_split=2_random.csv')
df17=pd.read_csv('16x16_1000e_opt=rmsprop_loss=mse_split=2_bk.csv')
df18=pd.read_csv('16x16_1000e_opt=rmsprop_loss=mse_split=2_kb.csv')
df19=pd.read_csv('16x16_1000e_opt=rmsprop_loss=mse_split=2_random.csv')
df20=pd.read_csv('16x16_1000e_opt=sgd_loss=huber_split=2_bk.csv')
df21=pd.read_csv('16x16_1000e_opt=sgd_loss=huber_split=2_kb.csv')
df22=pd.read_csv('16x16_1000e_opt=sgd_loss=huber_split=2_random.csv')
df23=pd.read_csv('16x16_1000e_opt=sgd_loss=mse_split=2_bk.csv')
df24=pd.read_csv('16x16_1000e_opt=sgd_loss=mse_split=2_kb.csv')
df25=pd.read_csv('16x16_1000e_opt=sgd_loss=mae=2_random.csv')
df26=pd.read_csv('16x16_1000e_opt=sgd_loss=mae_split=2_bk.csv')
df27=pd.read_csv('16x16_1000e_opt=sgd_loss=mae_split=2_kb.csv')
df28=pd.read_csv('16x16_1000e_opt=sgd_loss=mae=2_random.csv')