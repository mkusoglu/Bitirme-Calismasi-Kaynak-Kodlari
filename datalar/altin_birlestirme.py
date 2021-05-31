# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 02:11:02 2021

@author: MustafaKuşoğlu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('gram_altın.csv', error_bad_lines=False)
df2 = pd.read_excel('altin_eksikveriler.xlsx', error_bad_lines=False)

df=df.drop(['Açılış','Yüksek','Düşük','Fark %'],axis=1)


df.drop(df.tail(188).index,inplace=True)


df2.info()

altin_toplam= df.append(df2)
altin_toplam=altin_toplam.reset_index(drop=True)

altin_toplam.to_csv("altin_acilis2.csv")
df23= pd.read_csv('altin_acilis2.csv', error_bad_lines=False)
df23['Tarih']=pd.to_datetime(df23['Tarih'])
df23['Şimdi']=pd.to_numeric(df23['Şimdi'])


yen=pd.read_excel('altin_acilis2.xlsx', error_bad_lines=False)
yen.info()
yen.to_csv("altin_son.csv")
yen1 = pd.read_csv('altin_son.csv', error_bad_lines=False)
#altın verileri 2013-2021 arası birleştirildi

