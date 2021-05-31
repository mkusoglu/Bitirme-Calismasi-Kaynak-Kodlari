# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:54:22 2021

@author: MustafaKuşoğlu
"""


import pandas as pd
import numpy as np

df = pd.read_csv('dolar_tl.csv', error_bad_lines=False)

df2 = pd.read_csv('Ham Petrol.csv', error_bad_lines=False)

df3 = pd.read_csv('gram_altın.csv', error_bad_lines=False)

df4=pd.read_excel('faiz_2013-2020.xlsx')

df5 = pd.read_excel('altin_eksikveriler.xlsx', error_bad_lines=False)
df6=pd.read_excel('altin_eksikveriler.xlsx', error_bad_lines=False)
df7=pd.read_csv('gram_altın.csv', error_bad_lines=False)
df3.info()



df3['Tarih']=pd.to_datetime(df3['Tarih'])
df3.info()
df3=df3.drop(['Açılış','Yüksek','Düşük','Fark %'],axis=1)
df3=df3.sort_values(by='Tarih')

df5.info()


#altın açılış birleştir
df3=df3.reindex(index=df3.index[::-1])
df3=df3.drop(['Şimdi','Yüksek','Düşük','Fark %'],axis=1)
out = df5.append(df3)



df6.info()
df7['Tarih']=pd.to_datetime(df7['Tarih'])
df7['Açılış'] =pd.to_numeric(df7['Açılış'],errors = 'coerce')

df7=df7[(df7['Tarih'] > '2013-12-31')]
df7=df7.drop(['Şimdi','Yüksek','Düşük','Fark %'],axis=1)
df7.info()
out = df6.append(df7)
out['Açılış'] =pd.to_numeric(out['Açılış'])

out.to_csv("altin_acilis.csv")
yeni=pd.read_csv('altin_acilis.csv', error_bad_lines=False)
yeni.info()

yeni= yeni.stack().str.replace(',','.').unstack()
yeni['Açılış'] =pd.to_numeric(yeni['Açılış'])
yeni['Tarih']=pd.to_datetime(yeni['Tarih'])
yeni=yeni.sort_values(by='Tarih')
yeni.to_csv("altin_acilis2.csv")

#ham petrol açılış fiyatları al
#null verileri ekle
#df1,2,3 birleştir