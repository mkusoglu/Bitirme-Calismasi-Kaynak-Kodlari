# -*- coding: utf-8 -*-
"""
Created on Sat May  1 22:48:11 2021

@author: MustafaKuşoğlu
"""

import pandas as pd
import numpy as np
from datetime import datetime
import math
#altın işi

df = pd.read_csv('C:/Users/MustafaKuşoğlu/Desktop/BİTİRME/datalar/dolar_tl.csv', error_bad_lines=False)
df2=pd.read_csv('C:/Users/MustafaKuşoğlu/Desktop/BİTİRME/datalar/Ham Petrol.csv')
df2_=pd.read_csv('C:/Users/MustafaKuşoğlu/Desktop/BİTİRME/datalar/Ham Petrol.csv')
df3=pd.read_excel('C:/Users/MustafaKuşoğlu/Desktop/BİTİRME/datalar/bullion.xlsx')
df4=pd.read_excel('C:/Users/MustafaKuşoğlu/Desktop/BİTİRME/datalar/faiz_hergun.xlsx')
df5=pd.read_csv('C:/Users/MustafaKuşoğlu/Desktop/BİTİRME/tweet/gunkul_duygu_ort.csv')
df6=pd.read_excel('C:/Users/MustafaKuşoğlu/Desktop/BİTİRME/datalar/tufe.xlsx')
tahmin=pd.read_excel('C:/Users/MustafaKuşoğlu/Downloads/try.xlsx')


df=df.drop(['Açılış','Yüksek','Düşük','Fark %'],axis=1)
df2=df2.drop(['Açılış','Yüksek','Düşük','Fark %','Hac.'],axis=1)
df2_=df2_.drop(['Açılış','Yüksek','Düşük','Fark %','Hac.'],axis=1)
df5=df5.drop(['Unnamed: 0'],axis=1)

#tarih kısmını düzeltme
def converter(s, format_list, format_output):
    for date_format in format_list:
        try:
            return datetime.strptime(s, date_format).strftime(format_output)
        except ValueError:
            continue      
    return 'Nuffin'

df= df.stack().str.replace('.','-').unstack()
df2= df2.stack().str.replace('.','-').unstack()
df2_= df2_.stack().str.replace('.','-').unstack()
df3['Tarih']= df3.stack().str.replace('.','-').unstack()
tahmin['Tarih']= tahmin.stack().str.replace('.','-').unstack()
tahmin = tahmin.replace(np.nan, '2013-01-01', regex=True)

#df6 = df6.stack().str.replace('.','-').unstack()



for i in range(0,df.shape[0]):
    df['Tarih'][i] = converter(df['Tarih'][i],format_list=['%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y-%d-%m'],format_output='%Y-%m-%d')
    i=i+1

for i in range(0,df.shape[0]):
    df2['Tarih'][i] = converter(df2['Tarih'][i],format_list=['%d-%m-%Y'],format_output='%Y-%m-%d')
    i=i+1

for i in range(0,df.shape[0]):
    df3['Tarih'][i] = converter(df3['Tarih'][i],format_list=['%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y-%d-%m'],format_output='%Y-%m-%d')
    i=i+1

for i in range(0,tahmin.shape[0]):
    tahmin['Tarih'][i] = converter(tahmin['Tarih'][i],format_list=['%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y-%d-%m'],format_output='%Y-%m-%d')
    i=i+1
    
for i in range(0,df2_.shape[0]):
    df2_['Tarih'][i] = converter(df2_['Tarih'][i],format_list=['%d-%m-%Y'],format_output='%Y-%m-%d')
    i=i+1



df['Tarih']=pd.to_datetime(df['Tarih'])
df2['Tarih']=pd.to_datetime(df2['Tarih'])
df3['Tarih']=pd.to_datetime(df3['Tarih'])
tahmin['Tarih']=pd.to_datetime(tahmin['Tarih'])
df2_['Tarih']=pd.to_datetime(df2_['Tarih'])

#verilerin sayısallaştırılması
df['Şimdi'] = (df['Şimdi'].replace(',','.',regex=True))
df2['Şimdi'] = (df2['Şimdi'].replace(',','.',regex=True))
tahmin['Şimdi'] = (tahmin['Şimdi'].replace(',','.',regex=True))
df2_['Şimdi'] = (df2_['Şimdi'].replace(',','.',regex=True))



df['Şimdi'] =pd.to_numeric(df['Şimdi'])
df2['Şimdi'] =pd.to_numeric(df2['Şimdi'])
tahmin['Şimdi'] =pd.to_numeric(tahmin['Şimdi'])
df5['Tarih'] = pd.to_datetime(df5.Tarih).dt.tz_localize(None)
df2_['Şimdi'] =pd.to_numeric(df2_['Şimdi'])


#sütun isimleri hazırlanması
df=df.rename(columns={"Şimdi": "dolar"})
df2=df2.rename(columns={"Şimdi": "petrol"})
df3=df3.rename(columns={"Şimdi": "altın"})
df5=df5.rename(columns={"normalised_date": "Tarih"})
tahmin=tahmin.rename(columns={"Şimdi": "tahmin"})
df2_=df2_.rename(columns={"Şimdi": "petrol"})



#birleştirme versiyon 1.0
result=pd.merge(df,df2_,on='Tarih',how='left')
result=pd.merge(result,df3,on='Tarih',how='left')
result=pd.merge(result,df4,on='Tarih',how='left')
result=pd.merge(result,df5,on='Tarih',how='left')
result=pd.merge(result,df6,on='Tarih',how='left')
result=pd.merge(result,tahmin,on='Tarih',how='left')

result=result.drop(['altın','tahmin'],axis=1)
result.to_csv('merge_with_tufe.csv')

#nan values
merge=pd.read_csv('merge_with_tufe.csv')
merge_yedek=merge
merge.info()

merge['petrol']=merge['petrol'].fillna(0)
temp=1
for i in range(0,merge.shape[0]):
    if (merge['petrol'][i]==0):
        merge['petrol'][i]=temp
    if (merge['petrol'][i]!=0):
        temp=merge['petrol'][i]
    i=i+1

merge['altın']=merge['altın'].fillna(0)
temp1=1
for i in range(0,merge.shape[0]):
    if (merge['altın'][i]==0):
        merge['altın'][i]=temp1
    if (merge['altın'][i]!=0):
        temp1=merge['altın'][i]
    i=i+1

merge['ortalama']=merge['ortalama'].fillna(0)
merge.to_csv('merge_with_tufe.csv')



