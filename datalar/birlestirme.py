# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 02:11:02 2021

@author: MustafaKuşoğlu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('altin_son.csv', error_bad_lines=False)
df['Tarih']=pd.to_datetime(df['Tarih'])

df['Tarih'] = pd.to_datetime(df['Tarih'], dayfirst=True)
df=df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)


df3 = pd.read_csv('dolar_tl.csv', error_bad_lines=False)
df3=df3.drop(['Açılış','Yüksek','Düşük','Fark %'],axis=1)
df3=df3.rename(columns={"Şimdi": "dolar"})
df3['Tarih']=pd.to_datetime(df3['Tarih'])


#birleştirmeden önce tarihlere el at
df4 = pd.read_csv('Ham Petrol.csv', error_bad_lines=False)
df4=df4.drop(['Açılış','Yüksek','Düşük','Fark %','Hac.'],axis=1)
df4=df4.rename(columns={"Şimdi": "petrol"})
df4['Tarih']=pd.to_datetime(df4['Tarih'])


result=pd.merge(df,df3,on='Tarih',how='left')
result=pd.merge(result,df4,on='Tarih',how='left')
df5=pd.read_csv('C:/Users/MustafaKuşoğlu/Desktop/BİTİRME/tweet/gunkul_duygu_ort.csv')

result2=pd.merge(result,df4,on='Tarih',how='left')

result.to_csv('3_merge.csv')
#sonraki gün
df9=pd.read_excel('faiz_hergun.xlsx')
df9['Tarih']=pd.to_datetime(df9['Tarih'])


df6=pd.read_csv('3_merge.csv')
df6['Tarih']=pd.to_datetime(df6['Tarih'])

result2=pd.merge(df6,df9,on='Tarih',how='left')

i=result2[result2['Unnamed: 0']==74].index
i1=result2[result2['Unnamed: 0']==24].index
i2=result2[result2['Unnamed: 0']==48].index
i3=result2[result2['Unnamed: 0']==75].index
i4=result2[result2['Unnamed: 0']==40].index
i5=result2[result2['Unnamed: 0']==46].index
i6=result2[result2['Unnamed: 0']==72].index
i7=result2[result2['Unnamed: 0']==73].index
i8=result2[result2['Unnamed: 0']==23].index
i9=result2[result2['Unnamed: 0']==47].index

result2=result2.drop(i3)
result2=result2.drop(i4)
result2=result2.drop(i5)
result2=result2.drop(i6)
result2=result2.drop(i7)
result2=result2.drop(i8)
result2=result2.drop(i9)
result2=result2.drop(i2)
result2=result2.drop(i1)
result2=result2.drop(i)

result2=pd.read_csv('4_merge.csv')
result2=result2.sort_values(by='Tarih')
result2.to_csv('4_merge.csv')
#
result2=result2.rename(columns={"Şimdi": "altın"})
result2=result2.rename(columns={"Unnamed: 0": "id"})
result2.info()

duygu=pd.read_csv('C:/Users/MustafaKuşoğlu/Desktop/BİTİRME/tweet/gunkul_duygu_ort.csv')
duygu=duygu.rename(columns={"normalised_date": "Tarih"})
duygu=duygu.drop(['Unnamed: 0'],axis=1)
duygu['Tarih']=pd.to_datetime(duygu['Tarih'])
result2.info()
duygu.info()

duygu['Tarih'] = pd.to_datetime(duygu.Tarih).dt.tz_localize(None)
duygu.info()

result2=pd.merge(result2,duygu,on='Tarih',how='left')
result2=result2.rename(columns={"Unnamed: 0": "id"})
result2=result2.rename(columns={"Şimdi": "tahmin1"})
result2=result2.drop(['Unnamed: 0.1'],axis=1)
result3=result2
tahmin=pd.read_excel('tahmin1.xlsx')
tahmin=tahmin.drop(['Açılış','Yüksek','Düşük','Fark %'],axis=1)
tahmin=tahmin.drop(['Tarih'],axis=1)

duygu= duygu.stack().str.replace('.','-').unstack()
tahmin= tahmin.stack().str.replace('.','-').unstack()

tahmin['Tarih']=pd.to_datetime(tahmin['Tarih'])
tahmin = tahmin.iloc[1:]
tahmin.loc[2153] = '1,7837'

from datetime import datetime


def converter(s, format_list, format_output):
    for date_format in format_list:
        try:
            return datetime.strptime(s, date_format).strftime(format_output)
        except ValueError:
            continue      
    return 'Nuffin'

for i in range(0,result2.shape[0]):
    result2['Tarih'][i] = converter(result2['Tarih'][i],format_list=['%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y-%d-%m'],format_output='%Y-%m-%d')
    i=i+1

for i in range(0,duygu.shape[0]):
    duygu['Tarih'][i] = converter(duygu['Tarih'][i],format_list=['%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y-%d-%m'],format_output='%Y-%m-%d')
    i=i+1
    
for i in range(0,tahmin.shape[0]):
    tahmin['Tarih'][i] = converter(tahmin['Tarih'][i],format_list=['%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y-%d-%m'],format_output='%Y-%m-%d')
    i=i+1

result2['Tarih']=pd.to_datetime(result2['Tarih'])

result2=pd.merge(result2,tahmin,on='Tarih',how='left')
result2=pd.merge(result2,duygu,on='Tarih',how='left')


result2.info()
result2['dolar'] = (result2['dolar'].replace(',','.',regex=True))
result2['petrol'] = (result2['petrol'].replace(',','.',regex=True))
result2['tahmin1'] = (result2['tahmin1'].replace(',','.',regex=True))
result2.to_csv('all_merge.csv')
#indexleri düzenle
safe=result2
result2=result2.drop(['tahmin1'],axis=1)

result2=result2.sort_values(by='Tarih')

result2.insert(7, "tahmin1", tahmin, True)

result2['dolar'] =pd.to_numeric(result2['dolar'])
result2['petrol'] =pd.to_numeric(result2['petrol'])
result2['tahmin1'] =pd.to_numeric(result2['tahmin1'])
result2.to_csv('all_merge1.csv')



#tahmin1'i düzelt
