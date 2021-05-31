# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:30:46 2021

@author: MustafaKuşoğlu
"""

#2021-03-28
import pandas as pd
import numpy as np


df= pd.read_csv("turkish_economy.csv")
df2= pd.read_csv("turkish_economy.csv")
tweets=pd.read_csv("nan_tweets.csv")

tweets=tweets.drop(['Unnamed: 1', 'Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5', 'Unnamed: 6','Unnamed: 7','Unnamed: 8'], axis=1)

res = df[~(df['date'] < '2013-01-01')]
res2=df[~(df['date'] < '2013-01-01')]

res2['date'] =  res.date.apply(pd.to_datetime)
res2.info()
res2['normalised_date'] = res2['date'].dt.normalize()

a=pd.date_range(start = '2013-01-01 00:00:00+00:00', end = '2021-03-28 00:00:00+00:00' ).difference(res2.normalised_date)
a3=pd.date_range(start = '2013-01-01 00:00:00+00:00', end = '2021-03-28 00:00:00+00:00' ).difference(res2.normalised_date)

res2.info()
res2 = res2[~res2.index.duplicated()]

a2=pd.DataFrame(a)
NaN = np.nan
a2["content"] = NaN

a=a.to_frame()
a=a.reset_index(inplace = True)
tweets['date']=a3


tweets.to_csv("eksik_tweetler.csv")
rang=list(range(35656,35815))
tweets['Unnamed: 0']=rang

res3=res2
res3=res3.drop(['date'],axis=1)
tweets=tweets[['Unnamed: 0','content','date']]
tweets['date']=tweets['normalised_date']
tweets=tweets.rename(columns={"date": "normalised_date"})

frames = [res3, tweets]
result = pd.concat(frames)
#result.to_csv("full_tweetler.csv")
df5= pd.read_csv("turkish_economy.csv")