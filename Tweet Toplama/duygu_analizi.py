# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 02:48:15 2021

@author: MustafaKuşoğlu
"""
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


df= pd.read_csv("full_tweetler.csv")
df2=pd.read_csv("full_tweetler.csv")

def cleanTxt(text):
    text=re.sub(r'@[A-Za-z0-9]+','',text)
    text=re.sub(r'#','',text)
    text=re.sub(r'RT[\s]','',text)
    text=re.sub(r'https?:\/\/\S+','',text)
    #text=re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)','',text)
    
    return text

df2['content']=df2['content'].apply(cleanTxt)

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

df2['subject']=df2['content'].apply(getSubjectivity)
df2['polarity']=df2['content'].apply(getPolarity)

#Word Cloud kısmı
allWords=' '.join([twts for twts in df['content']])
wordCloud = WordCloud(width= 500,height=300,random_state=21,max_font_size=119).generate(allWords)


plt.imshow(wordCloud,interpolation="bilinear")
plt.axis('off')
plt.show()


def getAnalysis(score):
    if score < 0 :
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df2['analysis']=df2['polarity'].apply(getAnalysis)


negatif=1
sortedDF= df2.sort_values(by=['polarity'])
for i in range(0,sortedDF.shape[0]):
    if(sortedDF['analysis'][i]=='Negative'):
        #print(str(negatif) + ')' +sortedDF['content'][i])
        #print()
        negatif=negatif+1

tarafsiz=1
sortedDF= df2.sort_values(by=['polarity'])
for i in range(0,sortedDF.shape[0]):
    if(sortedDF['analysis'][i]=='Neutral'):
        #print(str(tarafsiz) + ')' +sortedDF['content'][i])
       # print()
        tarafsiz=tarafsiz+1
        
pozitif=1
sortedDF= df2.sort_values(by=['polarity'])
for i in range(0,sortedDF.shape[0]):
    if(sortedDF['analysis'][i]=='Positive'):
        #print(str(pozitif) + ')' +sortedDF['content'][i])
        #print()
        pozitif=pozitif+1


#df2 = df2[~df2.content.duplicated()]
df2.info()
df2['normalised_date']=pd.to_datetime(df2['normalised_date'])
df2.info()

df2.info()
sortedDF= df2.sort_values(by=['normalised_date'])
i = 1
for i in range(0,3813):
    if(sortedDF['normalised_date'][i]==sortedDF['normalised_date'][i+1]):
         df2['yeni_id']=i
         i=i+1
        
df2['Group_ID'] = df2.groupby('normalised_date').grouper.group_info[0]

df2=df2.drop(['week','gun','yeni_id'],axis=1)
df2['ort'] = df2.groupby('Group_ID')['polarity'].transform('mean')

duygu_or=df2['normalised_date']
duygu_or=duygu_or.to_frame()
duygu_or['ortalama']=df2['ort']
duygu_or=duygu_or.sort_values(by=['normalised_date'])
duygu_or=duygu_or[~duygu_or.normalised_date.duplicated()]

duygu_or.to_csv("gunkul_duygu_ort.csv")