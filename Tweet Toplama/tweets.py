# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 18:48:40 2021

@author: MustafaKuşoğlu
"""



import pandas as pd
import snscrape.modules.twitter as sntwitter
import itertools


search = '"turkish economy"'

# the scraped tweets, this is a generator
scraped_tweets = sntwitter.TwitterSearchScraper(search).get_items()

# slicing the generator to keep only the first 100 tweets
sliced_scraped_tweets = itertools.islice(scraped_tweets, 50000)

# convert to a DataFrame and keep only relevant columns
df = pd.DataFrame(sliced_scraped_tweets)[['date', 'content']]

pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(
    '"turkish economy"').get_items(), 50000))

df.to_csv("turkish_economy.csv")