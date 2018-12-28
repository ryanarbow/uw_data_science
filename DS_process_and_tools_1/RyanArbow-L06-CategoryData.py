#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:37:15 2018

@author: Ryan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

#assign url variable where we will pull data from
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"

#pull down the csv info into a pandas dataframe
ad_df = pd.read_csv(url, header=None, dtype=None)

#ad_df.head()

#ad_df.shape

#assign url variable where we will pull data from for column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.names"

#pull url content and store content as a requests object
response = requests.get(url)

#split url content based on column names
soup = BeautifulSoup(response.content, "lxml").text.split('\n')

#add column names into dataframe and assign column name 'attributes'
ad_names = pd.DataFrame({'attributes':soup})

print (ad_names)

#clean up the values in the column
def clean_attr(row):
    return str(row.replace("*","_").replace(".","").replace("0","").replace("1","").replace(",","").replace(":","").strip())
ad_names['attributes'] = ad_names['attributes'].apply(clean_attr)

#set empty column name for ad nonad values
ad_names.set_value(1567,'attributes','ad nonad')

#drop non column name attributes
ad_names = ad_names.drop(ad_names.index[[0,1,2,3,8,466,962,1435,1547]])

#create a column name list
ad_columns = ad_names["attributes"].tolist()

#add column names to ad dataframe
ad_df.columns = ad_columns

#ad_df.shape
ad_df.head()

#remove . from "ad nonad" column
ad_df["ad nonad"] = ad_df["ad nonad"].map(lambda x: str(x)[:-1])

#check data types
ad_df.dtypes

#count data types
ad_df.dtypes.value_counts()

#check why "local" is an object data type as it is supposed to be binary and does not contain any "?" in the expanded view above
print(ad_df.loc[:,"local"].unique()) 

# Coerce to numeric and impute medians for height continuous column
ad_df.loc[:, "height continuous"] = pd.to_numeric(ad_df.loc[:, "height continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"height continuous"])
ad_df.loc[HasNan, "height continuous"] = np.nanmedian(ad_df.loc[:,"height continuous"])

plt.hist(ad_df.loc[:, "height continuous"])

## The high limit for acceptable values is the mean plus 2 standard deviations
LimitHi = ad_df.loc[:, "height continuous"].mean() + 2*(ad_df.loc[:, "height continuous"].std())
print(LimitHi)

#Replace outliers
TooHigh = ad_df.loc[:, "height continuous"] > LimitHi
ad_df.loc[TooHigh, "height continuous"] = LimitHi

plt.hist(ad_df.loc[:, "height continuous"])

# Coerce to numeric and impute medians for width continuous column
ad_df.loc[:, "width continuous"] = pd.to_numeric(ad_df.loc[:, "width continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"width continuous"])
ad_df.loc[HasNan, "width continuous"] = np.nanmedian(ad_df.loc[:,"width continuous"])

plt.hist(ad_df.loc[:, "width continuous"])

# Coerce to numeric and impute medians for aratio continuous column
ad_df.loc[:, "aratio continuous"] = pd.to_numeric(ad_df.loc[:, "aratio continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"aratio continuous"])
ad_df.loc[HasNan, "aratio continuous"] = np.nanmedian(ad_df.loc[:,"aratio continuous"])

plt.hist(ad_df.loc[:, "aratio continuous"])

## The high limit for acceptable values is the mean plus 2 standard deviations
LimitHi = ad_df.loc[:, "aratio continuous"].mean() + 2*(ad_df.loc[:, "aratio continuous"].std())
print(LimitHi)

#Replace outliers
TooHigh = ad_df.loc[:, "aratio continuous"] > LimitHi
ad_df.loc[TooHigh, "aratio continuous"] = LimitHi

plt.hist(ad_df.loc[:, "aratio continuous"])

# Coerce to numeric and impute medians for local column
ad_df.loc[:, "local"] = pd.to_numeric(ad_df.loc[:, "local"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"local"])
ad_df.loc[HasNan, "local"] = np.nanmedian(ad_df.loc[:,"local"])

#check data type counts
ad_df.dtypes.value_counts()

ad_df.head()

# plot the counts for each category
ad_df.loc[:,"ad nonad"].value_counts().plot(kind='bar')

#create new numeric colmns
ad_df.loc[:,"ad"] = (ad_df.loc[:,"ad nonad"] == "ad").astype(int)
ad_df.loc[:,"nonad"] = (ad_df.loc[:,"ad nonad"] == "nonad").astype(int)

# Remove obsolete column "ad nonad"
ad_df = ad_df.drop("ad nonad", axis=1)

ad_df.head()

ad_df.shape

###Summary of Category Data assignment
#This data set has 3 continous attributes; all others binary 
#I found 5 data types which were objects; however, 4 of them should be numeric so
#I used coerced to numeric and imputed medians.
#For the 'ad nonad' object data I used a bar chart to plot the categories
#I then created new columns, and dropped the obsolete 'ad nonad' column