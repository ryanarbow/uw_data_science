#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:58:00 2018

@author: Ryan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

#assign url variable where we will pull data from
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"

#pull down the csv info into a pandas dataframe
ad_df = pd.read_csv(url, header=None, dtype=None)

ad_df.head()

ad_df.shape

#pull in the local file names.csv info into a pandas dataframe
ad_names = pd.read_csv('~/Downloads/column_names.csv', header=None)

ad_names.head()

#add a column name
column_name = ["attributes"]

#assign the column name
ad_names.columns = column_name

#clean up the values in the column
def clean_attr(row):
    return str(row.replace("*","_").replace(".","").replace("0","").replace("1","").replace(",","").replace(":","").strip())
ad_names['attributes'] = ad_names['attributes'].apply(clean_attr)

ad_names.head()

#create a column name list
ad_columns = ad_names["attributes"].tolist()

#add column names to ad dataframe
ad_df.columns = ad_columns

#expanded data view beyond .head()
ad_df[:25]

#check data types
ad_df.dtypes

#count data types
ad_df.dtypes.value_counts()

#check why "local" is an object data type as it does not contain any "?" in the expanded view above
print(ad_df.loc[:,"local"].unique())

# Coerce to numeric and impute medians for height continuous column
ad_df.loc[:, "height continuous"] = pd.to_numeric(ad_df.loc[:, "height continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"height continuous"])
ad_df.loc[HasNan, "height continuous"] = np.nanmedian(ad_df.loc[:,"height continuous"])

# Coerce to numeric and impute medians for width continuous column
ad_df.loc[:, "width continuous"] = pd.to_numeric(ad_df.loc[:, "width continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"width continuous"])
ad_df.loc[HasNan, "width continuous"] = np.nanmedian(ad_df.loc[:,"width continuous"])

# Coerce to numeric and impute medians for aratio continuous column
ad_df.loc[:, "aratio continuous"] = pd.to_numeric(ad_df.loc[:, "aratio continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"aratio continuous"])
ad_df.loc[HasNan, "aratio continuous"] = np.nanmedian(ad_df.loc[:,"aratio continuous"])

# Coerce to numeric and impute medians for local column
ad_df.loc[:, "local"] = pd.to_numeric(ad_df.loc[:, "local"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"local"])
ad_df.loc[HasNan, "local"] = np.nanmedian(ad_df.loc[:,"local"])

#check data type counts
ad_df.dtypes.value_counts()

#remove . from "ad nonad" column
ad_df["ad nonad"] = ad_df["ad nonad"].map(lambda x: str(x)[:-1])

ad_df.head()

###Summary of data set clean up
#First - Changed data to be more readable by removing extra characters with a
#function I created called "clean_attr".
#Second - I found 5 data types which were objects; however, 4 of them should be numeric so
#I used coerced to numeric and imputed medians.
#Third - I removed the "." from the ad/nonad