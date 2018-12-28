#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:05:10 2018

@author: Ryan
"""
#use the pandas package to convert csv to a dataframe
import pandas as pd

#assign url varaible where we will pull data from
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/bridges/bridges.data.version1"

#pull down the csv info into a pandas dataframe
bridge_df = pd.read_csv(url, header=None)

#print the first 5 rows of the data set
print(bridge_df.head())

#create the list of column names
bridge_columns = ["IDENTIF", "RIVER", "LOCATION", "ERECTED", "PURPOSE", 
                 "LENGTH", "LANES", "CLEAR-G", "T-OR-D", "MATERIAL", 
                 "SPAN", "REL-L", "TYPE"]

#assign column names to the dataframe
bridge_df.columns = bridge_columns

#print the first 5 rows of the data set to check if names were assigned
print(bridge_df.head())