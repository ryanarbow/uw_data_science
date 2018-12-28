
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import *


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#assign url variable where we will pull data from
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"


# In[4]:


#pull down the csv info into a pandas dataframe
ad_df = pd.read_csv(url, header=None, dtype=None)


# In[5]:


ad_df.head()


# In[6]:


ad_df.shape


# In[7]:


#assign url variable where we will pull data from for column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.names"


# In[8]:


response = requests.get(url)


# In[9]:


soup = BeautifulSoup(response.content, "lxml").text.split('\n')


# In[10]:


print(soup)


# In[11]:


ad_names = pd.DataFrame({'attributes':soup})
print (ad_names)


# In[12]:


#clean up the values in the column
def clean_attr(row):
    return str(row.replace("*","_").replace(".","").replace("0","").replace("1","").replace(",","").replace(":","").strip())
ad_names['attributes'] = ad_names['attributes'].apply(clean_attr)


# In[13]:


ad_names.set_value(1567,'attributes','ad nonad')


# In[14]:


ad_names = ad_names.drop(ad_names.index[[0,1,2,3,8,466,962,1435,1547]])


# In[15]:


ad_names.shape


# In[16]:


#create a column name list
ad_columns = ad_names["attributes"].tolist()


# In[17]:


#ad_names["attributes"][1400:1450]


# In[18]:


#add column names to ad dataframe
ad_df.columns = ad_columns


# In[19]:


ad_df.shape


# In[20]:


#expanded data view beyond .head()
ad_df[:25]


# In[21]:


#remove . from "ad nonad" column
ad_df["ad nonad"] = ad_df["ad nonad"].map(lambda x: str(x)[:-1])


# In[22]:


#check data types
ad_df.dtypes


# In[23]:


#count data types
ad_df.dtypes.value_counts()


# In[24]:


#check why "local" is an object data type as it is supposed to be binary and does not contain any "?" in the expanded view above
print(ad_df.loc[:,"local"].unique()) 


# In[25]:


# Coerce to numeric and impute medians for height continuous column
ad_df.loc[:, "height continuous"] = pd.to_numeric(ad_df.loc[:, "height continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"height continuous"])
ad_df.loc[HasNan, "height continuous"] = np.nanmedian(ad_df.loc[:,"height continuous"])


# In[26]:


plt.hist(ad_df.loc[:, "height continuous"])


# In[27]:


## The high limit for acceptable values is the mean plus 2 standard deviations
LimitHi = ad_df.loc[:, "height continuous"].mean() + 2*(ad_df.loc[:, "height continuous"].std())
print(LimitHi)


# In[28]:


#Replace outliers
TooHigh = ad_df.loc[:, "height continuous"] > LimitHi
ad_df.loc[TooHigh, "height continuous"] = LimitHi


# In[29]:


plt.hist(ad_df.loc[:, "height continuous"])


# In[30]:


# Coerce to numeric and impute medians for width continuous column
ad_df.loc[:, "width continuous"] = pd.to_numeric(ad_df.loc[:, "width continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"width continuous"])
ad_df.loc[HasNan, "width continuous"] = np.nanmedian(ad_df.loc[:,"width continuous"])


# In[31]:


plt.hist(ad_df.loc[:, "width continuous"])


# In[32]:


# Coerce to numeric and impute medians for aratio continuous column
ad_df.loc[:, "aratio continuous"] = pd.to_numeric(ad_df.loc[:, "aratio continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"aratio continuous"])
ad_df.loc[HasNan, "aratio continuous"] = np.nanmedian(ad_df.loc[:,"aratio continuous"])


# In[33]:


plt.hist(ad_df.loc[:, "aratio continuous"])


# In[34]:


## The high limit for acceptable values is the mean plus 2 standard deviations
LimitHi = ad_df.loc[:, "aratio continuous"].mean() + 2*(ad_df.loc[:, "aratio continuous"].std())
print(LimitHi)


# In[35]:


#Replace outliers
TooHigh = ad_df.loc[:, "aratio continuous"] > LimitHi
ad_df.loc[TooHigh, "aratio continuous"] = LimitHi


# In[36]:


plt.hist(ad_df.loc[:, "aratio continuous"])


# In[37]:


# Coerce to numeric and impute medians for local column
ad_df.loc[:, "local"] = pd.to_numeric(ad_df.loc[:, "local"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"local"])
ad_df.loc[HasNan, "local"] = np.nanmedian(ad_df.loc[:,"local"])


# In[38]:


#check data type counts
ad_df.dtypes.value_counts()


# In[39]:


ad_df.head()


# In[40]:


# plot the counts for each category
ad_df.loc[:,"ad nonad"].value_counts().plot(kind='bar')


# In[41]:


#create new numeric colmns
ad_df.loc[:,"ad"] = (ad_df.loc[:,"ad nonad"] == "ad").astype(int)
ad_df.loc[:,"nonad"] = (ad_df.loc[:,"ad nonad"] == "nonad").astype(int)


# In[42]:


# Remove obsolete column "ad nonad"
ad_df = ad_df.drop("ad nonad", axis=1)


# In[43]:


ad_df.head()


# ### Normalization

# In[44]:


ad_df.describe()


# In[45]:


p = ad_df[['height continuous','width continuous','aratio continuous']]


# In[46]:


standardization_scale = StandardScaler().fit(p)


# In[47]:


z = standardization_scale.transform(p)


# In[48]:


hc_scaled = pd.DataFrame(z)


# In[49]:


ad_df[['height continuous','width continuous','aratio continuous']] = hc_scaled


# In[50]:


ad_df.head()


# In[54]:


#Create limited dataframe to fit within 5MB submission limit
limit_ad_df = ad_df[:1000]


# In[57]:


#limit_ad_df


# In[56]:


#dataframe to csv
limit_ad_df.to_csv('RyanArbow-M02-Dataset.csv', index=None)

