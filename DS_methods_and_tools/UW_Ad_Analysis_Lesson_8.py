
# coding: utf-8

# In[189]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import *


# In[190]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Assemble data ###

# In[191]:


#assign url variable where we will pull data from
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"
#pull down the csv info into a pandas dataframe
ad_df = pd.read_csv(url, header=None, dtype=None)


# In[192]:


#ad_df.shape


# In[193]:


#Get column names
#assign url variable where we will pull data from for column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.names"
response = requests.get(url)
soup = BeautifulSoup(response.content, "lxml").text.split('\n')


# In[194]:


ad_names = pd.DataFrame({'attributes':soup})
print (ad_names)


# ### Data Prep

# In[195]:


#clean up the values in the column
def clean_attr(row):
    return str(row.replace("*","_").replace(".","").replace("0","").replace("1","").replace(",","").replace(":","").strip())
ad_names['attributes'] = ad_names['attributes'].apply(clean_attr)


# In[196]:


ad_names.set_value(1567,'attributes','ad nonad')


# In[197]:


ad_names = ad_names.drop(ad_names.index[[0,1,2,3,8,466,962,1435,1547]])


# In[198]:


#ad_names.shape


# In[199]:


#Add column names to ad dataframe
#create a column name list
ad_columns = ad_names["attributes"].tolist()
ad_df.columns = ad_columns #add column to dataframe


# In[200]:


#remove . from "ad nonad" column
ad_df["ad nonad"] = ad_df["ad nonad"].map(lambda x: str(x)[:-1])


# In[201]:


#check data types
ad_df.dtypes


# In[202]:


#count data types
ad_df.dtypes.value_counts()


# In[203]:


#check why "local" is an object data type as it is supposed to be binary and does not contain any "?" in the expanded view above
print(ad_df.loc[:,"local"].unique()) 


# In[204]:


# Coerce to numeric and impute medians for height continuous column
ad_df.loc[:, "height continuous"] = pd.to_numeric(ad_df.loc[:, "height continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"height continuous"])
ad_df.loc[HasNan, "height continuous"] = np.nanmedian(ad_df.loc[:,"height continuous"])


# In[205]:


plt.hist(ad_df.loc[:, "height continuous"])


# In[206]:


## The high limit for acceptable values is the mean plus 2 standard deviations
LimitHi = ad_df.loc[:, "height continuous"].mean() + 2*(ad_df.loc[:, "height continuous"].std())
print(LimitHi)


# In[207]:


#Replace outliers
TooHigh = ad_df.loc[:, "height continuous"] > LimitHi
ad_df.loc[TooHigh, "height continuous"] = LimitHi


# In[208]:


plt.hist(ad_df.loc[:, "height continuous"])


# In[209]:


# Coerce to numeric and impute medians for width continuous column
ad_df.loc[:, "width continuous"] = pd.to_numeric(ad_df.loc[:, "width continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"width continuous"])
ad_df.loc[HasNan, "width continuous"] = np.nanmedian(ad_df.loc[:,"width continuous"])


# In[210]:


plt.hist(ad_df.loc[:, "width continuous"])


# In[211]:


# Coerce to numeric and impute medians for aratio continuous column
ad_df.loc[:, "aratio continuous"] = pd.to_numeric(ad_df.loc[:, "aratio continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"aratio continuous"])
ad_df.loc[HasNan, "aratio continuous"] = np.nanmedian(ad_df.loc[:,"aratio continuous"])


# In[212]:


plt.hist(ad_df.loc[:, "aratio continuous"])


# In[213]:


## The high limit for acceptable values is the mean plus 2 standard deviations
LimitHi = ad_df.loc[:, "aratio continuous"].mean() + 2*(ad_df.loc[:, "aratio continuous"].std())
print(LimitHi)


# In[214]:


#Replace outliers
TooHigh = ad_df.loc[:, "aratio continuous"] > LimitHi
ad_df.loc[TooHigh, "aratio continuous"] = LimitHi


# In[215]:


plt.hist(ad_df.loc[:, "aratio continuous"])


# In[216]:


# Coerce to numeric and impute medians for local column
ad_df.loc[:, "local"] = pd.to_numeric(ad_df.loc[:, "local"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"local"])
ad_df.loc[HasNan, "local"] = np.nanmedian(ad_df.loc[:,"local"])


# In[217]:


#check data type counts
ad_df.dtypes.value_counts()


# In[218]:


ad_df.head()


# In[219]:


# plot the counts for each category
ad_df.loc[:,"ad nonad"].value_counts().plot(kind='bar')


# In[220]:


#create new numeric colmns
ad_df.loc[:,"ad"] = (ad_df.loc[:,"ad nonad"] == "ad").astype(int)
ad_df.loc[:,"nonad"] = (ad_df.loc[:,"ad nonad"] == "nonad").astype(int)


# In[221]:


# Remove obsolete column "ad nonad"
ad_df = ad_df.drop("ad nonad", axis=1)


# ### Normalization

# In[222]:


#columns to apply z-normalization aka standardization
p = ad_df[['height continuous','width continuous','aratio continuous']]


# In[223]:


#standardization - change the variable so that it’s mean is equal to 0.0 and its standard dev is equal to 1.0
standardization_scale = StandardScaler().fit(p)


# In[224]:


z = standardization_scale.transform(p)


# In[225]:


hc_scaled = pd.DataFrame(z)


# In[226]:


ad_df[['height continuous','width continuous','aratio continuous']] = hc_scaled


# In[227]:


#Drop 'nonad' column. 'ad' will be the target
ad_df = ad_df.drop("nonad", axis=1)


# In[228]:


ad_df.head()


# ### Export data

# In[229]:


#dataframe to csv
ad_df.to_csv('InternetAd_Dataset.csv', index=None)


# ### Data Modeling

# In[230]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 


# In[231]:


def split_dataset(data, r): # split a dataset in matrix format, using a given ratio for the testing set
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	nt = N - n # number of elements in training sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = data[ind_,:-1] # training features
	XX = data[ind,:-1] # testing features
	Y = data[ind_,-1] # training targets
	YY = data[ind,-1] # testing targests
	return X, XX, Y, YY


# In[232]:


r = 0.2 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)
dataset = np.genfromtxt('InternetAd_Dataset.csv', delimiter=",", skip_header=1)
X, XX, Y, YY = split_dataset(dataset, r)


# In[233]:


""" CLASSIFICATION MODELS """
# Logistic regression classifier
print ('\n\n\nLogistic regression classifier\n')
C_parameter = 50. / len(X) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter


# In[234]:


#Training the Model
clf = LogisticRegression(C=C_parameter, multi_class=class_parameter, penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter)
clf.fit(X, Y) 
print ('coefficients:')
print (clf.coef_) # each row of this matrix corresponds to each one of the classes of the dataset
print ('intercept:')
print (clf.intercept_) # each element of this vector corresponds to each one of the classes of the dataset

# Apply the Model
print ('predictions for test set:')
print (clf.predict(XX))
print ('actual class values:')
print (YY)


# In[235]:


# Naive Bayes classifier
print ('\n\nNaive Bayes classifier\n')
nbc = GaussianNB() # default parameters are fine
nbc.fit(X, Y)
print ("predictions for test set:")
print (nbc.predict(XX))
print ('actual class values:')
print (YY)


# In[236]:


# k Nearest Neighbors classifier
print ('\n\nK nearest neighbors classifier\n')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
knn.fit(X, Y)
print ("predictions for test set:")
print (knn.predict(XX))
print ('actual class values:')
print (YY)


# In[237]:


# Support vector machine classifier
t = 0.001 # tolerance parameter
kp = 'rbf' # kernel parameter
print ('\n\nSupport Vector Machine classifier\n')
clf = SVC(kernel=kp, tol=t)
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)
####################


# In[238]:


# Decision Tree classifier
print ('\n\nDecision Tree classifier\n')
clf = DecisionTreeClassifier() # default parameters are fine
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)
####################


# In[239]:


# Random Forest classifier
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)
####################

