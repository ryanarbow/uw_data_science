
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression


# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Assemble data ###

# In[62]:


#assign url variable where we will pull data from
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"
#pull down the csv info into a pandas dataframe
ad_df = pd.read_csv(url, header=None, dtype=None)


# In[63]:


#ad_df.shape


# In[64]:


#Get column names
#assign url variable where we will pull data from for column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.names"
response = requests.get(url)
soup = BeautifulSoup(response.content, "lxml").text.split('\n')


# In[65]:


ad_names = pd.DataFrame({'attributes':soup})
print (ad_names)


# ### Data Prep

# In[66]:


#clean up the values in the column
def clean_attr(row):
    return str(row.replace("*","_").replace(".","").replace("0","").replace("1","").replace(",","").replace(":","").strip())
ad_names['attributes'] = ad_names['attributes'].apply(clean_attr)


# In[67]:


ad_names.set_value(1567,'attributes','ad nonad')


# In[68]:


ad_names = ad_names.drop(ad_names.index[[0,1,2,3,8,466,962,1435,1547]])


# In[69]:


#Add column names to ad dataframe
#create a column name list
ad_columns = ad_names["attributes"].tolist()
ad_df.columns = ad_columns #add column to dataframe


# In[70]:


#remove . from "ad nonad" column
ad_df["ad nonad"] = ad_df["ad nonad"].map(lambda x: str(x)[:-1])


# In[71]:


#check data types
ad_df.dtypes


# In[72]:


#count data types
ad_df.dtypes.value_counts()


# In[73]:


#check why "local" is an object data type as it is supposed to be binary and does not contain any "?" in the expanded view above
print(ad_df.loc[:,"local"].unique()) 


# In[74]:


# Coerce to numeric and impute medians for height continuous column
ad_df.loc[:, "height continuous"] = pd.to_numeric(ad_df.loc[:, "height continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"height continuous"])
ad_df.loc[HasNan, "height continuous"] = np.nanmedian(ad_df.loc[:,"height continuous"])


# In[75]:


plt.hist(ad_df.loc[:, "height continuous"])


# In[76]:


## The high limit for acceptable values is the mean plus 2 standard deviations
LimitHi = ad_df.loc[:, "height continuous"].mean() + 2*(ad_df.loc[:, "height continuous"].std())
print(LimitHi)


# In[77]:


#Replace outliers
TooHigh = ad_df.loc[:, "height continuous"] > LimitHi
ad_df.loc[TooHigh, "height continuous"] = LimitHi


# In[78]:


plt.hist(ad_df.loc[:, "height continuous"])


# In[79]:


# Coerce to numeric and impute medians for width continuous column
ad_df.loc[:, "width continuous"] = pd.to_numeric(ad_df.loc[:, "width continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"width continuous"])
ad_df.loc[HasNan, "width continuous"] = np.nanmedian(ad_df.loc[:,"width continuous"])


# In[80]:


plt.hist(ad_df.loc[:, "width continuous"])


# In[81]:


# Coerce to numeric and impute medians for aratio continuous column
ad_df.loc[:, "aratio continuous"] = pd.to_numeric(ad_df.loc[:, "aratio continuous"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"aratio continuous"])
ad_df.loc[HasNan, "aratio continuous"] = np.nanmedian(ad_df.loc[:,"aratio continuous"])


# In[82]:


plt.hist(ad_df.loc[:, "aratio continuous"])


# In[83]:


## The high limit for acceptable values is the mean plus 2 standard deviations
LimitHi = ad_df.loc[:, "aratio continuous"].mean() + 2*(ad_df.loc[:, "aratio continuous"].std())
print(LimitHi)


# In[84]:


#Replace outliers
TooHigh = ad_df.loc[:, "aratio continuous"] > LimitHi
ad_df.loc[TooHigh, "aratio continuous"] = LimitHi


# In[85]:


plt.hist(ad_df.loc[:, "aratio continuous"])


# In[86]:


# Coerce to numeric and impute medians for local column
ad_df.loc[:, "local"] = pd.to_numeric(ad_df.loc[:, "local"], errors='coerce')
HasNan = np.isnan(ad_df.loc[:,"local"])
ad_df.loc[HasNan, "local"] = np.nanmedian(ad_df.loc[:,"local"])


# In[87]:


#check data type counts
ad_df.dtypes.value_counts()


# In[88]:


ad_df.head()


# In[89]:


# plot the counts for each category
ad_df.loc[:,"ad nonad"].value_counts().plot(kind='bar')


# In[90]:


#create new numeric colmns
ad_df.loc[:,"ad"] = (ad_df.loc[:,"ad nonad"] == "ad").astype(int)
ad_df.loc[:,"nonad"] = (ad_df.loc[:,"ad nonad"] == "nonad").astype(int)


# In[91]:


# Remove obsolete column "ad nonad"
ad_df = ad_df.drop("ad nonad", axis=1)


# ### Normalization

# In[92]:


#columns to apply z-normalization aka standardization
p = ad_df[['height continuous','width continuous','aratio continuous']]


# In[93]:


#standardization - change the variable so that it’s mean is equal to 0.0 and its standard dev is equal to 1.0
standardization_scale = StandardScaler().fit(p)


# In[94]:


z = standardization_scale.transform(p)


# In[95]:


hc_scaled = pd.DataFrame(z)


# In[96]:


ad_df[['height continuous','width continuous','aratio continuous']] = hc_scaled


# In[97]:


#Drop 'nonad' column. 'ad' will be the target
ad_df = ad_df.drop("nonad", axis=1)


# In[98]:


ad_df.head()


# ### Export data

# In[99]:


#dataframe to csv
ad_df.to_csv('InternetAd_Dataset.csv', index=None)


# ### Data Modeling

# ### Split dataset - training/test

# In[100]:


def split_dataset(data, r): # split a dataset
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


# In[101]:


r = 0.2 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)
dataset = np.genfromtxt('InternetAd_Dataset.csv', delimiter=",", skip_header=1)
X, XX, Y, YY = split_dataset(dataset, r)


# ### Train model

# In[102]:


""" CLASSIFICATION MODELS """
# Logistic regression classifier
C_parameter = 50. / len(X) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter


# In[103]:


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


# ### Assess model

# In[104]:


Y = (clf.predict(XX))
T = (YY)


# In[105]:


# Confusion Matrix
CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))


# In[106]:


# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'green' # Line Color


# In[107]:


preds = clf.predict_proba(XX)[:,1]
fpr, tpr, th = roc_curve(T, preds) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))


# In[108]:


plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()


# In[109]:


print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, preds), 2), "\n")

