#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors
# 
# ### I am working in a Data set from a company! They've hidden the feature column names but they given the data and the target classes.
# 
# ### I have tried to use KNN to create a model that directly predicts a class for a new data point based off of the features.

# In[1]:


# Checking whether the files are in the same folder or not
import os
print(os.listdir())


# In[2]:


# importing essential libraries which we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# To get the graphs inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Loading the dataSet into my DataFrame
dataSet = pd.read_csv('Classified Data.csv', index_col = 0)


# In[4]:


#checking the dataSet
dataSet.head(10)


# # Standardize the Variables
# 
# The KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale.

# In[5]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  #creating the instance


# In[6]:


# Fitting the dataSet by dropping the Target class
scaler.fit(dataSet.drop('TARGET CLASS', axis = 1)) #Only the features are taken into consideration


# In[7]:


#Transforming the data
scaled_features = scaler.transform(dataSet.drop('TARGET CLASS',axis=1))


# In[8]:


#Recreating the featured DataFrame
df_feat = pd.DataFrame(scaled_features,columns=dataSet.columns[:-1]) # The data is scaled_feature
# Ckecking the Standardized data
df_feat.head()


# # Splitting the dataSet into Train and Test Data using the Standardized Data

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,dataSet['TARGET CLASS'], test_size=0.30)


# # Using KNN

# In[10]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)  #Taking K = 1


# In[11]:


knn.fit(X_train,y_train)


# In[12]:


pred = knn.predict(X_test)


# # Predictions and Evaluations
# Let's evaluate our KNN model!

# In[13]:


from sklearn.metrics import classification_report,confusion_matrix


# In[14]:


print(confusion_matrix(y_test,pred))


# In[15]:


print(classification_report(y_test,pred))


# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value:

# In[16]:


error_rate = []

# Will take some time
for i in range(1,100, 2):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[17]:


error_index = error_rate.index(min(error_rate))
print(error_index)


# In[18]:


plt.figure(figsize=(10,6)) #Setting the Size of the Graph
plt.plot(range(1,100, 2),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[19]:


print("The minimum error rate is {}.".format(min(error_rate)))
print("The index with Minimum error rate is {}.".format(error_rate.index(min(error_rate))))


# In[20]:


knn = KNeighborsClassifier(n_neighbors=error_index)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K={}'.format(error_index))
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

