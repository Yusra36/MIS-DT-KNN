#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import important packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


# In[2]:


#to read the dataset file 
#dataset = pd.read_csv(r'C:\Users\vxlli\Downloads\spam.csv')
dataset = pd.read_csv(r'E:\Semester 8\MIS\archive (2)\data.csv', encoding='ISO-8859-1')


# In[3]:


#to read the first 5 rows of the dataset 
dataset.head().T


# In[4]:


#dropping all null values 
dataset = dataset.drop(['Unnamed: 32'], axis=1)


# In[5]:


#split the data
Y = dataset['diagnosis']
X = dataset.drop(['diagnosis'], axis = 1)


# In[6]:


#to read the first 5 rows of the dataset of X
X.head().T


# In[7]:


#split the train and test  data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=42)


# In[8]:


len(X_train)


# In[9]:


len(X_test)


# In[10]:


#create KNN Classifier 
#import KNN neighbors pakage 
from sklearn.neighbors import KNeighborsClassifier

#parameters in the classifier 
knn = KNeighborsClassifier(n_neighbors=3) # the value of K

#call KNN fit
knn.fit(X_train, Y_train)


# In[11]:


#find out the prediction and compare it with y test to see the accuracy 
knn.score(X_test, Y_test)


# In[12]:


#import confution matrix 
from sklearn.metrics import confusion_matrix

#use knn for making prediction 
Y_pred = knn.predict(X_test)

#it has 2 parameters truth and prediction 
cm = confusion_matrix(Y_test, Y_pred)
cm 


# In[13]:


#use classifier report 
#import the classification report 
from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))


# In[14]:


#train a simple decision tree classifier on the data
#import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

DTClassifier = DecisionTreeClassifier()
DTClassifier.fit (X_train, Y_train)


# In[15]:


#decision macking process 
#the texture visulization
from sklearn import tree

print(tree.export_text(DTClassifier))


# In[16]:


# Putting the feature names and class names into variables
fn = ['radius_mean','texture_mean','perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst','smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
cn = ['M', 'B']


# In[17]:


#call the function to show the decision tree process in visual visulization

fig = plt.figure(figsize= (80,60))
tree.plot_tree(DTClassifier, feature_names=fn, class_names=cn, filled= True )


# In[18]:


#find out the prediction and compare it with y test to see the accuracy 
DTClassifier.score(X_test, Y_test)


# In[19]:


#import confution matrix 
from sklearn.metrics import confusion_matrix

#use DT for making prediction 
Y_pred = DTClassifier.predict(X_test)

#it has 2 parameters truth and prediction 
cm1 = confusion_matrix(Y_test, Y_pred)
cm1 


# In[20]:


#use classifier report 

print(classification_report(Y_test, Y_pred))


# In[ ]:




