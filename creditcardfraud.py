#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary packages
import pandas as pd
import numpy as np


# In[4]:


#loading the data from Kaggle
data = pd.read_csv('D:\creditcard.csv')


# In[5]:


#Understanding the data
data.head()
#Due to confidentionality issues original features are replaced with V1,V2,.....,V28 which are the result of PCA transformation


# In[6]:


#shape of data
print(data.shape)


# In[7]:


#description of data
print(data.describe)


# In[8]:


#Determining no of fradulent cases in Dataset

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud)/len(valid)      #percentage of fraud cases
print('Percentage of Fraud Cases: {} %'.format(outlier_fraction,))
print("Count of Fraud cases : {}".format(len(fraud)))
print("Count of Valid cases : {}".format(len(valid)))


# In[9]:


print("Amount details of fradulent transcations")
fraud.Amount.describe()


# In[10]:


print("Amount details of Valid transcations")
valid.Amount.describe()


# In[11]:


#dividing the data into X and Y from the dataset
X_data = data.iloc[:,:-1].values          # selecting all columns except last one              
Y_data = data.iloc[:, -1].values           #selecting only the last column
print(X_data.shape)
print(Y_data.shape)


# In[12]:


#Using Scikit-Learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_data,Y_data,test_size = 0.2)


# In[13]:


#Building the Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 6)


# In[14]:


# fit the training data
model.fit(X_train,Y_train)


# In[15]:


predicted = model.predict(X_test)
print("Predicted values : \n",predicted)


# In[16]:


from sklearn import metrics
acc = metrics.accuracy_score(Y_test,predicted)*100
print("Accuracy Score using DecisionTreeClassifier is ",acc)


# In[17]:


#Finding evaluation parameters
#Once the accuracy level in the above step is acceptable
#we go on a further evaluation of the model by finding out different parameters.
#Which use Precision, recall value and F score as our parameters. 
#precision is the fraction of relevant instances among the retrieved instances,
#while recall is the fraction of the total amount of relevant instances that were actually retrieved.
#F score provides a single score that balances both the concerns of precision and recall in one number.
#accuracy score
from sklearn import metrics
acc = metrics.accuracy_score(Y_test,predicted)*100
print("Accuracy Score using DecisionTreeClassifier is ",acc)
from sklearn.metrics import precision_score,recall_score,f1_score
#Precision
Precision = precision_score(Y_test,predicted)
print('Precision Score: ',Precision)
recall = recall_score(Y_test,predicted)
print('Recall Score: ',recall)
f1 = f1_score(Y_test,predicted)
print('F Score: ',f1)


# In[ ]:




