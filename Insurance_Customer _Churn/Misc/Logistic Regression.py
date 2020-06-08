
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize
from sklearn.linear_model import LogisticRegression
from sklearn import metrics , cross_validation
import time
import numpy as np
from itertools import combinations


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


cop = pd.read_csv('Master_Dataset_1 - Copy.csv')


# In[4]:


cop.head()


# In[7]:


tuple(cop.columns)


# In[6]:


Y = cop['Churn'].values


# In[7]:


del cop['Churn']
del cop['Customer ID']
del cop['Effective To Date']


# In[8]:


cop.head()


# In[9]:


cop.columns


# In[10]:


cols_to_transform = ['City', 'Response', 'Coverage',  'Education', 'Employment_Status', 'Gender', 'Location_Code', 
                     'Marital Status','Policy_Type', 'Policy_Rating','Renew_Offer_Type', 'Sales_Channel', 'Feedback']


# In[11]:


def func():
    for i in cop.columns:
        X = cop[i]
        if i in cols_to_transform:
            X = pd.get_dummies(X)
            X = X.values
            model = LogisticRegression()
            model.fit(X,Y)
            
            print(i)
            print(model.score(X, Y))
            
        


# In[12]:


func()


# In[15]:



def func1():
    highest = []
    for l in reversed(range(cop.columns.size)):
        highscore = 0
        tup = ()
        if l <= 1: 
            continue
        com = list(combinations(cop.columns, r = l))
        for i in com:
            emp = None
            for j in i:
                if emp is None:
                    emp = pd.DataFrame(cop[j], columns=[j])
                else:
                    emp = emp.join(cop[j])
                if j in cols_to_transform:
                    emp = pd.get_dummies(emp, columns=[j])
            X = emp.values
            model = LogisticRegression()
            model.fit(X,Y)
            model.score(X, Y)
            #Y_pred = cross_validation.cross_val_predict(LogisticRegression(), X, Y, cv = 10)
            score = model.score(X,Y)
            if score > highscore:
                highscore = score
                tup = i
            
            if score > .70:
                print(i)
                #print(emp.columns)
                print(model.score(X,Y))
        
        highest.append((tup, highscore))
    return highest


# In[ ]:


high = func1()


# In[ ]:


high


# In[19]:


tuple(cop.iloc[0].values)


# In[22]:


X = range(cop.shape[0])


# In[26]:


X = cop.loc[cop['Churn'] == 0]


# In[27]:


X


# In[28]:




