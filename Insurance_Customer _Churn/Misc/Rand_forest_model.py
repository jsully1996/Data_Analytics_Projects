
# coding: utf-8

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics , cross_validation
import time
import numpy as np
from itertools import combinations


# In[18]:


pd.set_option('display.max_columns', None)


# In[19]:


cop = pd.read_csv('Master_Dataset_1 - Copy.csv')


# In[20]:


cop.head()


# In[21]:


cop.columns


# In[22]:


Y = cop['Churn'].values
del cop['Churn']
del cop['Customer ID']
del cop['Effective To Date']


# In[23]:


del cop['City']
del cop['Gender']
del cop['Marital Status']


# In[24]:


cop.head()


# In[25]:


cop.columns


# In[26]:


cols_to_transform = ['City', 'Response', 'Coverage',  'Education', 'Employment_Status', 'Gender', 'Location_Code', 
                     'Marital Status','Policy_Type', 'Policy_Rating','Renew_Offer_Type', 'Sales_Channel', 'Feedback']


# In[27]:



def func():
    for i in cop.columns:
        X = cop[i]
        if i in cols_to_transform:
            X = pd.get_dummies(X)
            X = X.values
            model = RandomForestClassifier()
            model.fit(X,Y)
            print(i)
            print(model.score(X, Y))
            
        


# In[28]:


func()


# In[10]:


for i in cols_to_transform:
    del cop[i]


# In[41]:



def func1():
    highest = []
    for l in range(cop.columns.size):
        highscore = 0
        tup = ()
        if not (l == 10  or l == 1000) : 
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
            model = RandomForestClassifier(n_estimators=10)
            model.fit(X,Y)
            model.score(X, Y)
            #Y_pred = cross_validation.cross_val_predict(LogisticRegression(), X, Y, cv = 10)
            score = model.score(X,Y)
            if score > highscore:
                highscore = score
                tup = i
            
            if score > .98:
                print(i)
                #print(emp.columns)
                print(model.score(X,Y))
        
        highest.append((tup, highscore))
    return highest


# In[42]:


high = func1()


# In[ ]:


high

