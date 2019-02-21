from __future__ import division
import pandas as pd
import numpy as np
import xlrd
import sqlite3
import math
import re
#from sklearn.ensemble import RandomForestClassifier as RF

churn_df = pd.read_csv('D:\Project_work\Data\Master_Dataset_1.csv')
col_names = churn_df.columns.tolist()

print ("Column names:")
print (col_names)


# We don't need these columns
to_drop = ['Customer ID','City','Education','Gender','Location Code','Marital Status','Effective To Date']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Response"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

#string_cols = ["Coverage"]
#churn_feat_space_2[string_cols] = churn_feat_space_2[yes_no_cols] == 'yes'
#@string_cols=["Coverage,Employment_Status"]
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Extended') == True, 3, churn_feat_space.Coverage)
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Premium') == True, 2, churn_feat_space.Coverage)
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Basic') == True, 1, churn_feat_space.Coverage)
#print(churn_feat_space['Coverage'])
churn_feat_space['Employment_Status'] = np.where(churn_feat_space.Employment_Status.str.contains('Unemployed') == True, 0, churn_feat_space.Employment_Status)
churn_feat_space['Employment_Status'] = np.where(churn_feat_space.Employment_Status.str.contains('Employed') == True, 3, churn_feat_space.Employment_Status)
churn_feat_space['Employment_Status'] = np.where(churn_feat_space.Employment_Status.str.contains('On leave') == True, 1, churn_feat_space.Employment_Status)
churn_feat_space['Employment_Status'] = np.where(churn_feat_space.Employment_Status.str.contains('Retired') == True, 2, churn_feat_space.Employment_Status)
#print(churn_feat_space['Employment_Status'])

churn_feat_space['Policy_Type'] = np.where(churn_feat_space.Policy_Type.str.contains('Corporate') == True, 4, churn_feat_space.Policy_Type)
churn_feat_space['Policy_Type'] = np.where(churn_feat_space.Policy_Type.str.contains('Special') == True, 3, churn_feat_space.Policy_Type)
churn_feat_space['Policy_Type'] = np.where(churn_feat_space.Policy_Type.str.contains('Privileged') == True, 2, churn_feat_space.Policy_Type)
churn_feat_space['Policy_Type'] = np.where(churn_feat_space.Policy_Type.str.contains('Personal') == True, 1, churn_feat_space.Policy_Type)
#print(churn_feat_space['Policy_Type'])

churn_feat_space['Policy_Rating'] = np.where(churn_feat_space.Policy_Rating.str.contains('Bad') == True, 1, churn_feat_space.Policy_Rating)
churn_feat_space['Policy_Rating'] = np.where(churn_feat_space.Policy_Rating.str.contains('Good') == True, 3, churn_feat_space.Policy_Rating)
churn_feat_space['Policy_Rating'] = np.where(churn_feat_space.Policy_Rating.str.contains('Average') == True, 2, churn_feat_space.Policy_Rating)
churn_feat_space['Policy_Rating'] = np.where(churn_feat_space.Policy_Rating.str.contains('Excellent') == True, 4, churn_feat_space.Policy_Rating)

churn_feat_space['Renew_Offer_Type'] = np.where(churn_feat_space.Renew_Offer_Type.str.contains('Offer 1') == True, 1, churn_feat_space.Renew_Offer_Type)
churn_feat_space['Renew_Offer_Type'] = np.where(churn_feat_space.Renew_Offer_Type.str.contains('Offer 2') == True, 2, churn_feat_space.Renew_Offer_Type)
churn_feat_space['Renew_Offer_Type'] = np.where(churn_feat_space.Renew_Offer_Type.str.contains('Offer 3') == True, 3, churn_feat_space.Renew_Offer_Type)
churn_feat_space['Renew_Offer_Type'] = np.where(churn_feat_space.Renew_Offer_Type.str.contains('Offer 4') == True, 4, churn_feat_space.Renew_Offer_Type)


churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Online') == True, 2, churn_feat_space.Sales_Channel)
churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Branch') == True, 4, churn_feat_space.Sales_Channel)
churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Call Center') == True, 1, churn_feat_space.Sales_Channel)
churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Agent') == True, 5, churn_feat_space.Sales_Channel)
churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Other') == True, 3, churn_feat_space.Sales_Channel)

churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Excellent') == True, 6, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Good') == True, 5, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Satisfactory') == True, 4, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Average') == True, 3, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Bad') == True, 2, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Horrible') == True, 1, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Not Given') == True, 0, churn_feat_space.Feedback)
#print(churn_feat_space['Feedback'])

#X = churn_feat_space.as_matrix().astype(np.float)

# This is important
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#print ("Feature space holds %d observations and %d features" %X.shape)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
