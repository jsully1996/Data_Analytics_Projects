import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

churn_df = pd.read_csv('D:\Project_work\Data\Training_Dataset_1.csv')
col_names = churn_df.columns.tolist()
     
# We don't need these columns
to_drop = ['City','Gender','Location Code','Marital Status','Effective To Date']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
churn_feat_space['Response'] = np.where(churn_feat_space.Response.str.contains('Yes') == True, 1, churn_feat_space.Response)
churn_feat_space['Response'] = np.where(churn_feat_space.Response.str.contains('No') == True, 0, churn_feat_space.Response)



#string_cols = ["Coverage"]
#churn_feat_space_2[string_cols] = churn_feat_space_2[yes_no_cols] == 'yes'
#@string_cols=["Coverage,Employment_Status"]
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Extended') == True, 3, churn_feat_space.Coverage)
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Premium') == True, 2, churn_feat_space.Coverage)
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Basic') == True, 1, churn_feat_space.Coverage)
#print(churn_feat_space['Coverage'])

churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Grade X') == True, 1, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Grade XII') == True, 2, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('PhD') == True, 6, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Diploma') == True, 3, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Graduate') == True, 4, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Master') == True, 5, churn_feat_space.Education)

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

churn_feat_space.to_csv('D:\Project_work\Data\out.csv')