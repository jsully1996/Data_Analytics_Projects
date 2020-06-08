import csv
import sys
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import itertools
churn_df = pd.read_csv('D:\Project_work\Data\Master_Dataset_50k.csv')
col_names = churn_df.columns.tolist()
     
# We don't need these columns
to_drop = ['City','Gender','Marital Status','Effective To Date']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
churn_feat_space['Response'] = np.where(churn_feat_space.Response.str.contains('Yes') == True, 1, churn_feat_space.Response)
churn_feat_space['Response'] = np.where(churn_feat_space.Response.str.contains('No') == True, 0, churn_feat_space.Response)


churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Extended') == True, 3, churn_feat_space.Coverage)
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Premium') == True, 2, churn_feat_space.Coverage)
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Basic') == True, 1, churn_feat_space.Coverage)
#print(churn_feat_space['Coverage'])

churn_feat_space['Location_Code'] = np.where(churn_feat_space.Location_Code.str.contains('Urban') == True, 3, churn_feat_space.Location_Code)
churn_feat_space['Location_Code'] = np.where(churn_feat_space.Location_Code.str.contains('Suburban') == True, 2, churn_feat_space.Location_Code)
churn_feat_space['Location_Code'] = np.where(churn_feat_space.Location_Code.str.contains('Rural') == True, 1, churn_feat_space.Location_Code)

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


#print(churn_feat_space)
"""
churn_feat_space.to_csv('numerical.csv')

clv_mean = churn_feat_space['Customer Lifetime Value'].mean()
cov_mean = churn_feat_space['Coverage'].mean()
loc_mean = churn_feat_space['Location_Code'].mean()
edu_mean = churn_feat_space['Education'].mean()
empstat_mean = churn_feat_space['Employment_Status'].mean()
income_mean = churn_feat_space['Income'].mean()
mpa_mean = churn_feat_space['Monthly Premium Auto'].mean()
twp_mean = churn_feat_space['Total Written Premium'].mean()
loss_mean = churn_feat_space['Losses'].mean()
lr_mean = churn_feat_space['Loss Ratio'].mean()
gr_mean = churn_feat_space['Growth Rate'].mean()
comm_mean = churn_feat_space['Commissions'].mean()
mslc_mean = churn_feat_space['Months Since Last Claim'].mean()
mspi_mean = churn_feat_space['Months Since Policy Inception'].mean()
noc_mean = churn_feat_space['Number of Open Complaints'].mean()
nop_mean = churn_feat_space['Number of Policies'].mean()
nopp_mean = churn_feat_space['Number of previous policies'].mean()
polt_mean = churn_feat_space['Policy_Type'].mean()
polr_mean = churn_feat_space['Policy_Rating'].mean()
rot_mean = churn_feat_space['Renew_Offer_Type'].mean()
slschnl_mean = churn_feat_space['Sales_Channel'].mean()
tca_mean = churn_feat_space['Total Claim Amount'].mean()
fdbk_mean = churn_feat_space['Feedback'].mean()

churnness = np.zeros(574361,dtype=int)
#print(churnness[341])
count = np.zeros(24,dtype=int)
for row in churn_feat_space['Customer Lifetime Value']:
    print(row)
    count[0]+=1    
"""
data = churn_feat_space
# Print the names of the columns in data.
print(data.columns)
print(data.shape)
data.corr()["Churn"]
# Get all the columns from the dataframe.
columns = data.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["Churn","Customer ID","Age","clv","clv","resp","cov","edu","emp","incm","loc","mpa","twp","los","lr","gr","comm","mslc","mspi","noc","nop","nopp","pt","pr","ren","sales","tca","fb","count"]]

# Store the variable we'll be predicting on.
target = "Churn"
# Import a convenience function to split the sets.
from sklearn.cross_validation import train_test_split

# Generate the training set.  Set random_state to be able to replicate
train, test = train_test_split(data, test_size=0.2)
# Print the shapes of both sets.
print("Shape of training data :",train.shape)
print("Shape of testing data :",test.shape)
# Import the random forest model.
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, regression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 

# Initialize the model with some parameters.
model = LinearRegression()# Fit the model to the data.
model.fit(train[columns], train[target])

model2 = SVC()
model2.fit(train[columns], train[target])

model3=RandomForestClassifier()
model3.fit(train[columns], train[target])

model4 = KNeighborsClassifier()
model4.fit(train[columns], train[target])

# Make predictions.
predictions = model.predict(test[columns])
print("Values according to Linear Regression model  :",predictions)
ss = mean_squared_error(predictions, test[target])
print("Error percentage in Linear Regression model  :")
print(ss)

predictions2 = model2.predict(test[columns])
print("Values according to SVC model:",predictions2)
ss2 = mean_squared_error(predictions2, test[target])
print("Error percentage in SVC model:")
print(ss2)

predictions3 = model3.predict(test[columns])
print("Values according to Random Forest Classifier:",predictions3)
ss3 = mean_squared_error(predictions3, test[target])
print("Error percentage in Random Forest Classifier :")
print(ss3)

predictions4 = model4.predict(test[columns])
print("Values according to K Neighbors Classifier:",predictions4)
ss4 = mean_squared_error(predictions4, test[target])
print("Error percentage in K Neighbors Classifier:")
print(ss4)