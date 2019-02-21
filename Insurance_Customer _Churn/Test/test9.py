# Import the pandas library.
import pandas
import numpy as np
# Read in the data.
data = pandas.read_csv("numerical01.csv")
# Print the names of the columns in data.
print(data.columns)
print(data.shape)
data.corr()["Churn?"]
# Get all the columns from the dataframe.
columns = data.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["Churn?","Customer ID","Age","clv","clv","resp","cov","edu","emp","incm","loc","mpa","twp","los","lr","gr","comm","mslc","mspi","noc","nop","nopp","pt","pr","ren","sales","tca","fb","count"]]

# Store the variable we'll be predicting on.
target = "Churn?"
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

# Initialize the model with some parameters.
model = LinearRegression()# Fit the model to the data.
model.fit(train[columns], train[target])
# Make predictions.
predictions = model.predict(test[columns])
print("Individual error for each row :",predictions)
ss = mean_squared_error(predictions, test[target])
print("Error percentage in model :")
print(ss)