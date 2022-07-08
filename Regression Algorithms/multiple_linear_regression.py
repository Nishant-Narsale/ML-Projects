# -*- coding: utf-8 -*-

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv("50_startups.csv")

# selecting input for linear regression model
X = dataset.iloc[:,:-1]
# as state is categorical data containing only 3 states
categorical_data_state = pd.get_dummies(X["State"],drop_first=True)
# removing state coloumns
X = X.iloc[:,:-1]
# adding two columns from get_dummies function to input
X = pd.concat([X,categorical_data_state],axis=1)


# selecting output for linear regression model
y = dataset.iloc[:,-1]


# import linear regression model from sklearn
from sklearn.linear_model import LinearRegression
# creating object of model
linear_regressor = LinearRegression()

# splitting dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# training model for training data
linear_regressor.fit(X_train,y_train)

# making prediction
y_predict = linear_regressor.predict(X_test)

# checking accuracy of our model by r-square
from sklearn.metrics import r2_score
accuracy = r2_score(y_test, y_predict) * 100
print(accuracy)
