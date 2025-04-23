# Import necessary libraries
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv(r"C:\Users\krishna\Downloads\Salary_Data.csv")

# Feature selection (independent variable X and dependent variable y)
x= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]

# Split the dataset into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.20, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predicting the results for the test set
y_pred = regressor.predict(x_test)

# Visualizing the Training set results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color ='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('years of Experience')
plt.ylabel('Salary')
plt.show()

# i want to predict the future
m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

pred_12yr_emp_exp = m_slope * 12 + c_intercept
print(pred_12yr_emp_exp)

pred_20yr_emp_exp = m_slope * 20 + c_intercept
print(pred_20yr_emp_exp)

bias = regressor.score(x_train,y_train)
print(bias)

variance = regressor.score(x_test,y_test)
print(variance)

#STATISTICS FOR MACHINE LEARNING

dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset['Salary'].median()

dataset['Salary'].mode()

dataset.var()

dataset['Salary'].var()

dataset.std()

dataset['Salary'].std()

from scipy.stats import variation

variation(dataset.values)

variation(dataset['Salary'])

dataset.corr()

# SSR
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

# SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#r2
r_square = 1-SSR/SST
print(r_square)

import pickle

filename = 'linear_regression_model.pkl'

with open(filename,'wb') as file:
    pickle.dump(regressor,file)

print("model has been pickled and saved as linear_regression_model.pkl")

import os 
os.getcwd()









