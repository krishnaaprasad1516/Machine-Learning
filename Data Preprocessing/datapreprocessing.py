import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\krishna\Downloads\18AprData (2).csv")

x= dataset.iloc[:,:-1].values

y= dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

imputer = imputer.fit(x[:,1:3])

x[:,1:3] = imputer.transform(x[:,1:3])

# IMPUTE CATEGORICAL VALUE FOR INDEPDENT
 
from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

labelencoder_x.fit_transform(x[:,0]) 

x[:,0] = labelencoder_x.fit_transform(x[:,0]) 

## IMPUTE CATEGORICAL VALUE FOR DEPENDENT 

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

# SPLIT THE DATA 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
