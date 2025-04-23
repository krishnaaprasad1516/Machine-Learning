# House price prediction 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# lets read the data 

data = pd.read_csv(r"C:\Users\krishna\OneDrive\Desktop\work\classroom\21stApr- SLR\21st- SLR\SLR - House price prediction\House_data.csv")

# lets divide the data into dependent and independent variables

x = np.array(data['sqft_living']).reshape(-1,1)

y = np.array(data['price'])

# lets split the data into training and testing data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#lets create the model

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)

#lets predict the values

prediction = model.predict(x_test)

#visualize the training data

plt.scatter(x_train,y_train,color = 'red')

plt.plot(x_train,model.predict(x_train),color = 'blue')

plt.title('House Price Prediction(Training set)')

plt.xlabel('Square Footage')

plt.ylabel('price')

plt.show()

#visualise the testing data
plt.scatter(x_test,y_test,color = 'red')

plt.plot(x_train,model.predict(x_train),color = 'blue')

plt.title('House Price Prediction (Testing Set)')

plt.xlabel('Square Footage')

plt.ylabel('Price')

plt.show()

#lets create the pickle file 


    
import pickle 
filename = 'HousePricePredictionModel.pkl'
with open(filename,'wb') as file:
    pickle.dump(model,file)

import os 
os.getcwd



