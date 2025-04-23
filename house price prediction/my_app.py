import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r'C:\Users\krishna\vs code projects\HousePricePredictionModel.pkl','rb'))

# Set the title of the Streamlit app
st.title("House Price Prediction App🏠")

# Add a brief description
st.write("This app predicts the House Price based on Square Footage using a simple linear regression model.")

# Add input widget for user to enter Square Footage
sqft_living = st.number_input("Enter Square Footage :", min_value=0.0, max_value=15000.0, value=1000.0, step=50.0)

# When the button is clicked, make predictions
if st.button("Predict House Price"):
    # Make a prediction using the trained model
    sqft_input = np.array([[sqft_living]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(sqft_input)
   
    # Display the result
    st.success(f"The predicted House Price for {sqft_living} square foot is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of House Price and Square Footage.built model by krishna")