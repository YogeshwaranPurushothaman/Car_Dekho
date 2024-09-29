import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sys

# Load the saved Random Forest model
model_path = r"C:\Users\Bhuvanesh\Desktop\Yogesh Guvi\Car Dekho\Model\best_random_forest_model_updated.pkl"
model = joblib.load(model_path)

# Title of the web app
st.title("Car Price Prediction App")

# User input section
st.header("Enter the car details")

# Example of features (you'll need to adjust these inputs based on your feature set)
modelYear = st.number_input("Model Year", min_value=1990, max_value=2023, step=1)
mileage = st.number_input("Mileage (km/l)", min_value=0.0, max_value=50.0, step=0.1)
engine_cc = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, step=50)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
no_of_features = st.number_input("Number of Features", min_value=0, max_value=50, step=1)

# Create a DataFrame from the input
data = pd.DataFrame({
    'modelYear': [modelYear],
    'Mileage': [mileage],
    'Engine(CC)': [engine_cc],
    'Car overview - Kms Driven': [kms_driven],
    'No of Features': [no_of_features]
})

# Button to predict
if st.button("Predict Price"):
    prediction = model.predict(data)
    st.write(f"The predicted price of the car is ₹ {prediction[0]:,.2f} lakhs")
