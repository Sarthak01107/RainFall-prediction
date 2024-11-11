# app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the pre-trained scaler

# Define a function for user input
def user_input_features():
    st.sidebar.header('Input Parameters')
    Location = st.sidebar.selectbox('Location', ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'])
    MinTemp = st.sidebar.slider('MinTemp', -10.0, 50.0, 15.0)
    MaxTemp = st.sidebar.slider('MaxTemp', -10.0, 50.0, 25.0)
    Rainfall = st.sidebar.slider('Rainfall', 0.0, 500.0, 50.0)
    Evaporation = st.sidebar.slider('Evaporation', 0.0, 20.0, 5.0)
    Sunshine = st.sidebar.slider('Sunshine', 0.0, 15.0, 8.0)
    WindGustDir = st.sidebar.selectbox('WindGustDir', ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']) 
    WindGustSpeed = st.sidebar.slider('WindGustSpeed', 0, 150, 40)
    WindDir9am = st.sidebar.selectbox('Wind Direction at 9am', ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    WindDir3pm = st.sidebar.selectbox('Wind Direction at 3pm', ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    WindSpeed9am = st.sidebar.slider('WindSpeed9am', 0, 100, 20)
    WindSpeed3pm = st.sidebar.slider('WindSpeed3pm', 0, 100, 15)
    Humidity9am = st.sidebar.slider('Humidity9am', 0, 100, 50)
    Humidity3pm = st.sidebar.slider('Humidity3pm', 0, 100, 50)
    Pressure9am = st.sidebar.slider('Pressure9am', 900.0, 1100.0, 1010.0)
    Pressure3pm = st.sidebar.slider('Pressure3pm', 900.0, 1100.0, 1010.0)
    Cloud9am = st.sidebar.slider('Cloud9am', 0, 9, 4)
    Cloud3pm = st.sidebar.slider('Cloud3pm', 0, 9, 4)
    Temp9am = st.sidebar.slider('Temp9am', -10.0, 50.0, 15.0)
    Temp3pm = st.sidebar.slider('Temp3pm', -10.0, 50.0, 25.0)
    RainToday = st.sidebar.selectbox('RainToday', ['Yes', 'No'])

    features = {
        'Location': Location,
        'MinTemp': MinTemp,
        'MaxTemp': MaxTemp,
        'Rainfall': Rainfall,
        'Evaporation': Evaporation,
        'Sunshine': Sunshine,
        'WindGustDir': WindGustDir,
        'WindGustSpeed': WindGustSpeed,
        'WindDir9am': WindDir9am,
        'WindDir3pm': WindDir3pm,
        'WindSpeed9am': WindSpeed9am,
        'WindSpeed3pm': WindSpeed3pm,
        'Humidity9am': Humidity9am,
        'Humidity3pm': Humidity3pm,
        'Pressure9am': Pressure9am,
        'Pressure3pm': Pressure3pm,
        'Cloud9am': Cloud9am,
        'Cloud3pm': Cloud3pm,
        'Temp9am': Temp9am,
        'Temp3pm': Temp3pm,
        'RainToday': 1 if RainToday == 'Yes' else 0
    }

    return pd.DataFrame(features, index=[0])

# Main app
st.title("Rainfall Prediction")

input_df = user_input_features()

# Display user inputs
st.write("User Input Parameters:")
st.write(input_df)

# Encode categorical columns before scaling
le = LabelEncoder()
categorical_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
for col in categorical_columns:
    input_df[col] = le.fit_transform(input_df[col].astype(str))

# Scale the input data using the pre-fitted scaler
input_df_scaled = scaler.transform(input_df)

# Make predictions
prediction = model.predict(input_df_scaled)
prediction_proba = model.predict_proba(input_df_scaled)
threshold = 0.6  # or any value between 0 and 1
st.write("Prediction: ", "Yes" if prediction_proba[0][1] > threshold else "No")
st.write("Prediction Probability: ", prediction_proba[0][1])


