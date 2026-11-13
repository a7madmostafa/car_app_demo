import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(layout="wide")

st.title("Car Price Prediction")

st.write("This app predicts the price of a car based on its features")

df = pd.read_csv('cleaned_data.csv')
st.write("Select the features of the car below")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", df['Location'].unique())
    year = st.number_input("Year of Manufacture", min_value=1999, max_value=2024, value=2015)
    km_driven = st.slider("Kilometers Driven", min_value=100, max_value=500000, value=50000)
    owner_type = st.selectbox("Owner Type", df['Owner_Type'].unique())
    brand = st.selectbox("Brand", df['Brand'].unique())
    model = st.selectbox("Model", df[df['Brand'] == brand]['Model'].unique())
    
with col2:
    transmission = st.selectbox("Transmission Type", df[df['Model'] == model]['Transmission'].unique())
    fuel_type = st.selectbox("Fuel Type", df[df['Model'] == model]['Fuel_Type'].unique())
    mileage = st.number_input("Mileage (km/l)", min_value=df[df['Model'] == model]['Mileage'].min(), max_value=df[df['Model'] == model]['Mileage'].max(), value=df[df['Model'] == model]['Mileage'].min())
    engine = st.number_input("Engine Capacity (cc)", min_value=df[df['Model'] == model]['Engine'].min(), max_value=df[df['Model'] == model]['Engine'].max(), value=df[df['Model'] == model]['Engine'].min())
    power = st.number_input("Power (bhp)", min_value=df[df['Model'] == model]['Power'].min(), max_value=df[df['Model'] == model]['Power'].max(), step=10.0, value=df[df['Model'] == model]['Power'].min())

data = {
    "Location": [location],
    "Kilometers_Driven": [km_driven],
    "Fuel_Type": [fuel_type],
    "Transmission": [transmission],
    "Owner_Type": [owner_type],
    "Mileage": [mileage],
    "Engine": [engine],
    "Power": [power],
    "Brand": [brand],
    "Model": [model],
    "Age": [2020 - year],
    "Km_per_year": [km_driven / (2020 - year)],
    "Power_per_litre": [power / (engine / 1000)],
}

df = pd.DataFrame(data)

pipeline = joblib.load('car_log_price_model.joblib')

if st.button("Predict"):
    prediction = pipeline.predict(df)
    price = np.expm1(prediction[0])
    
    st.write(f"The predicted price of the car is: ₹ {price:,.2f}")