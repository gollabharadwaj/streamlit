# depolying the model other than using AWS kind of(easy method)

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set the title for the web app
st.title("Promotion Prediction App")

# Read the dataset to populate input options for dropdowns
df = pd.read_csv('train_LZdllcl.xls')

# Create the input elements
department = st.selectbox("Department", pd.unique(df['department']))
region = st.selectbox("Region", pd.unique(df['region']))
education = st.selectbox("Education", pd.unique(df['education']))
gender = st.selectbox("Gender", pd.unique(df['gender']))
recruitment_channel = st.selectbox("Recruitment Channel", pd.unique(df['recruitment_channel']))

# Non-categorical columns
no_of_trainings = st.number_input("Number of Trainings", min_value=0, step=1)
age = st.number_input("Age", min_value=0, step=1)
previous_year_rating = st.number_input("Previous Year Rating", min_value=0.0, max_value=5.0, step=0.1)
length_of_service = st.number_input("Length of Service", min_value=0, step=1)
KPIs_met_80 = st.number_input("KPIs Met > 80%", min_value=0, max_value=1, step=1)
awards_won = st.number_input("Awards Won", min_value=0, max_value=1, step=1)
avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, step=1)

# Map the user inputs to the respective column format
inputs = {
    'department': department,
    'region': region,
    'education': education,
    'gender': gender,
    'recruitment_channel': recruitment_channel,
    'no_of_trainings': no_of_trainings,
    'age': age,
    'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service,
    'KPIs_met >80%': KPIs_met_80,
    'awards_won?': awards_won,
    'avg_training_score': avg_training_score
}
# load the model from the pickle file
model = joblib.load('promote_pipeline_model.pkl')

# action for submit button
if st.button('Predict'):
    X_input = pd.DataFrame(inputs, index=[0])
    prediction = model.predict(X_input)
    st.write("The predicted value is:")
    st.write(prediction)