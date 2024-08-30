import streamlit as st
import numpy as np
from model import predict_heart_disease
from preprocessing import load_scaler

# Load the scaler
scaler = load_scaler()

# Streamlit UI
st.title('Heart Disease Risk Prediction')

# Input fields for health metrics based on your dataset
age = st.number_input('Age', min_value=1, max_value=100, value=25)
sex = st.selectbox('Sex', options=['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', options=['0', '1', '2', '3'])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[True, False])
restecg = st.selectbox('Resting Electrocardiographic results (restecg)', options=['0', '1', '2'])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', options=[True, False])
oldpeak = st.number_input('ST Depression induced by exercise relative to rest (oldpeak)', min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox('Slope of the peak exercise ST segment (slope)', options=['Upsloping', 'Flat', 'Downsloping'])
ca = st.number_input('Number of major vessels colored by fluoroscopy (ca)', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thalassemia (thal)', options=['Normal', 'Fixed defect', 'Reversable defect'])

# Map categorical inputs to corresponding numerical values
sex = 1 if sex == 'Male' else 0
fbs = 1 if fbs else 0
exang = 1 if exang else 0
thal_mapping = {'Normal': 0, 'Fixed defect': 1, 'Reversable defect': 2}
thal = thal_mapping[thal]
slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
slope = slope_mapping[slope]

# Prepare input data
input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Button to make prediction
if st.button('Predict'):
    result = predict_heart_disease(input_data, scaler)
    if result == 1:
        st.write('The model predicts a high risk of heart disease.')
    else:
        st.write('The model predicts a low risk of heart disease.')
