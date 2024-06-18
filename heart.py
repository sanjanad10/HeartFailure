import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data
@st.cache_resource
def load_data():
    return pd.read_csv('C:\\Users\\sanjana\\Desktop\\Machine Learning\\HeartFailure.csv')



data = load_data()

# Feature selection based on heatmap
relevant_features = ['age', 'high_blood_pressure', 'serum_creatinine',
                     'sex', 'ejection_fraction', 'diabetes', 'smoking', 'time']

# Prepare data
x = data[relevant_features].copy()
y = data['DEATH_EVENT']

# Data split
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7, random_state=83)

# Standardize data for SVC
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# Train models
model_LR = LogisticRegression()
model_LR.fit(train_x, train_y)

model_SVM = SVC(probability=True)  # Enable probability estimates
model_SVM.fit(train_x_scaled, train_y)

# User Input for prediction
st.subheader('Enter Input Values')
age = st.slider('Age', float(data['age'].min()), float(data['age'].max()), float(data['age'].mean()))
high_blood_pressure = st.selectbox('High Blood Pressure', [0, 1])
serum_creatinine = st.slider('Serum Creatinine', float(data['serum_creatinine'].min()), float(data['serum_creatinine'].max()), float(data['serum_creatinine'].mean()))
sex = st.selectbox('Sex', ['Male', 'Female'])
sex = 1 if sex == 'Male' else 0
ejection_fraction = st.slider('Ejection Fraction (%)', int(data['ejection_fraction'].min()), int(data['ejection_fraction'].max()), int(data['ejection_fraction'].mean()))
diabetes = st.selectbox('Diabetes', [0, 1])
smoking = st.selectbox('Smoking', [0, 1])
time = st.slider('Follow-up Time (days)', int(data['time'].min()), int(data['time'].max()), int(data['time'].mean()))

# Create input dataframe
input_data = pd.DataFrame({'age': [age],
                           'high_blood_pressure': [high_blood_pressure],
                           'serum_creatinine': [serum_creatinine],
                           'sex': [sex],
                           'ejection_fraction': [ejection_fraction],
                           'diabetes': [diabetes],
                           'smoking': [smoking],
                           'time': [time]})

# Predictions and probabilities
if st.button('Predict'):
    prediction_lr = model_LR.predict(input_data)[0]
    prob_lr = model_LR.predict_proba(input_data)[0]

    prediction_svm = model_SVM.predict(input_data)[0]
    prob_svm = model_SVM.predict_proba(input_data)[0]

    st.subheader('Prediction and Probabilities (Logistic Regression Model)')
    death_event_lr = 'Yes' if prediction_lr == 1 else 'No'
    st.write(f'Does the model predict death event? {death_event_lr}')
    st.write(f'Probability of death event: {prob_lr[1]:.2f}')

    st.subheader('Prediction and Probabilities (SVM Model)')
    death_event_svm = 'Yes' if prediction_svm == 1 else 'No'
    st.write(f'Does the model predict death event? {death_event_svm}')
    st.write(f'Probability of death event: {prob_svm[1]:.2f}')
