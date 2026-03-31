from urllib import response

import joblib
import streamlit as st
import pandas as pd
import joblib
import requests
from fastapi.responses import JSONResponse

# Load the trained pipeline
model = joblib.load("model.pkl")
df=joblib.load("processed_dataset.pkl")
path='http://localhost:8001/predict'

st.title('Titanic Survival Prediction')
st.write('Enter your data')

# ---------------- INPUT FIELDS ---------------- #
Pclass = st.selectbox("Passenger Class", ['1st', '2nd', '3rd'])
Pclass=1 if Pclass=='1st' else 2 if Pclass=='2nd' else 3
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 1, 80, 25)
SibSp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare of trip ", min_value=0.0, value=50.0)
Cabin = st.selectbox("Do you have a cabin?", ['Yes', "No"])
Cabin=1 if Cabin=='Yes' else 0
Embarked = st.selectbox("Embarked", ["Southampton", "Cherbourg", "Queenstown"])
Embarked='S' if Embarked=="Southampton" else 'C' if Embarked=="Cherbourg" else 'Q'

input_data = {
        'Pclass': Pclass,
        'Sex':Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch':Parch,
        'Fare':Fare,
        'Cabin':Cabin,
        'Embarked':Embarked

}

if st.button('Predict'):
    try:
        response=requests.post(path,json=input_data)
        if response.status_code==200:
            data = response.json()
            prediction = data["prediction"]
            probability = data["probability"]
            if prediction == 1:
                st.success(f"Passenger likely SURVIVED ❤️ (Confidence: {probability:.2f})")
            else:
                st.error(f"Passenger likely DID NOT survive 💔 (Confidence: {1-probability:.2f})")
        else:
         st.error(f'error in prediction {response.status_code}')
    except Exception as e:
        st.error(f'An error occurred: {e}')