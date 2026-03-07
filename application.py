import joblib
import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load("model.pkl")

st.title('Titanic Survival Prediction')
st.write('Enter your data')

# ---------------- INPUT FIELDS ---------------- #
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
cabin = st.selectbox("Cabin known?", [0, 1])
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
familysize = st.number_input("Family size", 1, 10, 1)
isalone = st.selectbox("Is alone?", [0, 1])

# ---------------- CREATE DATAFRAME ---------------- #
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Cabin': [cabin],
    'Embarked': [embarked],
    'familysize': [familysize],
    'isalone': [isalone]
})

# ---------------- PREDICTION ---------------- #
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f"Passenger likely SURVIVED ❤️ (Confidence: {probability:.2f})")
    else:
        st.error(f"Passenger likely DID NOT survive 💔 (Confidence: {1-probability:.2f})")