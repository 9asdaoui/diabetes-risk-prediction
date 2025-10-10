import pandas as pd
import pickle
import streamlit as st


st.title("Diabetes Prediction App")

with open("../models/best_SVM_model.pkl", "rb") as f:
    model = pickle.load(f)

st.write("Enter patient data below:")

Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=150, value=70)
SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5)
Age = st.number_input("Age", min_value=0, max_value=120, value=33)

if st.button("Predict"):
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
                              columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][prediction]
    
    st.write(f"### Prediction: {'Diabetic' if prediction==1 else 'Non-Diabetic'}")
    st.write(f"### Confidence: {prediction_proba:.2f}")
