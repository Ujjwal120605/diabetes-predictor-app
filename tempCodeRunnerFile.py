
import streamlit as st
import joblib
import numpy as np

st.title("🩺 Diabetes Prediction App")

# Load the model
model = joblib.load("model.joblib")

# Input fields
pregnancies = st.number_input('Pregnancies', 0, 20)
glucose = st.number_input('Glucose', 0, 200)
blood_pressure = st.number_input('Blood Pressure', 0, 150)
skin_thickness = st.number_input('Skin Thickness', 0, 100)
insulin = st.number_input('Insulin', 0, 900)
bmi = st.number_input('BMI', 0.0, 70.0)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', 0.0, 2.5)
age = st.number_input('Age', 10, 100)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    prediction = model.predict(input_data)[0]
    st.success("✅ Diabetic" if prediction == 1 else "❌ Not Diabetic")