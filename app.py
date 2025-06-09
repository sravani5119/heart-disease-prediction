import streamlit as st
import pandas as pd
import pickle

# Load saved model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("❤️ Heart Disease Prediction")

# Input fields with your dataset columns
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0,1])
cp = st.number_input("Chest Pain Type (0-4)", 0, 4, 4)
bp = st.number_input("Resting Blood Pressure (BP)", 80, 200, 130)
chol = st.number_input("Cholesterol", 100, 600, 322)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0,1])
ekg = st.number_input("EKG results (0-2)", 0, 2, 2)
max_hr = st.number_input("Max Heart Rate Achieved", 60, 220, 109)
exang = st.selectbox("Exercise Angina (1=Yes, 0=No)", [0,1])
st_dep = st.number_input("ST Depression", 0.0, 6.0, 2.4)
slope = st.number_input("Slope of ST Segment (0-2)", 0, 2, 2)
num_vessels = st.number_input("Number of Major Vessels Fluro (0-3)", 0, 3, 3)
thal = st.number_input("Thallium (1-3)", 1, 3, 3)

if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, cp, bp, chol, fbs, ekg, max_hr, exang, st_dep, slope, num_vessels, thal]],
                              columns=['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
                                       'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
                                       'Slope of ST', 'Number of vessels fluro', 'Thallium'])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected!")
    else:
        st.success("✅ No Heart Disease Detected!")
