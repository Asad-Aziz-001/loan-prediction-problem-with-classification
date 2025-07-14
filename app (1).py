import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('loan_approval_model.pkl')

st.title("🏦 Loan Approval Prediction App")

# Input fields
Gender = st.selectbox("Gender", ['Male', 'Female'])
Married = st.selectbox("Married", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Term (months)", min_value=0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

# Encode inputs
def encode_input():
    gender = 1 if Gender == 'Male' else 0
    married = 1 if Married == 'Yes' else 0
    dependents = {'0': 0, '1': 1, '2': 2, '3+': 3}[Dependents]
    education = 0 if Education == 'Graduate' else 1
    self_employed = 1 if Self_Employed == 'Yes' else 0
    area = {'Urban': 2, 'Rural': 1, 'Semiurban': 0}[Property_Area]
    return [gender, married, dependents, education, self_employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, area]

if st.button("Predict"):
    features = np.array(encode_input()).reshape(1, -1)
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")