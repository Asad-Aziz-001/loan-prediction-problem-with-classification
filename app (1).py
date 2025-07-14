import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Loan Approval Prediction App", layout="centered")
st.title("🏦 Loan Approval Prediction App")

# --- Load model with error handling ---
model_path = 'loan_approval_model.pkl'

if not os.path.exists(model_path):
    st.error("❌ Model file 'loan_approval_model.pkl' not found!")
    st.markdown("""
        Please make sure:
        - You uploaded `loan_approval_model.pkl` in the **same folder** as `app.py`
        - You **committed and pushed** it to your GitHub repo
        - Then **re-deploy** this app on [Streamlit Cloud](https://share.streamlit.io/)
    """)
    st.stop()

# Load the model
model = joblib.load(model_path)

# --- Input fields ---
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

# --- Encoding function ---
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

# --- Prediction ---
if st.button("Predict"):
    features = np.array(encode_input()).reshape(1, -1)
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")

