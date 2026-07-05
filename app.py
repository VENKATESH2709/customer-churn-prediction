"""
Customer Churn Prediction - Streamlit Interface
Run locally with: streamlit run app.py
Deploy free at: https://share.streamlit.io (connect this GitHub repo)
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

MODEL_DIR = "models"


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    label_encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
    model_name = joblib.load(f"{MODEL_DIR}/best_model_name.pkl")
    return model, scaler, label_encoders, feature_columns, model_name


model, scaler, label_encoders, feature_columns, model_name = load_artifacts()

st.title("📉 Customer Churn Predictor")
st.caption(f"Model in use: **{model_name}** | Trained on the Telco Customer Churn dataset")

st.write("Enter customer details below to estimate their probability of churning.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([raw_input])

    # Apply the same label encoders used during training
    for col, le in label_encoders.items():
        if col in input_df.columns and col != "Churn":
            input_df[col] = le.transform(input_df[col])

    input_df = input_df[feature_columns]

    if model_name == "Logistic Regression":
        input_final = scaler.transform(input_df)
    else:
        input_final = input_df

    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **churn** — probability: {probability:.1%}")
    else:
        st.success(f"✅ This customer is likely to **stay** — churn probability: {probability:.1%}")

    st.progress(float(probability))
