# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

ALL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]
# -----------------------------
# Load model & preprocessor
# -----------------------------
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write(
    "Predict customer churn risk and understand the key drivers behind the prediction."
)

# -----------------------------
# Sidebar: User Inputs
# -----------------------------
st.sidebar.header("Customer Information")

def user_input_features():
    st.sidebar.header("Customer Information")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )

    internet = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

    online_security = st.sidebar.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.sidebar.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.sidebar.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.sidebar.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.sidebar.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

    contract = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

    payment = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.sidebar.slider("Monthly Charges", 20.0, 120.0, 70.0)
    total_charges = st.sidebar.slider("Total Charges", 0.0, 9000.0, 1500.0)

    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    return pd.DataFrame([data])


input_df = user_input_features()

# -----------------------------
# Prediction
# -----------------------------
X_processed = preprocessor.transform(input_df)
churn_proba = model.predict_proba(X_processed)[0][1]

# Risk labeling
if churn_proba < 0.3:
    risk = "Low"
    color = "green"
elif churn_proba < 0.6:
    risk = "Medium"
    color = "orange"
else:
    risk = "High"
    color = "red"

st.subheader("🔮 Prediction Result")
st.markdown(
    f"""
    **Churn Probability:** `{churn_proba:.2f}`  
    **Risk Level:** <span style="color:{color}; font-weight:bold;">{risk}</span>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# SHAP Explainability
# -----------------------------
st.subheader("🧠 Prediction Explanation (SHAP)")

explainer = shap.Explainer(model, X_processed)
shap_values = explainer(X_processed)

fig, ax = plt.subplots()
shap.waterfall_plot(shap_values[0], show=False)
st.pyplot(fig)

# -----------------------------
# Business Insight
# -----------------------------
st.subheader("💼 Business Insight")

if risk == "High":
    st.write(
        "⚠️ This customer is at **high risk of churn**. "
        "Consider offering discounts, contract upgrades, or personalized retention offers."
    )
elif risk == "Medium":
    st.write(
        "⚠️ This customer shows **moderate churn risk**. "
        "Monitor closely and engage proactively."
    )
else:
    st.write(
        "✅ This customer has **low churn risk**. "
        "Maintain service quality and satisfaction."
    )