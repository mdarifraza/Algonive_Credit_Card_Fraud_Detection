import streamlit as st
import numpy as np
import joblib
from xgboost import XGBClassifier  # pyright: ignore[reportMissingImports]

# -------------------------
# Load model & scaler
# -------------------------
model = XGBClassifier()
model.load_model("fraud_model.json")

scaler = joblib.load("scaler.pkl")

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether it is **Fraud** or **Legit**")

st.markdown("---")

# -------------------------
# Input fields
# -------------------------
st.subheader("Transaction Features")

time = st.number_input("Time", min_value=0.0)
amount = st.number_input("Amount", min_value=0.0)

V_features = []
for i in range(1, 29):
    V_features.append(
        st.number_input(f"V{i}", value=0.0)
    )

# -------------------------
# Prediction
# -------------------------
if st.button("🔍 Predict"):
    
    input_data = np.array(
        [[time] + V_features + [amount]]
    )

    # scale
    input_scaled = scaler.transform(input_data)

    # predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legit Transaction")

    st.write(f"**Fraud Probability:** {probability:.4f}")
