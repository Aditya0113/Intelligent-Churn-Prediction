
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# Load Model
try:
    model = joblib.load('models/churn_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}. Please run 'src/model_train.py' first.")
    st.stop()

# Title and Intro
st.title("📡 Telco Customer Churn Prediction")
st.markdown("""
Predict customer churn probability using the Telco Customer Churn dataset model.
Adjust key factors to see how they impact the risk.
""")

# Sidebar - User Input (Simplified mostly to key features)
st.sidebar.header("Customer Profile")

# Key Inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, monthly_charges * tenure)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Additional inputs (Defaulted for simplicity, but exposed if needed)
with st.sidebar.expander("Additional Services"):
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# Construct DataFrame with ALL expected columns (Defaults for missing ones)
# Note: The model pipeline expects a specific set of columns. We must provide them.
# We will use reasonable defaults for the UI-hidden fields.

input_data = {
    'gender': 'Male', # Default
    'SeniorCitizen': 0, # Default
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': tenure,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': tech_support,
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert to DataFrame
df_input = pd.DataFrame(input_data, index=[0])

# Feature Engineering (Must match model_train.py logic)
# 1. Avg Usage
df_input['Avg_Monthly_Usage'] = np.where(df_input['tenure'] > 0, df_input['TotalCharges'] / df_input['tenure'], 0)

# 2. Tenure Group
df_input['Tenure_Group'] = pd.cut(
    df_input['tenure'], 
    bins=[-1, 12, 24, 48, 999], 
    labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years']
).astype(str) # Ensure string type for category matching if needed, or rely on pipeline logic
# Pandas 'cut' returns Categorical, but single row might be tricky. 
# Let's map it explicitly to be safe and avoid categorical errors on single row
if tenure <= 12:
    df_input['Tenure_Group'] = '0-1 Year'
elif tenure <= 24:
    df_input['Tenure_Group'] = '1-2 Years'
elif tenure <= 48:
    df_input['Tenure_Group'] = '2-4 Years'
else:
    df_input['Tenure_Group'] = '4+ Years'

# 3. Payment Type Simplified
payment_desc = str(payment_method).lower()
df_input['Payment_Type_Simplified'] = 'Automatic' if 'automatic' in payment_desc else 'Manual'


# Prediction Logic
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Customer Data Review")
    st.dataframe(df_input)

    if st.button("Predict Churn Risk"):
        try:
            # Predict using the Pipeline
            prediction = model.predict(df_input)[0]
            prediction_proba = model.predict_proba(df_input)[0][1]
            
            st.subheader("Result")
            if prediction == 1:
                st.error(f"⚠️ High Churn Risk ({prediction_proba:.2%})")
                st.write("Suggestion: Consider offering a long-term contract or discount.")
            else:
                st.success(f"✅ Low Churn Risk ({prediction_proba:.2%})")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.write("Ensure all features match the training data keys.")

with col2:
    st.info("ℹ️ **Model Info**")
    st.write("Model: Random Forest Pipeline")
    st.write("Trained on: WA Telco Customer Churn Data")
    st.write("Key Drivers: Tenure, Contract, Monthly Charges")
