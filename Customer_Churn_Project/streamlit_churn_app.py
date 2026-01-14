import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

#Page_configration 
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction & Action Recommendation System")
st.markdown("""
This app predicts whether a customer is likely to **churn** and **recommend actions** to retain the customer based on input features.
""")

# csv_file upload 
file = st.file_uploader("Upload your csv file here:", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Data Preview")
    st.dataframe(df)

if file:
    st.subheader("Summary Stats")
    st.write(df.describe())

# calling saved trained model by joblib
trained_model = joblib.load("customer_churn_model.joblib")   #trained model from churn_codefile (relative_path)  

# Sidebar – Customer (for user_inputs)
st.sidebar.header("🧾 Customer Information")

age = st.sidebar.number_input("Age", 18, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
usage_freq = st.sidebar.slider("Usage Frequency", 0, 100, 10)
payment_delay = st.sidebar.slider("Payment Delay (days)", 0, 60, 5)
total_spend = st.sidebar.slider("Total Spend", 0.0, 5000.0, 500.0)
last_interaction = st.sidebar.slider("Last Interaction (days ago)", 0, 365, 30)
subscription = st.sidebar.selectbox("Subscription Type",
                                     ["Basic", "Standard", "Premium"]
                                     )

#Convert Inputs → Model Format
#Convert dictionary → DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Tenure": int(tenure),
    "Usage Frequency": int(usage_freq),
    "Payment Delay": int(payment_delay),
    "Total Spend": float(total_spend),
    "Last Interaction": int(last_interaction),
    "Subscription Type": subscription
}])

st.subheader("📌 Customer Summary")
st.write(input_data)

#Predict Button 
if st.button("🔍 Predict Churn"):
    
    prediction = trained_model.predict(input_data) 
    probability = trained_model.predict_proba(input_data)[0][1] 
    st.write(f"churn Probability: {probability:.2f}") 
    
    #section 5: Result Display 
    st.subheader("📊 Prediction Result")

    if probability >=0.6: 
        st.error("❌ Customer is likely to churn")
        st.markdown("""
                    ### 🛠 Recommended Actions:
                    - 📞 Immediate retention call
                    - 💸 Offer **15–20% discount**
                    - 🎁 Loyalty benefits
                    """)
        
    elif probability >= 0.4:
        st.warning("⚠ Medium churn risk")

    else:
        st.success("✅ Customer is likely to stay")

st.divider()
st.subheader("Model Transparency")
with st.expander("ℹ️ Model Information"):
    st.write("""
    - Dataset: Customer Churn          
    - Features: Age, Gender, Tenure, Usage Frequency, Payment Delay, Total Spend, Last Interaction, Subscription
    - Model: Gradient Boost Classifier           
    - Accuracy: 92% 
             """)        








