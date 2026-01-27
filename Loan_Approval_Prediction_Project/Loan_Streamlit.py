import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# page configration
st.set_page_config(page_title="Loan_Approval_Prediction_System")

st.title("🏦 Loan Approval Prediction System")
st.subheader("Predict High-Risk Applicants Using Financial & Demographic Data")

# Load Model & Metrics 
trained_model = joblib.load("loan_approval_model,joblib")
metrics_df = pd.read_csv("trained_models_metrics.csv")

#Tabs
tab1, tab2 = st.tabs(["Loan Prediction", "Model Comparison"])
#Tab1 : Model Prediction
with tab1:
    #csv upload
    file1 = st.file_uploader("upload your csv file:", type=["csv"])
    if file1:
        df = pd.read_csv(file1)
        st.subheader("data preview")
        st.dataframe(df)

        st.subheader("Summary Stats") 
        st.write(df.describe())
    # file 2 to upload data after feature Extraction, user input would based on this dataset.
    file2 = st.file_uploader("uplaod after feature_engineering csv file:", type="csv")
    if file2:
        df = pd.read_csv(file2)
        st.subheader("data preview")
        st.dataframe(df)

    # sidebar for user inputs
    st.sidebar.header(" 🧾Loan Applicants Details")

    no_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=8, value=0)
    income_annum = st.sidebar.slider("Annual Income", min_value=200000, max_value=9900000, value=5000000)
    loan_amount = st.sidebar.slider("Loan Amount",min_value=300000, max_value=39500000, value=30000000)
    loan_term = st.sidebar.slider("Loan Term (Years)", min_value=2, max_value=20, value=5)
    cibil_score = st.sidebar.slider("CIBIL Score", min_value=300, max_value=900, value=700)
    loan_to_income_ratio = st.sidebar.slider("Loan to Income Ratio", min_value=0.0, max_value= 5.0, value=2.0)
    total_assets_value = st.sidebar.slider("Total Assets Value", min_value=0, max_value=80000000,  value=20000000)
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

#Convert Inputs → Model Format
#Convert dictionary → DataFrame
    input_data = pd.DataFrame([{
        "no_of_dependents": int(no_of_dependents),
        "income_annum": int(income_annum),
        "loan_amount": int(loan_amount),
        "loan_term": int(loan_term),
        "cibil_score": int(cibil_score),
        "loan_to_income_ratio": float(loan_to_income_ratio),
        "total_assets_value": int(total_assets_value),
        "education": education,
        "self_employed": self_employed,
    }])
    st.subheader("loan Applicant Summary")
    st.write(input_data)
    #prediction button 
    if st.button("Predict Loan Approval"):
        prediction = trained_model.predict(input_data)
        probability = trained_model.predict_proba(input_data) [0][1]
        st.write(f"Loan Approval Probability: {probability:.2f}")
        # result display
        if probability <= 0.6:
            st.warning("❌ Loan Rejected!")
        else:
            st.success("✅ Loan Approved!")

    st.divider()
    st.subheader("Model Transparency")  
    with st.expander("ℹ️ Model Information"):
        st.write("""
        - **Model** : XGBoost Classifier 
        - **Problem** : Loan Approval Classification  
        - **Metric Used**: ROC-AUC  
        - **Goal**: Identify high-risk Applicants
        """)

#TAB 2: MODEL COMPARISON
with tab2:
    st.subheader("📊 Models Performance Comparison")

    st.write("Precision, Recall, and F1-score comparison across models:")
    st.dataframe(metrics_df)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df.set_index("Model").plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1) #Fix Y-axis range (All metrics are between 0 and 1)
    ax.set_title("Model Metrics Comparison")

    st.pyplot(fig)





                                    



