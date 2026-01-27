# Loan Approval Prediction System
## 📌 Project Overview
An end-to-end **Machine Learning Loan Approval Prediction System** that classifies loan applications as **`1` → Loan Approved** or **`0` → Loan Rejected** based on applicant attribute such as financial history and demographic data.  
The trained model is deployed as an **interactive Streamlit web application**, enabling real-time loan approval predictions.

## Dataset & Features
### Demographic Features
- Education  
- Self-employment status  
- Number of dependents  
### Financial Features
- Annual income  
- Loan amount  
- Loan term  
- CIBIL score  
- Residential, Commercial, Luxury, and Bank assets  

## Feature Engineering
### Loan-to-Income Ratio (loan_amount / income_annum)
- Measures financial burden
- Higher ratio → higher rejection risk
### Total Assets Value (residential + commercial + luxury + bank assets)
- Represents overall financial stability
- Improves prediction accuracy

## Exploratory Data Analysis (EDA)
**Visualizations include**:
- Boxplots (CIBIL vs Loan Status)
- Count plots (Approved vs Rejected)
- Approved applicants have higher median CIBIL scores
- Rejected cases often show weak assets.
- Financial stability plays a major role in loan decisions

## Machine Learning Models & Evaluation
**Trained models**:
- Logistic Regression, Random Forest, Gradient Boosting, **XGBoost (Final)**  
**Evaluation metrics**: Precision, Recall, F1-score, ROC-AUC  
All models use **Scikit-learn Pipelines** with proper preprocessing to prevent data leakage.

**Threshold Optimization**
A probability threshold of **0.6** was applied to:
- Identify high-risk (rejected) applicants more conservatively
- Align predictions with real-world banking risk policies

## Cross-Validation
5-fold cross-validation with ROC-AUC scoring that ensures model stability across data splits.

---
## 🚀 Streamlit Web Application
The trained model is deployed using **Streamlit** for real-time predictions.
### Application Features

#### Loan Prediction Tab
- Manual user input via sidebar
- CSV upload for data preview
- Displays:
  - Loan approval probability
  - Approval or rejection decision

#### Model Comparison Tab
- Performance comparison of all trained models
- Interactive bar chart for Precision, Recall, and F1-score

#### Model Transparency
- Model used: XGBoost
- Problem type: Binary Classification
- Evaluation metric: ROC-AUC
- Business goal: Loan Risk identification
---
## 🛠️ Tech Stack
Python| Pandas| NumPy| Seaborn| Matplotlib| Scikit-learn| Joblib |Streamlit.

---
## 📌 Key Takeaways
- CIBIL score alone is not enough.
- Asset ownership significantly improves prediction accuracy.
- Machine Learning enables data-driven lending decisions.
- Streamlit deployment provides a production-ready interface for loan predictions.

## 💼 Business Impact
- **Reduced credit risk**: By identifying high-risk applicants early.
- **Improved loan decision quality**: Considers full financial history, not just CIBIL score.
- **Faster loan processing**: Automated predictions reduce manual review time & Improves customer experience.
- **Scalable system**: Can be integrated into banking systems that supports real-time and batch predictions.









