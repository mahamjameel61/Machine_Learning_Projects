#  Customer Churn Prediction & Action Recommendation System

## Project Overview
This project is an **end-to-end Machine Learning application** that predicts whether a customer is likely to churn and provides **actionable retention recommendations**.  
The solution is deployed as an **interactive Streamlit web app**, allowing business users to upload data, analyze customers, and make decisions.

### Problem Statement
Customer churn is a major challenge for subscription-based businesses.  
Identifying **high-risk customers early** helps companies reduce revenue loss and improve customer lifetime value.

### Solution & Model
I developed an end-to-end machine learning system that predicts **customer churn probability**, classifies customers into **Low / Medium / High risk segments**, and recommends **retention actions** for high-risk users while maintaining **model transparency**.

To optimize performance, the modeling process evolved iteratively:
- Started with **Logistic Regression** as a baseline
- Improved performance using **Random Forest Classifier**
- Achieved best results with a **Gradient Boosting Classifier**

**Final Model:** Gradient Boosting Classifier  
**Accuracy:** ~92%  
**Outputs:** Binary churn prediction + probability score

### Tech Stack
Python, Pandas & NumPy, Matplotlib, Scikit-learn, Joblib, Streamlit

---
## 🖥 Application Features
🔹 CSV Upload & Data Preview

🔹 Interactive Customer inputs (Sidebar) for real-time predictions

🔹  Displays Churn Prediction & Probability

🔹 Action Recommendations
   For **high-risk customers**, the app suggests:
   - Immediate retention call  
   - 15–20% discount offers  
   - Loyalty benefits

🔹 Model Transparency

---
## 📈 Business Impact
- Helps businesses **reduce churn**
- Enables **data-driven decision making**
- Bridges the gap between **ML models and real business actions**.



