"""Loan Approval Prediction, where the goal is to classify a loan application as either Approved or Rejected based on the applicant's attributes.
❓Problem Statement (Binary Classification)
Given a set of applicant and loan details, predict the value of the binary target variable, Loan_Status (1 = Approved, 0 = Rejected).
"""
# loan approval classification problem
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import joblib

#1. Load data
loan_data = pd.read_csv('loan_approval_dataset.csv', sep=',')
print(loan_data)

print(loan_data.head())
print(loan_data.describe())
print(loan_data.shape)
print(loan_data.isnull())
print(loan_data.isnull().sum())
print(loan_data.columns)
#remove spaces in col/features (names)
loan_data.columns=loan_data.columns.str.strip() 
print(loan_data.head())

#----Feature Engineering--- 
loan_data['loan_to_income_ratio'] = loan_data['loan_amount'] / loan_data['income_annum']
loan_data['total_assets_value'] = loan_data[['residential_assets_value',
                                       'commercial_assets_value',
                                       'luxury_assets_value',
                                       'bank_asset_value']].sum(axis=1) # Adds all asset values row-wise, axis=1 → sum across columns
print(loan_data.head(3))
""" 
loan_to_income_ratio capture higher risk that if requested loan_ammount is more than income_annum (Higher ratio → higher risk)
Total assets represent overall financial strength, which helps the model assess default risk more effectively
"""
#----EDA----
#Boxplot: cibil score vs loan_status (Approved loans have significantly higher median CIBIL scores.)
sns.boxplot(data=loan_data, x="loan_status", y="cibil_score", palette="coolwarm")
plt.title("cibil_score vs loan_status")
plt.xlabel("loan_status")
plt.ylabel("cibil_score")
plt.show(block=True)

#countplot/barplot to check which category of loan_status is high.
sns.countplot(data=loan_data, x="loan_status", palette=["blue", "red"])
plt.title("approved vs rejected case")
plt.xlabel("loan_status cases")
plt.ylabel("cases_count")
plt.show(block=True)

# ---x,y Split----
x = loan_data.drop(["loan_id", "residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value", "loan_status"], axis=1)
y = loan_data['loan_status'].map({" Rejected": 0, " Approved" : 1}) 
print(x)
print(y)

## saving dataset after Feature Enginnering 
x_df = pd.DataFrame(x)
x_df.to_csv("After_Feature_Engineering_Dataset.csv")
print(x_df)

"""
I explicitly mapped loan status to binary values where Rejected = 0 and Approved = 1 
to ensure clear class interpretation and consistent probability meaning.
"""
# ---Scaling & Encoding---
# separate all numerical, categorical, target column, apply encoding on categorical column
num_col = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'loan_to_income_ratio', 'total_assets_value',] # "residential assest value", 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
onehot_col = ['education','self_employed']

#-----train_test splitting---- 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.7, random_state=42)
print("xtrained is:", xtrain)
print("ytrained is :", ytrain)
#----Column Transformer & Pipeline----
# column transformer
clt = ColumnTransformer(transformers=[
    ('onehotencoding', OneHotEncoder(drop='first', handle_unknown='ignore'), onehot_col),
    ('scaling', StandardScaler(), num_col)
])
print(clt)

# pipeline
# logisticregression 
model1 = Pipeline(steps=[
    ('clt', clt),
    ('algorithm', LogisticRegression())
])

model1.fit(xtrain, ytrain)
log_predicty = model1.predict(xtest)
print("log_predictedy is:", log_predicty)

result1 = classification_report(ytest, log_predicty)
print("logistic regression:", result1)
print('logistic Confusion matrix:', confusion_matrix(ytest, log_predicty))

# randomforest classifier
model2 = Pipeline(steps=[
    ('clt', clt),
    ('algorithm', RandomForestClassifier())
])

model2.fit(xtrain, ytrain)
RF_predicty = model2.predict(xtest)
print("predictedy is:", RF_predicty)

result2 = classification_report(ytest, RF_predicty)
print("randomforest classifier:", result2)
print('randomforest Confusion matrix:', confusion_matrix(ytest, RF_predicty))

# gradientboosting classifier 
model3 = Pipeline(steps=[
    ('clt', clt),
    ('algorithm', GradientBoostingClassifier())
])

model3.fit(xtrain, ytrain)
GRB_predicty = model3.predict(xtest)
print("predictedy is:", GRB_predicty)

result3 = classification_report(ytest, GRB_predicty)
print('gardientboosting classifier:', result3)
print('Gradientboost Confusion matrix:', confusion_matrix(ytest, GRB_predicty))

# xgboost classifiers
model4 = Pipeline(steps=[
    ('clt', clt),
    ('algorithm', XGBClassifier())
])

model4.fit(xtrain, ytrain)
XGB_predicty = model4.predict(xtest)
print("predictedy is:", XGB_predicty)

result4 = classification_report(ytest, XGB_predicty)
print('xbgboost:', result4)
print('Xgboost Confusion matrix:', confusion_matrix(ytest, XGB_predicty))

#---Comparison Between all Algorithms---
"""
After fitting the data to the model, all of the ensemble algorithms return almost similar kind of results.
To visualize the performance of all the algorithms on the same data, we can plot the bar graph between the y_test and y_pred of all the algorithms.
"""
# Bar chart for comparison
from sklearn.metrics import precision_score, recall_score, f1_score

models = {
    "Logistic": log_predicty,
    "Random Forest": RF_predicty,
    "Gradient Boosting": GRB_predicty,
    "XGBoost": XGB_predicty
}

metrics = []

for name, y_pred in models.items():
    metrics.append({
        "Model": name,
        "Precision": precision_score(ytest, y_pred),
        "Recall": recall_score(ytest, y_pred),
        "F1-score": f1_score(ytest, y_pred)
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("trained_models_metrics.csv", index=False)
print(metrics_df)

metrics_df.set_index("Model").plot(kind="bar", figsize=(8, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.show()
 
#---- Model Evaluation----
#Threshold
pipe = model4 # final_model xgboost
pipe.fit(xtrain, ytrain)
y_probability = pipe.predict_proba(xtest)[:,1]      #(':' for rows, 1 for columns)
threshold = 0.6 # to predict maximum rejected cases (as 0 = rejected, below 0.6 predicts rejected_cases)
y_pred_thresh = (y_probability >= threshold).astype(int) #when prob >= threshold it classify as loan approved.
print(classification_report(ytest, y_pred_thresh))
fpr, tpr, threshold = roc_curve(ytest, y_probability)
auc = roc_auc_score(ytest, y_probability)

# Plot ROC_Curve 
#ROC_Curve shows how well the model/probabilities separates Rejected & Approved loan_applicants.
plt.plot(fpr, tpr, label=f" xgboost (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--") # random guessing diagonal line from ((0, 0) bottom-left) to ((1, 1) → top-right).
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Loan_approval")
plt.legend()
plt.show(block = True) 

# ---Cross_validation--- 
"""
I'm using 5-fold cross-validation with ROC-AUC scoring to ensure the model's performance is consistent across different data splits
and not dependent on a single train-test split. 
"""
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model4, x, y, cv=5, scoring= "roc_auc")
print(scores)
print(scores.mean()) #Average performance across all folds

# ---Deployable---
import joblib
final_model = model1 #xgboost
final_model.fit(xtrain, ytrain)
#save model
joblib.dump(final_model, "loan_approval_model,joblib")
#load model
loaded_model = joblib.load("loan_approval_model,joblib")
#predict on new data
new_ypredict = loaded_model.predict(xtest)
print("new data predictions by joblib model:", new_ypredict)








