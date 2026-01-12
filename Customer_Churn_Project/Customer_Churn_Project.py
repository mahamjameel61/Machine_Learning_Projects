"""
Customer churn refers to the phenomenon where customers discontinue their relationship or subscription with a company or service provider.
It represents the rate at which customers stop using a company's products or services within a specific period.
"""
# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# load data
churn_data = pd.read_csv("customer_churn_dataset-training-master[1].csv", sep=',')
print(churn_data)

# EDA
print(churn_data.head())
print(churn_data.tail())
print(churn_data.describe())
print(churn_data.shape)
print(churn_data.info())
print(churn_data.isnull())
print(churn_data.isnull().sum())

#Data cleaning and Preprocrssing
# handle missing values
churn_data.fillna({
    "CustomerID" : churn_data["CustomerID"].mean(),
    "Age" : churn_data["Age"].mean(),
    "Gender": churn_data['Gender'].mode()[0],
    "Tenure": churn_data['Tenure'].mean(),
    "Usage Frequency": churn_data['Usage Frequency'].mean(),
    "Support Calls": churn_data['Support Calls'].mean(),
    "Payment Delay": churn_data['Payment Delay'].mean(),
    "Subscription Type": churn_data['Subscription Type'].mode()[0],
    "Contract Length": churn_data['Contract Length'].mode()[0],
    "Total Spend": churn_data['Total Spend'].mean(),
    "Last Interaction": churn_data['Last Interaction'].mean(),
    "Churn": churn_data['Churn'].mode()[0],
}, inplace=True)

print("after filling missing values:", churn_data.isnull().sum())

# visualization
# univariate analysis
sns.histplot(x="Age", bins=8, hue="Gender", kde=True, data=churn_data)
plt.title("customer ages distributon")
plt.show(block = True)

#bivariate analysis
#to check which age group of people have high frequency usage 
sns.scatterplot(x="Age", y="Usage Frequency", hue="Gender", data=churn_data)
plt.title(" ages group with high frequency usage")
plt.show(block = True)

# category wise analysis which gender have high frequency usage
sns.barplot(x="Gender", y="Usage Frequency",palette=["orange", "purple"], data=churn_data)
plt.title("gender with freguency usage")
plt.show(block = True)

#Heatmap to check the correlation in Numerical features/columns
numeric_cd = churn_data.select_dtypes(include=['float64', 'int64'])
numeric_cd.corr()
sns.heatmap(numeric_cd.corr(), annot=True, cmap="magma") # cmap= "coolwarm", "viridis", "magma", "plasma", "Greens", "Blues"
plt.title("customer churn Heatmap")
plt.show()

# column selection
x = churn_data.drop(["CustomerID", "Support Calls", "Contract Length" ,"Churn"], axis=1)
y = churn_data["Churn"] # target column
print("x_features:", x)

# splitting
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state= 42)
print("xtest:", xtest)

#encodeing
numerical_cols = ["Age","Tenure","Usage Frequency","Payment Delay","Total Spend","Last Interaction"]
nominal_encode = ["Gender"]
ordinal_encode = ["Subscription Type"] # for ranking classes
ordinal_categories = [
    ["Basic", "Standard", "Premium"]      # Subscription Type
    ]

# column transformer
clt = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("nominal", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal_encode),
        ("ordinal", OrdinalEncoder(categories=ordinal_categories), ordinal_encode)],
          remainder="passthrough")

#Pipeline
# use a common preprocessing pipeline and attach different classifiers separately to fairly compare models without duplicating code.
# Model_Selection
pipelines = {
    "logistic": Pipeline([
        ("clt", clt),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "random_forest": Pipeline([
        ("clt", clt),
        ("model", RandomForestClassifier(random_state=42))
    ]),

    "gradient_boost": Pipeline([
        ("clt", clt),
        ("model", GradientBoostingClassifier(random_state=42))
    ])
}

# predictions 
def train_and_predict(model, xtrain, ytrain, xtest, clt):
    pipeline = Pipeline(steps=[
        ("clt", clt),
        ("model", model)
    ])
    pipeline.fit(xtrain, ytrain)
    return pipeline.predict(xtest)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

for name, model in models.items():
    preds = train_and_predict(model, xtrain, ytrain, xtest, clt)
    print(f"{name} predictions:\n", preds)

#Model_Evaluation
# Confusion Matrix 
# Classification Report 
for name, pipe in pipelines.items():
    pipe.fit(xtrain, ytrain)
    y_pred = pipe.predict(xtest)
    print(f"\n{name.upper()} confusion_matrix")
    print(confusion_matrix(ytest, y_pred))
    print(f"\n{name.upper()} classification_report")
    print(classification_report(ytest, y_pred))

# Threshold 
"""
Thresholds are used to convert probabilities into class decisions
 and allow to control the trade-off between false positives and false negatives.
"""
pipe = pipelines["gradient_boost"]
pipe.fit(xtrain, ytrain)
y_proba = pipe.predict_proba(xtest)[:, 1]   # churn probability
threshold = 0.4                             # custom threshold
y_pred_thresh = (y_proba >= threshold).astype(int)
print(classification_report(ytest, y_pred_thresh))
fpr, tpr, thresholds = roc_curve(ytest, y_proba)
auc = roc_auc_score(ytest, y_proba)

# Plot ROC_Curve
# ROC_Curve shows how well the model/probabilities separates churn customers from non-churn customers.
plt.plot(fpr, tpr, label=f" GradientBoosting (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--") # random guessing diagonal line from ((0, 0) bottom-left) to ((1, 1) → top-right).
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn")
plt.legend()
plt.show()   

# Deployable
import joblib
#select final model
final_model = pipelines["gradient_boost"] # training model
final_model.fit(xtrain, ytrain)
# # save model
joblib.dump(final_model, "customer_churn_model.joblib")
#Load the model for prediction / deployment
loaded_model = joblib.load("customer_churn_model.joblib")
#Predict on new data
new_ypredict = loaded_model.predict(xtest)
print("new data predictions by joblib:", new_ypredict)







