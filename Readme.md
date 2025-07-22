## 🚀 Lead Conversion Prediction System
This project builds a scalable, explainable ML system that predicts whether a lead will convert, enabling sales/marketing teams to focus on high-potential leads.

📌 Project Goal
To develop a robust ML pipeline to:

🎯 Predict lead conversion (binary classification: 0 = Not Converted, 1 = Converted)

📈 Maximize conversion rate and sales ROI

🔍 Provide explainable predictions via SHAP

📊 Monitor model/data drift with Evidently

📦 Deploy the model with Flask + PostgreSQL

## 🗃️ Dataset Summary
Feature	Description
Converted	Target variable (0 or 1)
Do Not Email, Do Not Call	Binary features (Yes/No)
TotalVisits, Time on Website	Numerical engagement metrics
Source, Specialization	Categorical attributes
Asymmetrique_* features	Ordinal behavioral scoring

## 🏗️ Architecture Overview
java
Copy
Edit
CSV/DB → Preprocessing → Model Training → MLflow Logging/Registry
         ↓                                     ↓
    SMOTE Balancing                     Model Deployment (Flask)
         ↓                                     ↓
     Evaluation (ROC, F1)       Predictions → PostgreSQL → Results
         ↓                                     ↓
    SHAP Explainability             Evidently Drift Monitoring
## 🧰 Libraries Used
txt
Copy
Edit
Python 3.10.11
pandas, numpy, matplotlib, seaborn
scikit-learn, imblearn, xgboost
mlflow, shap, evidently
sqlalchemy, psycopg2-binary, flask
joblib, warnings, re, os
⚙️ Setup Instructions
🔧 1. Create and Activate Environment
bash
Copy
Edit
conda create -n leadenv python=3.10.11
conda activate leadenv
## 2. Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt
## 3. Setup .env (for PostgreSQL)
ini
Copy
Edit
DB_USER=postgres
DB_PASSWORD=vinay
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mydb1
##  🧪 Model Training Pipeline
## 📂 Data Preparation
python
Copy
Edit
df = pd.read_csv("Lead Scoring.csv")
df.replace(["Select", ""], np.nan, inplace=True)
🛠️ Preprocessing
Ordinal encoding: asymmetrique_activity_index, asymmetrique_profile_index

Binary encoding for Yes/No columns

OneHot encoding for categoricals

Scaling for numerical values

Pipeline saved as preprocess.pkl

🧪 Class Balancing with SMOTE
python
Copy
Edit
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_preprocessed, y)
##  MLflow Integration
Automatically tracks metrics, parameters, SHAP plots

Registers best model (based on ROC-AUC)

Automatically transitions to Production stage

Sample log:

python
Copy
Edit
mlflow.log_metric("roc_auc", 0.91)
mlflow.sklearn.log_model(best_model, "model")
🤖 Model Deployment (Flask)
## Files
app.py: Flask server with /predict

upload.html: Upload CSV

results.html: Display predictions in tabular format

Saves results to PostgreSQL (lead_scoring table)

## Auto-load Model from MLflow
python
Copy
Edit
from mlflow.tracking import MlflowClient

def get_latest_production_model_name(stage="Production"):
    ...
    return model_name

model_name = get_latest_production_model_name()
model_uri = f"models:/{model_name}/Production"
pipeline = mlflow.pyfunc.load_model(model_uri)
## Drift Monitoring (Evidently)
Prepares train, val, and test sets

Applies preprocessing

Uses Evidently to generate and log drift reports to MLflow

python
Copy
Edit
from evidently.report import Report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_test)
report.save_html("drift_report.html")
mlflow.log_artifact("drift_report.html")
🧪 Sample Prediction Flow
bash
Copy
Edit
## Run Flask server
python app.py

## Go to: http://localhost:5001
 Upload CSV file with required columns
 Get predictions + PostgreSQL save
 📄 File Checklist
File	Purpose
app.py	Flask prediction server
upload.html	CSV upload form
results.html	Prediction table output
requirements.txt	Package list
.env	Environment variables
preprocess.pkl	Saved preprocessing pipeline

## ✅ Best Practices Followed
🧼 Clean column names and handle missing values

💡 Explainable ML with SHAP

⚖️ SMOTE to handle class imbalance

📦 MLflow for end-to-end experiment tracking

🧠 Drift tracking with Evidently

🚀 Fully automated production deployment via Flask
