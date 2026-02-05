📊 Customer Churn Prediction System (End-to-End ML Project)

An end-to-end machine learning system that predicts customer churn, explains why customers are likely to leave, and presents results through an interactive Streamlit web application.

This project is designed to mirror real-world ML workflows including data preprocessing, class imbalance handling, model explainability, and deployment.

🎯 Project Objective

Customer churn directly impacts business revenue. The goal of this project is to:

Predict whether a customer is likely to churn

Identify key factors driving churn

Enable proactive, targeted retention strategies

🧩 Problem Statement

Build a machine learning system that predicts customer churn in advance, explains the key drivers behind churn, and makes predictions accessible to non-technical users.

📂 Dataset

Dataset: Telco Customer Churn (Kaggle)

Target Variable: Churn (Yes → 1, No → 0)

Key Feature Groups

Customer demographics: gender, senior citizen, partner, dependents

Service usage: internet service, streaming, tech support

Contract & billing: contract type, payment method, paperless billing

Behavioral metrics: tenure, monthly charges, total charges

Note: customerID is dropped during preprocessing as it is an identifier, not a predictive feature.

🏗️ Project Architecture

Data Ingestion
      ↓
Data Cleaning & EDA
      ↓
Feature Engineering
      ↓
Model Training (Imbalance Handling)
      ↓
Model Evaluation
      ↓
Explainability (SHAP)
      ↓
Deployment (Streamlit)

🔍 Exploratory Data Analysis (EDA)

Key insights discovered during EDA:

The dataset is imbalanced (~25–30% churn)

Short-tenure customers churn more frequently

Higher monthly charges increase churn risk

Month-to-month contracts have the highest churn rate

Long-term contracts significantly reduce churn

These insights guided feature engineering and model selection.

⚙️ Data Preprocessing

Dropped identifier column (customerID)

Converted TotalCharges to numeric and handled missing values

Encoded categorical features using OneHotEncoding

Scaled numerical features using StandardScaler

Used ColumnTransformer + Pipeline to prevent data leakage

⚖️ Class Imbalance Handling

Customer churn is an imbalanced classification problem.

Techniques used:

Class-weighted learning (class_weight='balanced')

scale_pos_weight in XGBoost

Evaluation focused on:

ROC-AUC

Recall for churn class

Accuracy was intentionally deprioritized as it is misleading for imbalanced datasets.

🤖 Model Training

Models trained and compared:

Logistic Regression (baseline)

Random Forest

XGBoost (final model)

Final Model Performance

ROC-AUC: ~0.86

Strong recall for churn customers

XGBoost was selected due to superior performance on tabular data and imbalanced classes.

🧠 Model Explainability (SHAP)

To ensure transparency and business trust:

Used SHAP (SHapley Additive exPlanations)

Generated global and local explanations

Key SHAP Insights

Contract type is the strongest churn predictor

Short tenure increases churn probability

Higher monthly charges raise churn risk

SHAP explanations are also integrated into the Streamlit app.

🌐 Deployment (Streamlit App)

The project is deployed as an interactive Streamlit web application.

App Features

Input customer details via UI

Predict churn probability

Risk labeling: Low / Medium / High

SHAP-based explanation for individual predictions

📁 Project Structure

churn-prediction/
│
├── data/
│   └── churn.csv
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── explain.py
│   ├── predict.py
│   ├── churn_model.pkl
│   └── preprocessor.pkl
├── app.py
├── requirements.txt
└── README.md

▶️ How to Run the Project

1️⃣ Install dependencies

pip install -r requirements.txt

2️⃣ Train the model

python src/train.py

3️⃣ Run the Streamlit app

streamlit run app.py

💼 Business Impact

Example business translation:

Identify top 20% high-risk customers

Targeted retention campaign saves ~30%

Significant monthly revenue preservation

The system enables proactive retention instead of reactive loss management.

🧠 Interview-Ready Highlights

Built an end-to-end churn prediction system using XGBoost

Achieved ROC-AUC of 0.86 on imbalanced data

Applied class-weighted learning to improve churn recall

Used SHAP for transparent model explainability

Deployed the solution using Streamlit for real-world usability

🚀 Future Improvements

Hyperparameter tuning with cross-validation

Time-based churn prediction

API deployment using FastAPI

Monitoring model drift in production

👤 Author

Shaik Ibtihajulla ShaEnd-to-End Machine Learning Project