# src/train.py

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# Import preprocessing pipeline
from preprocessing import preprocess_and_split


# --------------------------------------------------
# MODEL TRAINING FUNCTIONS
# --------------------------------------------------

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression with class_weight
    """
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest with class_weight
    """
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost with scale_pos_weight
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=3,  # approx imbalance ratio
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model using ROC-AUC, classification report,
    and confusion matrix
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n================ {model_name} =================")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

    print("🔄 Loading and preprocessing data...")

    # Get preprocessed train-test data
    X_train, X_test, y_train, y_test = preprocess_and_split(
        filepath="../data/Telco_Cusomer_Churn.csv"
    )

    # -------------------------
    # 1️⃣ Logistic Regression
    # -------------------------
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # -------------------------
    # 2️⃣ Random Forest
    # -------------------------
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # -------------------------
    # 3️⃣ XGBoost (Final Model)
    # -------------------------
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Save final model
    joblib.dump(xgb_model, "churn_model.pkl")

    print("\n✅ Training completed successfully.")
    print("📦 Final model saved as churn_model.pkl")
