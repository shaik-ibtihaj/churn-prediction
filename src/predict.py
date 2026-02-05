import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict_churn(input_data: dict):
    df = pd.DataFrame([input_data])
    X_processed = preprocessor.transform(df)
    probability = model.predict_proba(X_processed)[0][1]
    return probability


if __name__ == "__main__":
    sample = {
        "tenure": 3,
        "MonthlyCharges": 95.0,
        "TotalCharges": 300.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic"
    }

    prob = predict_churn(sample)
    print(f"Churn probability: {prob:.2f}")