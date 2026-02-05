# src/preprocessing.py

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Load data with robust path handling
# --------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, filepath)
    df = pd.read_csv(full_path)
    return df


# --------------------------------------------------
# Clean data (STEP 3 FIX INCLUDED)
# --------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning:
    - Drop customerID (identifier, not a feature)
    - Convert TotalCharges to numeric
    - Handle missing values
    - Encode target variable
    """

    # 🔥 STEP 3: Drop identifier column
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values with median
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


# --------------------------------------------------
# Build preprocessing pipeline
# --------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create preprocessing pipeline for numerical and categorical features
    """

    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# --------------------------------------------------
# Full preprocessing + train-test split
# --------------------------------------------------
def preprocess_and_split(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    End-to-end preprocessing:
    - Load data
    - Clean data
    - Split train/test
    - Fit preprocessing pipeline
    - Save preprocessor
    """

    # Load & clean data
    df = load_data(filepath)
    df = clean_data(df)

    # Split features & target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Build & apply preprocessing pipeline
    preprocessor = build_preprocessor(X)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessor for reuse (used by Streamlit)
    joblib.dump(preprocessor, "preprocessor.pkl")

    return X_train_processed, X_test_processed, y_train, y_test


# --------------------------------------------------
# Local test run
# --------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_and_split(
        filepath="../data/Telco_Customer_Churn.csv"  # or ../data/churn.csv
    )

    print("✅ Preprocessing completed successfully.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)