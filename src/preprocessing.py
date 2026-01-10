
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(filepath: str) -> pd.DataFrame:
    
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
   
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
   
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def preprocess_and_split(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    

    # Load & clean data
    df = load_data(filepath)
    df = clean_data(df)

    # Split features & target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Build & apply preprocessing pipeline
    preprocessor = build_preprocessor(X)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessor for reuse
    joblib.dump(preprocessor, "preprocessor.pkl")

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    # For quick testing
    X_train, X_test, y_train, y_test = preprocess_and_split(
        filepath="../data/churn.csv"
    )

    print("Preprocessing completed successfully.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
