# src/explain.py

import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_data, clean_data, build_preprocessor


def load_model_and_data():
    """
    Load trained model, preprocessor, and cleaned data
    """
    model = joblib.load("churn_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")

    df = load_data("../data/churn.csv")
    df = clean_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return model, preprocessor, X, y


def generate_shap_explanations():
    """
    Generate SHAP global and local explanations
    """
    model, preprocessor, X, y = load_model_and_data()

    # Transform data
    X_processed = preprocessor.transform(X)

    # Get feature names after OneHotEncoding
    feature_names = preprocessor.get_feature_names_out()

    X_processed_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        columns=feature_names
    )

    # SHAP Explainer for tree-based models
    explainer = shap.Explainer(model, X_processed_df)
    shap_values = explainer(X_processed_df)

    # ---------------------------
    # Global Explanation
    # ---------------------------
    print("📊 Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_processed_df, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

    # ---------------------------
    # Feature Importance (Bar)
    # ---------------------------
    shap.summary_plot(
        shap_values,
        X_processed_df,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png")
    plt.close()

    print("✅ SHAP plots saved successfully.")


if __name__ == "__main__":
    generate_shap_explanations()