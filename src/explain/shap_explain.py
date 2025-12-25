"""
Create a SHAP summary plot for the trained XGBoost model.

Run from project root:
    python -m src.explain.shap_explain
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.data.load_data import load_data
from src.preprocess import (
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def load_artifacts(model_path, preproc_path):
    model = joblib.load(model_path)
    preproc = joblib.load(preproc_path)
    return model, preproc


def get_feature_names(preprocessor):
    """
    Extract feature names from ColumnTransformer:
    - numeric features (unchanged)
    - one-hot encoded categorical features
    """
    feature_names = []

    # numeric
    feature_names.extend(NUMERIC_FEATURES)

    # categorical (one-hot)
    cat_pipeline = preprocessor.named_transformers_["cat"]
    ohe = cat_pipeline.named_steps["encoder"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_names.tolist())

    return feature_names


# ----------------------------------------------------------------------
# SHAP explanation
# ----------------------------------------------------------------------
def explain(data_path, model_path, preproc_path, out_path, sample_size=1000):

    # Load data
    df = load_data(data_path)
    X_df = df[ALL_FEATURES]

    # Load model + preprocessor
    model, preproc = load_artifacts(model_path, preproc_path)

    # Sample for speed (important for SHAP)
    if sample_size and sample_size < len(X_df):
        X_df = X_df.sample(sample_size, random_state=42).reset_index(drop=True)

    # Transform features
    X = preproc.transform(X_df)

    # Build SHAP explainer (tree model)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Feature names
    feature_names = get_feature_names(preproc)

    # Safety check
    if len(feature_names) != X.shape[1]:
        print("⚠️ Feature name mismatch — using generic names")
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    # Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"✅ Saved SHAP summary plot to {out_path}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/anti_poaching_with_env.csv",
        help="Dataset with environmental features"
    )
    parser.add_argument(
        "--model",
        default="models/xgb_env.joblib",
        help="Trained XGBoost model"
    )
    parser.add_argument(
        "--preproc",
        default="models/preprocessor_env.joblib",
        help="Trained preprocessor"
    )
    parser.add_argument(
        "--out",
        default="docs/shap_summary.png",
        help="Output SHAP plot"
    )

    args = parser.parse_args()

    explain(
        data_path=args.data,
        model_path=args.model,
        preproc_path=args.preproc,
        out_path=args.out
    )
