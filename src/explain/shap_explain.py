"""
Create a SHAP summary plot for the saved XGBoost model.
Usage:
    python -m src.explain.shap_explain --data data/anti_poaching.csv --model models/xgb_model.joblib --preproc models/preprocessor.joblib --out docs/shap_summary.png
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.data.load_data import load_data
from src.preprocess import ALL_FEATURES

def load_artifacts(model_path, preproc_path):
    model = joblib.load(model_path)
    preproc = joblib.load(preproc_path)
    return model, preproc

def get_feature_names(preproc):
    try:
        from src.preprocess import NUMERIC, CATEGORICAL
        num_names = NUMERIC
        ohe = preproc.named_transformers_["cat"].named_steps.get("encoder", None) or \
              preproc.named_transformers_["cat"].named_steps.get("ohe", None)
        if ohe is not None:
            cat_names = list(ohe.get_feature_names_out(CATEGORICAL))
        else:
            cat_names = []
        return num_names + cat_names
    except Exception:
        return None

def explain(data_path, model_path, preproc_path, out_path, sample_size=1000):
    df = load_data(data_path)
    X_df = df[ALL_FEATURES]

    model, preproc = load_artifacts(model_path, preproc_path)

    # sample for speed
    if sample_size and sample_size < len(X_df):
        X_df = X_df.sample(sample_size, random_state=42).reset_index(drop=True)

    X = preproc.transform(X_df)

    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)
    # shap_values shape: (n_samples, n_features) for binary classification
    shap_values = explainer.shap_values(X) if hasattr(explainer, "shap_values") else explainer(X)

    # build feature names if possible
    feat_names = get_feature_names(preproc)
    if feat_names is None or len(feat_names) != X.shape[1]:
        feat_names = [f"f{i}" for i in range(X.shape[1])]

    # summary plot (save to file)
    plt.figure(figsize=(8,6))
    # Use shap.summary_plot which writes to plt.gcf() when show=False in older versions
    try:
        shap.summary_plot(shap_values, X, feature_names=feat_names, show=False)
    except TypeError:
        shap.summary_plot(shap_values, X, feat_names, show=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/anti_poaching.csv")
    p.add_argument("--model", default="models/xgb_model.joblib")
    p.add_argument("--preproc", default="models/preprocessor.joblib")
    p.add_argument("--out", default="docs/shap_summary.png")
    args = p.parse_args()
    explain(args.data, args.model, args.preproc, args.out)
