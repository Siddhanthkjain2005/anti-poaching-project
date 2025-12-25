"""
Train an XGBoost classifier on the anti-poaching dataset.

Compatible with XGBoost >= 2.0
Uses early stopping correctly (defined in model init).

Run from project root:
    python -m src.train.train_xgb --data data/anti_poaching.csv
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

from src.data.load_data import load_data
from src.preprocess import build_preprocessor, ALL_FEATURES


def train_xgb(
    data_path: str,
    model_out: str,
    preproc_out: str,
    random_state: int = 42
):
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = load_data(data_path)

    X_df = df[ALL_FEATURES]
    y = df["poaching_occurred"].astype(int)

    # ------------------------------------------------------------------
    # 2. Train / Val / Test split (NO DATA LEAKAGE)
    # ------------------------------------------------------------------
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    X_train_sub_df, X_val_df, y_train_sub, y_val = train_test_split(
        X_train_df,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=random_state
    )

    # ------------------------------------------------------------------
    # 3. Preprocessing (fit ONLY on training subset)
    # ------------------------------------------------------------------
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_sub_df)

    X_train = preprocessor.transform(X_train_sub_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    # ------------------------------------------------------------------
    # 4. Handle class imbalance
    # ------------------------------------------------------------------
    pos = np.sum(y_train_sub == 1)
    neg = np.sum(y_train_sub == 0)
    scale_pos_weight = neg / max(1, pos)

    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}, Test size: {X_test.shape}")
    print(f"Positive samples: {pos}, Negative samples: {neg}")
    print(f"scale_pos_weight = {scale_pos_weight:.3f}")

    # ------------------------------------------------------------------
    # 5. XGBoost model (early stopping defined HERE)
    # ------------------------------------------------------------------
    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,   # ✅ correct for XGBoost ≥ 2.0
        n_jobs=-1,
        random_state=random_state
    )

    # ------------------------------------------------------------------
    # 6. Train with early stopping
    # ------------------------------------------------------------------
    model.fit(
        X_train,
        y_train_sub,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # ------------------------------------------------------------------
    # 7. Evaluation on test set
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n===== TEST SET RESULTS =====")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    # ------------------------------------------------------------------
    # 8. Save model & preprocessor
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(os.path.dirname(preproc_out), exist_ok=True)

    joblib.dump(model, model_out)
    joblib.dump(preprocessor, preproc_out)

    print("\nSaved files:")
    print(f"Model        -> {model_out}")
    print(f"Preprocessor -> {preproc_out}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/anti_poaching.csv",
        help="Path to dataset CSV"
    )
    parser.add_argument(
        "--model_out",
        default="models/xgb_model.joblib",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--preproc_out",
        default="models/preprocessor.joblib",
        help="Path to save preprocessor"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    args = parser.parse_args()

    train_xgb(
        data_path=args.data,
        model_out=args.model_out,
        preproc_out=args.preproc_out,
        random_state=args.random_state
    )
