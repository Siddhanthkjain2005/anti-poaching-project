"""
Optuna hyperparameter tuning for XGBoost on the anti-poaching dataset.

Usage:
    python -m src.train.tune_optuna --data data/anti_poaching.csv --trials 40 --out models/xgb_optuna_best.joblib
"""

import os
import argparse
import joblib
import numpy as np
from functools import partial
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import optuna

from src.data.load_data import load_data
from src.preprocess import build_preprocessor, ALL_FEATURES

def objective(trial, X, y):
    # parameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    # use a modest classifier for CV
    clf = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        **params
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # note: cross_val_score uses the estimator's fit/score; scoring='roc_auc'
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())

def tune_and_save(data_path, trials, model_out, preproc_out):
    df = load_data(data_path)
    X_df = df[ALL_FEATURES]
    y = df["poaching_occurred"].astype(int)

    # train-only preprocessor fit (avoid leaking val/test)
    # we'll fit on entire X_df before final retrain but for CV it's okay to fit transform once for speed
    preprocessor = build_preprocessor()
    preprocessor.fit(X_df)
    X = preprocessor.transform(X_df)

    # study
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    func = partial(objective, X=X, y=y)
    study.optimize(func, n_trials=trials, show_progress_bar=True)

    print("Best trial:")
    print(study.best_trial.params)
    best_params = study.best_trial.params

    # Retrain final model on full data using best params
    # add some safe defaults
    model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        **best_params
    )
    # Fit on full dataset (no early stopping here)
    model.fit(X, y)

    os.makedirs(os.path.dirname(preproc_out), exist_ok=True)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(preprocessor, preproc_out)
    joblib.dump(model, model_out)
    print(f"Saved preprocessor -> {preproc_out}")
    print(f"Saved tuned model -> {model_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/anti_poaching.csv")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--model_out", default="models/xgb_optuna_best.joblib")
    p.add_argument("--preproc_out", default="models/preprocessor.joblib")
    args = p.parse_args()
    tune_and_save(args.data, args.trials, args.model_out, args.preproc_out)
