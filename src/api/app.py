# src/api/app.py
import joblib
import numpy as np
import pandas as pd
import shap
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictRequest
from src.preprocess import (
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    STATE_RANGES
)

app = FastAPI(title="Anti-Poaching Risk API")

# CORS - allow local frontend and Render etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved artifacts (expect these paths exist)
MODEL_PATH = "models/xgb_model.joblib"
PREPROC_PATH = "models/preprocessor.joblib"

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    MODEL = None
    print(f"Warning: could not load model from {MODEL_PATH}: {e}")

try:
    PREPROC = joblib.load(PREPROC_PATH)
except Exception as e:
    PREPROC = None
    print(f"Warning: could not load preprocessor from {PREPROC_PATH}: {e}")


# ---------------- utilities ----------------
def validate_state_latlon(state: str, lat: float, lon: float):
    """
    Enforce state bounding box validation (same ranges as frontend).
    Raises HTTPException(400) if invalid.
    """
    if not state:
        raise HTTPException(status_code=400, detail="State is required for validation")
    r = STATE_RANGES.get(state)
    if r is None:
        # If we don't have ranges for the state, accept but warn
        return
    if lat < r["lat"][0] or lat > r["lat"][1] or lon < r["lon"][0] or lon > r["lon"][1]:
        raise HTTPException(
            status_code=400,
            detail=f"Latitude/longitude are outside the typical range for {state}. Expected Lat {r['lat'][0]}–{r['lat'][1]}, Lon {r['lon'][0]}–{r['lon'][1]}."
        )


def _get_transformed_feature_names(preprocessor) -> List[str]:
    """
    Returns transformed feature names in the same order as PREPROC.transform() output.
    Works for ColumnTransformer with named 'num' and 'cat' pipelines.
    """
    names: List[str] = []
    # numeric features (same order)
    names.extend(NUMERIC_FEATURES)

    # categorical one-hot features
    try:
        cat_pipe = preprocessor.named_transformers_.get("cat")
        if cat_pipe is None:
            # try to find the categorical transformer by scanning
            for name, trans, cols in preprocessor.transformers_:
                if name == "cat":
                    cat_pipe = trans
                    break
        if cat_pipe is not None:
            # encoder may be nested in pipeline
            encoder = getattr(cat_pipe, "named_steps", {}).get("encoder", None)
            if encoder is None and hasattr(cat_pipe, "get_feature_names_out"):
                # cat_pipe might itself be an encoder
                encoder = cat_pipe
            if encoder is not None:
                # sklearn >=1.0 has get_feature_names_out
                try:
                    cat_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
                except Exception:
                    # older get_feature_names signature
                    cat_names = encoder.get_feature_names(CATEGORICAL_FEATURES)
                names.extend(list(cat_names))
    except Exception:
        # fallback: don't crash; return numeric names only
        pass

    return names


def _map_shap_to_original(shap_vals_row: np.ndarray, transformed_names: List[str]):
    """
    Aggregate SHAP values back to original features.
    Numeric features map 1:1.
    Categorical one-hot features are summed per original categorical column.
    """
    contributions = {}

    # numeric (first len(NUMERIC_FEATURES) columns)
    for i, feat in enumerate(NUMERIC_FEATURES):
        contributions[feat] = float(shap_vals_row[i])

    # one-hot categorical (rest)
    offset = len(NUMERIC_FEATURES)
    for idx, name in enumerate(transformed_names[offset:], start=offset):
        val = float(shap_vals_row[idx])
        # try to extract the original feature name from OHE output
        # common formats: "species_Tiger" or "species__Tiger" etc.
        if "_" in name:
            orig = name.split("_", 1)[0]
        elif "__" in name:
            orig = name.split("__", 1)[0]
        else:
            orig = name
        contributions[orig] = contributions.get(orig, 0.0) + val

    return contributions


# ---------------- endpoints ----------------
@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None or PREPROC is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")

    # validate state / lat / lon (backend enforcement)
    validate_state_latlon(req.state, req.latitude, req.longitude)

    # Build input DataFrame and ensure ALL_FEATURES columns exist (missing -> NaN -> imputed)
    payload = req.dict()
    df = pd.DataFrame([payload])
    df = df.reindex(columns=ALL_FEATURES)  # missing env features become NaN (imputer will fill)

    try:
        X = PREPROC.transform(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    # Predict probability
    try:
        proba = float(MODEL.predict_proba(X)[0, 1])
    except Exception as e:
        # if model is a pipeline or doesn't support predict_proba
        try:
            pred = MODEL.predict(X)[0]
            proba = float(pred)
        except Exception as ee:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e} / {ee}")

    return {
        "poaching_risk_score": proba,
        "poaching_occurred": int(proba >= 0.5),
        "model_version": "xgb_v1"
    }


@app.post("/explain")
def explain(req: PredictRequest):
    """
    Returns SHAP-based explanation for a single prediction.
    """
    if MODEL is None or PREPROC is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")

    # validate state / lat / lon
    validate_state_latlon(req.state, req.latitude, req.longitude)

    payload = req.dict()
    df = pd.DataFrame([payload])
    df = df.reindex(columns=ALL_FEATURES)

    try:
        X = PREPROC.transform(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    # Build / cache SHAP explainer on the model (safe for tree models)
    if not hasattr(MODEL, "__shap_explainer__"):
        try:
            # If MODEL is a sklearn pipeline, try to extract final estimator
            final_estimator = getattr(MODEL, "named_steps", {}).get("model", MODEL)
            MODEL.__shap_explainer__ = shap.TreeExplainer(final_estimator)
        except Exception:
            MODEL.__shap_explainer__ = shap.Explainer(MODEL)

    explainer = MODEL.__shap_explainer__

    try:
        # shap API differences: both .shap_values(X) and explainer(X) are supported; handle both
        try:
            shap_values = explainer.shap_values(X)
        except Exception:
            shap_result = explainer(X)
            shap_values = shap_result.values
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP error: {e}")

    # handle binary-class list output
    if isinstance(shap_values, list) or (hasattr(shap_values, "shape") and shap_values.ndim == 3):
        # shap_values might be [class0, class1]
        if isinstance(shap_values, list):
            # prefer positive class
            arr = np.array(shap_values[1])
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
        else:
            # some versions return (n_samples, n_classes, n_features); pick positive class (1)
            arr = np.array(shap_values)[:, 1, :]
            base_value = explainer.expected_value[1] if hasattr(explainer, "expected_value") else 0.0
    else:
        arr = np.array(shap_values)
        base_value = float(getattr(explainer, "expected_value", 0.0))

    # ensure arr is shape (n_features,) for single sample
    if arr.ndim == 2 and arr.shape[0] == 1:
        shap_row = arr[0]
    elif arr.ndim == 1:
        shap_row = arr
    else:
        # try to flatten
        shap_row = arr.reshape(-1)

    # Feature names on transformed space
    transformed_names = _get_transformed_feature_names(PREPROC)

    contributions = _map_shap_to_original(shap_row, transformed_names)

    top_features = sorted(
        [
            {"feature": k, "contribution": float(v), "abs": abs(float(v))}
            for k, v in contributions.items()
        ],
        key=lambda x: x["abs"],
        reverse=True
    )

    return {
        "base_value": float(base_value),
        "top_features": top_features[:8],
        "all_contributions": contributions
    }
