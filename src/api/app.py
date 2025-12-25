# --- Add these imports near the top of src/api/app.py ---
import numpy as np
import pandas as pd
import shap

from fastapi import HTTPException

from src.preprocess import (
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)

# -------------------------------------------------------------------

# --- After your existing /predict endpoint code, add this new endpoint ---

def _get_transformed_feature_names(preprocessor):
    """
    Returns transformed feature names in the same order
    as PREPROC.transform() output.
    """
    names = []

    # numeric features (same order)
    names.extend(NUMERIC_FEATURES)

    # categorical one-hot features
    cat_pipe = preprocessor.named_transformers_.get("cat")
    if cat_pipe:
        encoder = cat_pipe.named_steps.get("encoder")
        if encoder:
            cat_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            names.extend(cat_names.tolist())

    return names


def _map_shap_to_original(shap_vals_row, transformed_names):
    """
    Aggregate SHAP values back to original features.
    Numeric features map 1:1.
    Categorical one-hot features are summed.
    """
    contributions = {}

    # numeric
    for i, feat in enumerate(NUMERIC_FEATURES):
        contributions[feat] = float(shap_vals_row[i])

    # categorical
    offset = len(NUMERIC_FEATURES)
    for idx, name in enumerate(transformed_names[offset:], start=offset):
        val = float(shap_vals_row[idx])
        if "__" in name:
            orig = name.split("__")[0]
        else:
            orig = name
        contributions[orig] = contributions.get(orig, 0.0) + val

    return contributions

@app.post("/explain")
def explain(req: PredictRequest):
    """
    Returns SHAP-based explanation for a single prediction.
    """

    if MODEL is None or PREPROC is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build input DataFrame
    payload = req.dict()
    df = pd.DataFrame([payload])
    df = df.reindex(columns=ALL_FEATURES)

    # Transform input
    try:
        X = PREPROC.transform(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    # Build / cache SHAP explainer
    if not hasattr(MODEL, "__shap_explainer__"):
        try:
            MODEL.__shap_explainer__ = shap.TreeExplainer(MODEL)
        except Exception:
            MODEL.__shap_explainer__ = shap.Explainer(MODEL)

    explainer = MODEL.__shap_explainer__

    # Compute SHAP values
    try:
        shap_values = explainer.shap_values(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP error: {e}")

    # Binary classifier handling
    if isinstance(shap_values, list):
        # positive class
        shap_arr = np.array(shap_values[1])
        base_value = explainer.expected_value[1]
    else:
        shap_arr = np.array(shap_values)
        base_value = explainer.expected_value

    shap_arr = shap_arr.reshape(1, -1)

    # Feature names
    transformed_names = _get_transformed_feature_names(PREPROC)

    # Aggregate back to original features
    contributions = _map_shap_to_original(shap_arr[0], transformed_names)

    # Sort by absolute contribution
    top_features = sorted(
        [
            {
                "feature": k,
                "contribution": float(v),
                "abs": abs(float(v))
            }
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
