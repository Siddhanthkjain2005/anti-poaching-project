# --- Add these imports near the top of src/api/app.py ---
import shap
from typing import List
from src.preprocess import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
# -------------------------------------------------------------------

# --- After your existing /predict endpoint code, add this new endpoint ---

def _get_transformed_feature_names(preprocessor):
    """
    Returns the list of transformed feature names (numeric first, then categorical one-hot names)
    Assumes ColumnTransformer with ('num', ..., NUMERIC_FEATURES) and ('cat', ..., CATEGORICAL_FEATURES)
    """
    names = []
    # numeric features first
    names.extend(NUMERIC_FEATURES)

    # categorical encoder
    try:
        cat_transformer = preprocessor.named_transformers_["cat"]
        encoder = cat_transformer.named_steps["encoder"]
        categories = encoder.categories_
        for feat_name, cats in zip(CATEGORICAL_FEATURES, categories):
            for cat_val in cats:
                # use a clear separator
                names.append(f"{feat_name}__{cat_val}")
    except Exception:
        # fallback: create generic names for categories (best-effort)
        for feat_name in CATEGORICAL_FEATURES:
            names.append(f"{feat_name}__unknown")
    return names

def _map_shap_to_original(shap_vals_row, transformed_names):
    """
    Map the shap values for transformed features back to original features:
    - numeric features map 1:1
    - categorical one-hot features are summed per categorical original feature
    Returns dict {original_feature: contribution_value}
    """
    contributions = {}
    # numeric features
    for i, feat in enumerate(NUMERIC_FEATURES):
        contributions[feat] = float(shap_vals_row[i])

    # categorical features
    # transformed names after numeric start at index = len(NUMERIC_FEATURES)
    offset = len(NUMERIC_FEATURES)
    for tn in transformed_names[offset:]:
        # tn looks like "species__Tiger"
        if "__" in tn:
            orig, val = tn.split("__", 1)
        else:
            orig = tn
        contributions.setdefault(orig, 0.0)
    # now sum actual values
    for idx, tn in enumerate(transformed_names[offset:], start=offset):
        val = float(shap_vals_row[idx])
        if "__" in tn:
            orig, _ = tn.split("__", 1)
        else:
            orig = tn
        contributions[orig] = contributions.get(orig, 0.0) + val
    return contributions

@app.post("/explain")
def explain(req: PredictRequest):
    """
    Return a local SHAP explanation for the single input.
    Response:
      {
        "base_value": float,
        "top_features": [
           {"feature":"poacher_signs_count", "contribution": -0.123, "abs": 0.123},
           ...
        ],
        "all_contributions": {"species": 0.1, "poacher_signs_count": -0.2, ...}
      }
    """
    if MODEL is None or PREPROC is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not available")

    payload = req.dict()
    # Build dataframe with expected columns
    df = pd.DataFrame([payload])
    # Ensure ordering and presence of ALL_FEATURES
    try:
        df = df.reindex(columns=ALL_FEATURES)
    except Exception:
        df = df.reindex(columns=ALL_FEATURES, fill_value=np.nan)

    # transform into model input
    try:
        X = PREPROC.transform(df[ALL_FEATURES])
    except Exception as e:
        logger.exception("Preprocessing failed for explain")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    # prepare/cached explainer
    try:
        explainer = getattr(MODEL, "__explainer__", None)
        if explainer is None:
            # Try TreeExplainer first (fast for tree models), else fallback to generic Explainer
            try:
                explainer = shap.TreeExplainer(MODEL)
            except Exception:
                explainer = shap.Explainer(MODEL)
            MODEL.__explainer__ = explainer
    except Exception as e:
        logger.exception("Failed to create SHAP explainer")
        raise HTTPException(status_code=500, detail=f"Explainer error: {e}")

    # compute SHAP values for this single sample (fast)
    try:
        shap_out = explainer(X)  # shap.Explanation or array-like depending on version
        # extract numeric array of shap values
        if hasattr(shap_out, "values"):
            shap_vals = shap_out.values
        else:
            # fallback: try shap_values
            shap_vals = explainer.shap_values(X)
    except Exception as e:
        logger.exception("SHAP computation failed")
        raise HTTPException(status_code=500, detail=f"SHAP error: {e}")

    # shap_vals shape handling:
    # If shap_vals is a list (multi-output), pick index 1 if binary classifier (probability for positive class),
    # otherwise pick first.
    try:
        if isinstance(shap_vals, list):
            # Choose the array with shape matching X's features dimension
            # For binary classifier, shap_values[1] often corresponds to positive class
            if len(shap_vals) == 2:
                shap_arr = np.array(shap_vals[1])
            else:
                shap_arr = np.array(shap_vals[0])
        else:
            shap_arr = np.array(shap_vals)
    except Exception:
        shap_arr = np.array(shap_vals)

    # ensure we have 2D array [n_samples, n_transformed_features]
    if shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)

    # transformed feature names (numeric + categorical one-hot)
    try:
        transformed_names = _get_transformed_feature_names(PREPROC)
    except Exception:
        # fallback: use ALL_FEATURES (not exact if one-hot used)
        transformed_names = ALL_FEATURES

    # Map back to original features and sum one-hot contributions per categorical feature
    contributions = _map_shap_to_original(shap_arr[0], transformed_names)

    # Build top features sorted by absolute contribution
    top = sorted(
        [{"feature": k, "contribution": float(v), "abs": abs(float(v))} for k, v in contributions.items()],
        key=lambda x: x["abs"], reverse=True
    )

    # include base / expected value if available
    base_value = None
    try:
        if hasattr(explainer, "expected_value"):
            base_value = float(explainer.expected_value if not isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value[0])
        elif hasattr(shap_out, "base_values"):
            base_value = float(shap_out.base_values.tolist()[0])
    except Exception:
        base_value = None

    return {
        "base_value": base_value,
        "top_features": top[:8],      # return top 8 contributors
        "all_contributions": contributions
    }
