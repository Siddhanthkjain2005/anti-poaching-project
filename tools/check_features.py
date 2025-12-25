# tools/check_features.py (part 2)
import joblib
import numpy as np
from collections import defaultdict
from src.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def aggregate_importances(importances, transformed_names):
    # Aggregate one-hot columns back to original categorical
    agg = defaultdict(float)
    offset = len(NUMERIC_FEATURES)
    # numeric: one-to-one
    for i, feat in enumerate(NUMERIC_FEATURES):
        agg[feat] += float(importances[i])
    # categorical: parse prefix before '_' or '__'
    for j, name in enumerate(transformed_names[offset:], start=offset):
        val = float(importances[j])
        if "_" in name:
            orig = name.split("_", 1)[0]
        elif "__" in name:
            orig = name.split("__", 1)[0]
        else:
            orig = name
        agg[orig] += val
    return dict(agg)

# load
pipeline = joblib.load("models/xgb_model.joblib")  # your saved pipeline or model
# if pipeline is a sklearn Pipeline with preprocessor + model:
try:
    preproc = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
except Exception:
    # if you saved raw model and preproc separately:
    preproc = joblib.load("models/preprocessor.joblib")
    model = joblib.load("models/xgb_model.joblib")

# transformed feature names (use function from part 1)
def get_transformed_feature_names(preprocessor):
    names = []
    names.extend(NUMERIC_FEATURES)
    try:
        cat_pipe = preprocessor.named_transformers_.get("cat")
        encoder = None
        if cat_pipe:
            encoder = cat_pipe.named_steps.get("encoder")
            if encoder is None and hasattr(cat_pipe, "get_feature_names_out"):
                encoder = cat_pipe
        if encoder is not None:
            cat_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            names.extend(list(cat_names))
    except Exception:
        pass
    return names

names = get_transformed_feature_names(preproc)

# get importances
if hasattr(model, "feature_importances_"):
    imp = model.feature_importances_
else:
    # XGBoost sklearn wrapper may have .feature_importances_ or else use booster
    try:
        imp = model.get_booster().get_score(importance_type='gain')
        # get_score returns dict mapping 'f0','f1' -> score. Convert to array.
        max_idx = max(int(k[1:]) for k in imp.keys())
        imp_arr = np.zeros(max_idx + 1)
        for k, v in imp.items():
            idx = int(k[1:])
            imp_arr[idx] = v
        imp = imp_arr
    except Exception:
        raise RuntimeError("Model has no accessible feature_importances_")

# align length
if len(imp) != len(names):
    print("Warning: feature_importance length != transformed feature names. Lengths:", len(imp), len(names))

agg = aggregate_importances(imp, names)
# print sorted
for feat, val in sorted(agg.items(), key=lambda x: x[1], reverse=True)[:30]:
    print(f"{feat:25s}  {val:.6f}")
