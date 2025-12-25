# src/preprocess.py

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------
# Feature definitions
# ----------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "species",
    "state",
    "time_of_day",
    "terrain_type"
]

NUMERIC_FEATURES = [
    "proximity_to_village_km",
    "distance_to_road_km",
    "past_poaching_incidents",
    "poacher_signs_count",
    "market_value_usd",
    "community_awareness_score",
    "patrol_presence_last_24h",
    "camera_trap_present",
    "latitude",
    "longitude",
    "nightlight",
    "ndvi",
    "elevation"
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

NIGHTLIGHT_INDEX = NUMERIC_FEATURES.index("nightlight")


# ----------------------------------------------------------------------
# Optional: state lat/lon ranges (frontend validation)
# ----------------------------------------------------------------------
STATE_RANGES = {
    "Assam":             {"lat":[24.1793, 28.7293], "lon":[91.1351, 95.1843]},
    "Gujarat":           {"lat":[20.6742, 24.1503], "lon":[69.4059, 73.0288]},
    "Himachal Pradesh":  {"lat":[29.6463, 33.4447], "lon":[74.6491, 78.4641]},
    "Karnataka":         {"lat":[13.2367, 17.3336], "lon":[73.9850, 77.8989]},
    "Kerala":            {"lat":[8.2183, 12.4075],  "lon":[74.2017, 78.3782]},
    "Madhya Pradesh":    {"lat":[19.9267, 24.2686], "lon":[76.3461, 79.9605]},
    "Odisha":            {"lat":[18.4529, 22.4759], "lon":[83.8550, 87.6928]},
    "Rajasthan":         {"lat":[24.0967, 27.8451], "lon":[70.9231, 74.6447]},
    "Uttarakhand":       {"lat":[28.0152, 31.8810], "lon":[77.0423, 80.9724]},
    "West Bengal":       {"lat":[21.0603, 25.1401], "lon":[86.0419, 89.6897]}
}


# ----------------------------------------------------------------------
# Custom transformer for nightlight
# ----------------------------------------------------------------------
class NightlightLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nightlight_index: int = NIGHTLIGHT_INDEX):
        self.nightlight_index = nightlight_index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a 2D numpy array (numeric pipeline receives arrays)
        X = X.copy()
        # guard: if index out of bounds, don't crash
        if 0 <= self.nightlight_index < X.shape[1]:
            X[:, self.nightlight_index] = np.log1p(
                np.clip(X[:, self.nightlight_index], 0, None)
            )
        return X


# ----------------------------------------------------------------------
# Build preprocessing pipeline
# ----------------------------------------------------------------------
def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("nightlight_log", NightlightLogTransformer(NIGHTLIGHT_INDEX)),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor
