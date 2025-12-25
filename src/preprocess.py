# src/preprocess.py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Categorical features (unchanged)
CATEGORICAL_FEATURES = [
    "species",
    "state",
    "time_of_day",
    "terrain_type"
]

# Numeric features - add latitude & longitude here
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
    "longitude"
]

# Combined
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Optional: state -> lat/lon ranges (same ranges used in frontend)
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



def build_preprocessor():
    """
    Returns a ColumnTransformer preprocessor that imputes & scales numeric features
    and imputes & one-hot-encodes categorical features.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUMERIC_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor
