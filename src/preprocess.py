from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

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
    "camera_trap_present"
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_preprocessor():
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
