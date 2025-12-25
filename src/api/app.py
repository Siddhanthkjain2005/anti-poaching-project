# inside predict(req: PredictRequest)
payload = req.dict()

# Build dataframe in expected order using ALL_FEATURES from preprocess
try:
    from src.preprocess import ALL_FEATURES
except Exception:
    ALL_FEATURES = [
        "species","state","time_of_day","terrain_type",
        "proximity_to_village_km","distance_to_road_km",
        "past_poaching_incidents","poacher_signs_count",
        "market_value_usd","community_awareness_score",
        "patrol_presence_last_24h","camera_trap_present",
        "latitude","longitude"
    ]

# Ensure all features are present in df (pandas will fill missing with NaN)
df = pd.DataFrame([payload])
# Reorder if possible
df = df.reindex(columns=ALL_FEATURES)

# transform and predict
try:
    X = PREPROC.transform(df[ALL_FEATURES])
except Exception as e:
    logger.exception("Preprocessing failed")
    raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")
