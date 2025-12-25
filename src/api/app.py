"""
FastAPI app for Anti-Poaching risk prediction.

Run (from project root):
    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

Notes:
- Ensure models are saved at models/xgb_model.joblib and models/preprocessor.joblib,
  or set environment variables MODEL_PATH and PREPROC_PATH.
- This app exposes:
    GET  /health
    GET  /metadata
    POST /predict
"""
import os
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

# try to import helper loaders if present; otherwise fallback to joblib
try:
    from src.models.save_load import load_model as load_model_helper
except Exception:
    load_model_helper = None

try:
    from src.preprocess import load_preprocessor as load_preproc_helper
    from src.preprocess import ALL_FEATURES
except Exception:
    load_preproc_helper = None
    ALL_FEATURES = [
        "species","state","time_of_day","terrain_type",
        "proximity_to_village_km","distance_to_road_km",
        "past_poaching_incidents","poacher_signs_count",
        "market_value_usd","community_awareness_score",
        "patrol_presence_last_24h","camera_trap_present"
    ]

from src.api.schemas import PredictRequest, PredictResponse
from src.api.logger import log_prediction

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Paths (can be overridden with env vars)
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_model.joblib")
PREPROC_PATH = os.getenv("PREPROC_PATH", "models/preprocessor.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

app = FastAPI(title="Anti-Poaching Risk API", version="1.0")

# CORS - allow all origins for dev. In production restrict to frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_artifacts():
    global MODEL, PREPROC
    # load model
    try:
        if load_model_helper:
            MODEL = load_model_helper(MODEL_PATH)
        else:
            MODEL = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}")
        MODEL = None
    except Exception as e:
        logger.exception("Failed to load model")
        MODEL = None

    # load preprocessor
    try:
        if load_preproc_helper:
            PREPROC = load_preproc_helper(PREPROC_PATH)
        else:
            PREPROC = joblib.load(PREPROC_PATH)
    except FileNotFoundError:
        logger.error(f"Preprocessor not found at {PREPROC_PATH}")
        PREPROC = None
    except Exception:
        logger.exception("Failed to load preprocessor")
        PREPROC = None

    if MODEL is not None:
        # try infer a readable version string from model
        try:
            params = MODEL.get_params()
            MODEL.__meta__ = {"params": params}
        except Exception:
            MODEL.__meta__ = {"info": "no params available"}


@app.get("/health")
def health():
    ok = MODEL is not None and PREPROC is not None
    return {"ok": ok, "model_loaded": MODEL is not None, "preprocessor_loaded": PREPROC is not None}


@app.get("/metadata")
def metadata():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    meta = getattr(MODEL, "__meta__", {})
    return {"model_version": MODEL_VERSION, "meta": meta}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None or PREPROC is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not available (check logs)")

    # Build dataframe in the expected column order
    payload = req.dict()
    try:
        df = pd.DataFrame([payload], columns=ALL_FEATURES)
    except Exception:
        # fallback: construct and reorder
        df = pd.DataFrame([payload])
        df = df.reindex(columns=ALL_FEATURES)

    # transform
    try:
        X = PREPROC.transform(df[ALL_FEATURES])
    except Exception as e:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    # predict
    try:
        proba = MODEL.predict_proba(X)[:, 1][0]
        pred = int(proba >= 0.5)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # build response
    resp = {
        "poaching_risk_score": float(round(float(proba), 4)),
        "poaching_occurred": int(pred),
        "model_version": MODEL_VERSION
    }

    # log prediction (fire-and-forget simple write)
    try:
        log_prediction(payload, resp, MODEL_VERSION)
    except Exception:
        logger.exception("Failed to log prediction (non-fatal)")

    return resp
