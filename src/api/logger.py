import json
import os
from datetime import datetime
from typing import Any, Dict

LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "logs/predictions.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_prediction(payload: Dict[str, Any], result: Dict[str, Any], model_version: str) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_version": model_version,
        "input": payload,
        "result": result
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
