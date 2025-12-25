from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    species: str
    state: str
    time_of_day: str
    terrain_type: str
    proximity_to_village_km: float
    distance_to_road_km: float
    patrol_presence_last_24h: int
    camera_trap_present: int
    past_poaching_incidents: int
    poacher_signs_count: int
    market_value_usd: float
    community_awareness_score: int

class PredictResponse(BaseModel):
    poaching_risk_score: float = Field(..., ge=0.0, le=1.0)
    poaching_occurred: int = Field(..., ge=0, le=1)
    model_version: str
