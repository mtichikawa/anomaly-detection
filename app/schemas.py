from typing import List, Literal
from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    values: List[float] = Field(..., min_length=10, description="Time series values to analyze")
    train_ratio: float = Field(0.5, gt=0, lt=1, description="Fraction of values used as training window")
    detector_type: Literal["ensemble", "isolation", "statistical", "lstm"] = "ensemble"
    voting_threshold: float = Field(0.5, gt=0, le=1, description="Fraction of detectors that must agree (ensemble only)")


class PointResult(BaseModel):
    index: int
    value: float
    is_anomaly: bool
    score: float


class DetectionStats(BaseModel):
    total_points: int
    anomalies_detected: int
    anomaly_rate: float
    avg_score: float
    max_score: float
    score_std: float


class DetectResponse(BaseModel):
    anomaly_count: int
    anomaly_rate: float
    points: List[PointResult]
    stats: DetectionStats
