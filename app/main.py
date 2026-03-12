import sys
from pathlib import Path

# Make src/ importable when running via uvicorn from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from fastapi import FastAPI
from app.schemas import DetectRequest, DetectResponse, DetectionStats, PointResult
from detectors.pipeline import StreamingPipeline

app = FastAPI(
    title="Anomaly Detection API",
    description="REST endpoint wrapping the StreamingPipeline ensemble detector",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(body: DetectRequest):
    values = body.values
    train_n = max(1, int(len(values) * body.train_ratio))

    train_data = np.array(values[:train_n])
    test_data = np.array(values[train_n:]) if train_n < len(values) else np.array(values)

    pipeline = StreamingPipeline(detector_type=body.detector_type)

    # Override voting threshold when using ensemble
    if body.detector_type == "ensemble":
        pipeline.detector.voting_threshold = body.voting_threshold

    pipeline.train(train_data)
    output = pipeline.process_stream(test_data)

    points = [
        PointResult(
            index=r["index"],
            value=float(r["value"]),
            is_anomaly=bool(r["is_anomaly"]),  # cast np.bool_ to avoid JSON bug
            score=float(r["score"]),
        )
        for r in output["results"]
    ]

    raw_stats = output["stats"]
    stats = DetectionStats(
        total_points=int(raw_stats["total_points"]),
        anomalies_detected=int(raw_stats["anomalies_detected"]),
        anomaly_rate=float(raw_stats["anomaly_rate"]),
        avg_score=float(raw_stats["avg_score"]),
        max_score=float(raw_stats["max_score"]),
        score_std=float(raw_stats["score_std"]),
    )

    return DetectResponse(
        anomaly_count=stats.anomalies_detected,
        anomaly_rate=stats.anomaly_rate,
        points=points,
        stats=stats,
    )
