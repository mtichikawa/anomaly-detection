import sys
from pathlib import Path

# src/ for detector imports, project root for app/ imports
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_detect_returns_valid_structure():
    np.random.seed(0)
    values = list(np.random.randn(50) * 2 + 100) + [999.0]
    response = client.post("/detect", json={"values": values})
    assert response.status_code == 200
    body = response.json()
    assert "anomaly_count" in body
    assert "anomaly_rate" in body
    assert "points" in body
    assert "stats" in body
    assert len(body["points"]) > 0


def test_detect_flags_spike():
    np.random.seed(42)
    normal = list(np.random.randn(40) * 2 + 100)
    spike = [999.0, 998.0]
    response = client.post("/detect", json={
        "values": normal + spike,
        "train_ratio": 0.8,
        "detector_type": "ensemble",
    })
    assert response.status_code == 200
    flagged = [p for p in response.json()["points"] if p["is_anomaly"]]
    assert len(flagged) >= 1


def test_detect_rejects_too_few_values():
    response = client.post("/detect", json={"values": [1.0, 2.0, 3.0]})
    assert response.status_code == 422


@pytest.mark.parametrize("detector_type", ["ensemble", "isolation", "statistical", "lstm"])
def test_detect_all_detector_types(detector_type):
    np.random.seed(7)
    values = list(np.random.randn(60) * 2 + 50)
    response = client.post("/detect", json={"values": values, "detector_type": detector_type})
    assert response.status_code == 200


def test_detect_is_anomaly_is_plain_bool():
    """Ensure is_anomaly fields are plain Python bool, not np.bool_ (JSON serialization guard)."""
    np.random.seed(1)
    values = list(np.random.randn(40) * 2 + 100) + [999.0]
    response = client.post("/detect", json={"values": values})
    assert response.status_code == 200
    for point in response.json()["points"]:
        assert isinstance(point["is_anomaly"], bool)


def test_detect_stats_fields_present():
    np.random.seed(3)
    values = list(np.random.randn(50) * 2 + 100)
    response = client.post("/detect", json={"values": values})
    assert response.status_code == 200
    stats = response.json()["stats"]
    for key in ("total_points", "anomalies_detected", "anomaly_rate", "avg_score", "max_score", "score_std"):
        assert key in stats
