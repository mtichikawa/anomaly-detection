# Real-Time Anomaly Detection System

Production-ready streaming anomaly detection with multiple ML algorithms and ensemble voting, now also available as a containerized REST API.

## Features

- **Multiple Algorithms**
  - Isolation Forest (unsupervised)
  - Statistical Z-score, two thresholds (unsupervised)
  - LSTM Autoencoder (unsupervised)
  - LightGBM gradient-boosted trees (**supervised**, added Apr 2026)
  - Ensemble Voting

- **Real-Time Processing**
  - Streaming data pipeline
  - Low latency detection
  - Configurable thresholds

- **Comprehensive Analysis**
  - Performance metrics
  - Visualization dashboard
  - Statistical evaluation

## Supervised Member: LightGBM Detector

The original ensemble was four unsupervised detectors. The Apr 2026 upgrade adds a supervised 5th member using LightGBM gradient-boosted trees, trained on labeled synthetic data with engineered rolling-window features. Mixing supervised and unsupervised members in one voting scheme is the architectural point: when labels are available, the supervised detector contributes calibrated probability; when they're not, the four unsupervised members still cover the space.

### Engineered features

Computed from a sliding window of recent values (default window=10):
- `value` — the raw point
- `rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max` — local statistics
- `z_score` — `(value - rolling_mean) / rolling_std`
- `diff_prev` — `value - history[-1]`
- `ratio_to_mean` — `value / rolling_mean`
- `range_pct` — `(rolling_max - rolling_min) / rolling_mean`

### Training: chronological split, not random

Random train/test splits leak future state into past tests for time-series. The training script (`scripts/train_lightgbm.py`) splits the synthetic data **chronologically** at 80% — train on the first 80% of bars, test on the last 20%. This is the same lookahead-bias control that walk-forward validation uses for time-series ML.

### Permutation importance

After training, permutation importance ranks features by how much accuracy degrades when each one is shuffled. Result on the synthetic dataset (anomaly rate 5%, 5,000 points):

| Feature | Importance (mean ± std) |
|---|---|
| `z_score` | +0.024 ± 0.003 |
| `ratio_to_mean` | +0.020 ± 0.002 |
| `diff_prev` | +0.008 ± 0.002 |
| value, rolling_mean, rolling_std, rolling_min, rolling_max, range_pct | ~0 |

Two features carry almost all the signal: `z_score` (deviation from local mean, normalized) and `ratio_to_mean` (proportional deviation, scale-invariant). The raw value and the rolling window summary statistics turn out to add no marginal predictive power on top of the derived features. That's an honest result for the writeup — most engineered features are redundant; a small number do all the work.

### Test metrics on the held-out chronological split

| Metric | Value |
|---|---|
| Accuracy | 0.999 |
| Precision | 1.000 |
| Recall | 0.974 |
| F1 | 0.987 |
| AUC | 0.9999 |

These numbers are higher than what a real production deployment would see — synthetic data has cleaner anomaly signatures than real-world streams. The interesting result is the ranked feature importance, which identifies which engineered signals to keep when this gets adapted to a real dataset.

### Graceful degradation

The ensemble loads the LightGBM detector only if `models/lightgbm_anomaly.pkl` exists. If the model file is missing or LightGBM isn't installed, the ensemble silently falls back to the four unsupervised detectors. No code path crashes. To regenerate the model: `python scripts/train_lightgbm.py`.

## Quick Start

```bash
pip install -r requirements.txt
python examples/quick_demo.py
```

## Usage

```python
from src.detectors.pipeline import StreamingPipeline

# Create and train
pipeline = StreamingPipeline(detector_type='ensemble')
pipeline.train(training_data)

# Detect anomalies
results = pipeline.process_stream(test_data)

# Get anomalies
for anomaly in pipeline.anomalies:
    print(f"Anomaly at {anomaly['index']}: {anomaly['value']}")
```

## REST API (Dockerized)

The detection system is also available as a containerized FastAPI service for real-time anomaly detection over HTTP.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health, detector status, uptime |
| `POST` | `/detect` | Submit a single observation for anomaly detection |
| `GET` | `/docs` | Swagger UI (auto-generated) |

### Running with Docker

```bash
docker-compose up --build
```

The API starts on `http://localhost:8000`. Hit `/health` to confirm detectors are loaded, then POST observations to `/detect`:

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"value": 42.5}'
```

The `/detect` endpoint returns the ensemble vote (anomaly or normal), individual detector results, and a confidence score. Detector selection is configurable at startup.

## Code Structure

- `src/detectors/pipeline.py` (500+ lines) - Detectors + EnsembleDetector + StreamingPipeline
- `src/detectors/feature_engineering.py` - Rolling-window features for the supervised detector
- `src/detectors/lightgbm_detector.py` - LightGBM detector with graceful-degradation loading
- `scripts/train_lightgbm.py` - Trains the supervised model + permutation importance
- `models/lightgbm_anomaly.pkl` - Serialized trained model (regenerable)
- `models/feature_importance.json` - Permutation importance results
- `src/visualizations.py` (350+ lines) - Visualization suite
- `notebooks/complete_demo.ipynb` - Full workflow
- `examples/quick_demo.py` - 5-minute demo

## Performance

Tested on synthetic data:
- F1 Score: 0.81+
- Precision: 0.85+
- Recall: 0.78+

## What I Learned

- Streaming data processing
- Ensemble machine learning methods
- Production system design
- Trade-offs: accuracy vs latency

Contact: Mike Ichikawa - projects.ichikawa@gmail.com

# 2026-01-05
# 2026-01-05
# 2026-01-08
# 2026-01-11
# 2026-01-14
# 2026-01-17
# 2026-01-20
# 2026-01-23
# 2026-01-26
# 2026-01-29
# 2026-02-01
# 2026-02-04
# 2026-02-07
# 2026-02-10
# 2026-02-13
# 2026-02-16
# 2026-02-17
# 2026-02-18
<!-- reviewed 2026-03-07 -->
