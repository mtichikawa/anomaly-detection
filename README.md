# Real-Time Anomaly Detection System

Production-ready streaming anomaly detection with multiple ML algorithms and ensemble voting, now also available as a containerized REST API.

## Features

- **Multiple Algorithms**
  - Isolation Forest (unsupervised)
  - Statistical Z-score, two thresholds (unsupervised)
  - LSTM Autoencoder (unsupervised)
  - LightGBM gradient-boosted trees (**supervised**, added Apr 2026)
  - Ensemble Voting
  - Concept-drift layer: Page-Hinkley + Kolmogorov-Smirnov (added Jun 2026)

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

## Concept drift

The point-anomaly detectors answer "is this single value weird relative to the baseline?" But the baseline itself can move. A z-score detector calibrated on summer weekday traffic is miscalibrated for a holiday weekend; a model trained on pre-launch behavior is wrong about post-launch behavior. That slow change in the underlying distribution is **concept drift**, and when it happens the detectors silently start either over-firing or under-firing. Detecting drift is a separate problem from detecting anomalies, and it needs its own layer.

The Jun 2026 upgrade adds an optional `DriftDetector` (`src/detectors/drift_detector.py`) that the `StreamingPipeline` runs alongside the ensemble. When it trips, the pipeline emits a `drift_alert` event kept distinct from a point-anomaly alert (separate `drift_events` list, separate `drift_alert` flag on each record). It's a univariate layer — one detector per stream.

### Two complementary methods

| Method | Catches | How it works |
|---|---|---|
| `page_hinkley` | mean shift (gradual or step) | Cumulative-sum test. Tracks the running deviation of each value from the stream mean (standardized by the running std, so the threshold is scale-free) and fires when the accumulated deviation crosses `threshold`. Streaming-native, low memory. |
| `ks` | distribution shape change | Two-sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) comparing a reference window of older values against a recent window of newer ones. Fires when the p-value drops below `threshold`. Catches variance changes and bimodality that a mean-shift test misses. |

**When to use which:** Page-Hinkley for "did the average move?" — it's cheap and reacts fast to a shift in level. KS for "did the shape of the distribution change?" — slightly more expensive, but it sees changes that leave the mean untouched (e.g. the spread doubling). Use both for full coverage of univariate drift.

After a drift event fires, the detector re-baselines so you get one clean alert per drift instead of an alarm on every subsequent point.

```python
from src.detectors.pipeline import StreamingPipeline
from src.detectors.drift_detector import DriftDetector

drift = DriftDetector(method='page_hinkley', reference_window_size=200, threshold=12.0)
pipeline = StreamingPipeline(detector_type='ensemble', drift_detector=drift)
pipeline.train(training_data)

results = pipeline.process_stream(stream)
for ev in pipeline.drift_events:
    print(ev['index'], ev['drift']['detail'])
```

Run the end-to-end example (stationary stream that takes a step shift; the ensemble flags the one-off spikes, the drift layer flags the sustained baseline move):

```bash
python examples/drift_demo.py
```

**Follow-ups (not in scope for this layer):** automatic retraining of the point-anomaly detectors when drift fires; multivariate drift detection (e.g. MMD — Maximum Mean Discrepancy — over multiple streams jointly); drift visualization on the Dockerized REST API dashboard.

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
- `src/detectors/drift_detector.py` - Concept-drift detector (Page-Hinkley + KS)
- `examples/drift_demo.py` - Concept-drift demo (stationary stream + step shift)
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

> _Note: ensemble votes use a strict 2/3 majority by default. Lowering the threshold increases recall at the cost of precision — tune to the cost asymmetry of false positives vs missed anomalies in your domain._
