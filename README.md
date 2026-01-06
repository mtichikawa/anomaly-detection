# Real-Time Anomaly Detection System

Production-ready streaming anomaly detection with multiple ML algorithms and ensemble voting.

## Features

- **Multiple Algorithms**
  - Isolation Forest
  - Statistical (Z-score)
  - LSTM Autoencoder
  - Ensemble Voting

- **Real-Time Processing**
  - Streaming data pipeline
  - Low latency detection
  - Configurable thresholds

- **Comprehensive Analysis**
  - Performance metrics
  - Visualization dashboard
  - Statistical evaluation

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

## Code Structure

- `src/detectors/pipeline.py` (500+ lines) - Complete detection system
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