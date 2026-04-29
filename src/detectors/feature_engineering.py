"""
Feature engineering for the supervised LightGBM detector.

The other detectors in the ensemble (Isolation Forest, Statistical, LSTM)
are unsupervised — they operate on the raw value alone. LightGBM is
supervised and earns its keep by exploiting engineered features that
encode local context: rolling statistics, z-score, recent volatility,
diffs, and ratios.

Features are computed from a sliding history window so the detector can
run in streaming mode (no full-sequence access required).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List

import numpy as np


# Names in deterministic order. The training script and the streaming
# detector both use this list to keep feature ordering aligned with the
# trained LightGBM model's expectation.
FEATURE_NAMES: List[str] = [
    "value",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "z_score",
    "diff_prev",
    "ratio_to_mean",
    "range_pct",
]


def make_features(value: float, history: Deque[float]) -> dict:
    """
    Compute the feature vector for one streaming point.

    Args:
        value:   the current data point.
        history: a deque of recent values (NOT including the current point).
                 Empty or short histories produce sensible defaults so the
                 streaming detector can score its first few points.

    Returns:
        dict keyed by FEATURE_NAMES. Caller is responsible for converting
        to a list/array in FEATURE_NAMES order before model inference.
    """
    if len(history) == 0:
        return {
            "value":         float(value),
            "rolling_mean":  float(value),
            "rolling_std":   0.0,
            "rolling_min":   float(value),
            "rolling_max":   float(value),
            "z_score":       0.0,
            "diff_prev":     0.0,
            "ratio_to_mean": 1.0,
            "range_pct":     0.0,
        }

    arr = np.asarray(history, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    rmin = float(arr.min())
    rmax = float(arr.max())
    last = float(arr[-1])

    z = 0.0 if std < 1e-10 else (value - mean) / std
    ratio = 1.0 if abs(mean) < 1e-10 else value / mean
    range_pct = 0.0 if abs(mean) < 1e-10 else (rmax - rmin) / mean

    return {
        "value":         float(value),
        "rolling_mean":  mean,
        "rolling_std":   std,
        "rolling_min":   rmin,
        "rolling_max":   rmax,
        "z_score":       float(z),
        "diff_prev":     float(value - last),
        "ratio_to_mean": float(ratio),
        "range_pct":     float(range_pct),
    }


def features_to_vector(feats: dict) -> List[float]:
    """Convert a feature dict to a list in the canonical FEATURE_NAMES order."""
    return [feats[name] for name in FEATURE_NAMES]


def build_training_matrix(
    data: np.ndarray,
    labels: np.ndarray,
    window: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a feature matrix X and label vector y from a 1-D time series.

    A sliding window of `window` previous points generates features for
    each point. The first `window` points are skipped (not enough history
    to compute meaningful features) so X has fewer rows than `data`.

    Args:
        data:   1-D numeric array.
        labels: 1-D boolean array, same length as data.
        window: history length used for rolling features.

    Returns:
        (X, y) — X is shape (n - window, len(FEATURE_NAMES)).
    """
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")
    if len(data) <= window:
        raise ValueError(
            f"Need more than `window` ({window}) data points; got {len(data)}"
        )

    history: Deque[float] = deque(maxlen=window)
    rows: list[list[float]] = []
    targets: list[bool] = []

    for i, value in enumerate(data):
        if len(history) >= window:
            feats = make_features(float(value), history)
            rows.append(features_to_vector(feats))
            targets.append(bool(labels[i]))
        history.append(float(value))

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=bool)
