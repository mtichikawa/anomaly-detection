"""
Tests for the LightGBMDetector and its feature engineering.

Tests run in two modes:
  - Always: feature engineering, graceful-degradation when no model exists,
    ensemble construction with and without the model file.
  - When lightgbm + model are available: scoring behavior on a freshly-fit
    detector against synthetic data.

The graceful-degradation tests are the load-bearing ones — they enforce
the contract that the ensemble works regardless of whether LightGBM is
installed or the model file is present.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors.feature_engineering import (
    FEATURE_NAMES,
    build_training_matrix,
    features_to_vector,
    make_features,
)
from src.detectors.lightgbm_detector import LightGBMDetector
from src.detectors.pipeline import EnsembleDetector, generate_synthetic_data


# ── Feature engineering ───────────────────────────────────────────────────────

def test_features_with_empty_history_returns_safe_defaults():
    feats = make_features(42.0, deque())
    assert feats["value"] == 42.0
    assert feats["rolling_mean"] == 42.0
    assert feats["rolling_std"] == 0.0
    assert feats["z_score"] == 0.0
    assert feats["diff_prev"] == 0.0
    # All keys present
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_features_with_history_compute_correctly():
    history = deque([10.0, 12.0, 14.0, 16.0, 18.0], maxlen=10)
    feats = make_features(20.0, history)
    assert feats["value"] == 20.0
    assert feats["rolling_mean"] == pytest.approx(14.0)
    assert feats["rolling_min"] == 10.0
    assert feats["rolling_max"] == 18.0
    assert feats["diff_prev"] == pytest.approx(2.0)   # 20 - 18
    # z_score = (20 - 14) / std(10..18) ≈ (6 / 2.828) ≈ 2.12
    assert feats["z_score"] == pytest.approx(6.0 / np.std([10, 12, 14, 16, 18]), rel=1e-3)


def test_features_to_vector_canonical_order():
    feats = make_features(5.0, deque([1.0, 2.0, 3.0]))
    vec = features_to_vector(feats)
    assert len(vec) == len(FEATURE_NAMES)
    # First element is always 'value'
    assert vec[0] == 5.0


def test_build_training_matrix_skips_first_window():
    np.random.seed(0)
    data = np.arange(100, dtype=float)
    labels = np.zeros(100, dtype=bool)
    labels[50] = True

    X, y = build_training_matrix(data, labels, window=10)
    assert X.shape == (90, len(FEATURE_NAMES))
    assert y.shape == (90,)
    # The True label was at index 50, which becomes index (50 - 10) = 40 in the matrix
    assert y[40] == True
    assert y.sum() == 1


def test_build_training_matrix_rejects_short_input():
    data = np.array([1.0, 2.0, 3.0])
    labels = np.array([False, False, False])
    with pytest.raises(ValueError):
        build_training_matrix(data, labels, window=10)


def test_build_training_matrix_rejects_mismatched_lengths():
    data = np.arange(50, dtype=float)
    labels = np.zeros(40, dtype=bool)
    with pytest.raises(ValueError):
        build_training_matrix(data, labels, window=10)


# ── Graceful degradation ──────────────────────────────────────────────────────

def test_detector_inert_when_model_missing(tmp_path):
    detector = LightGBMDetector(model_path=str(tmp_path / "no_such_model.pkl"))
    assert detector.trained is False
    # predict and get_score should return safe defaults
    assert detector.predict(100.0) is False
    assert detector.get_score(100.0) == 0.0
    # And history still updates so the inert detector doesn't break callers
    assert len(detector.history) >= 1


def test_ensemble_constructs_without_lightgbm_model(tmp_path, monkeypatch):
    # Force the default path to point somewhere with no model
    import src.detectors.lightgbm_detector as ld
    monkeypatch.setattr(ld, "DEFAULT_MODEL_PATH", tmp_path / "no_model.pkl")
    ensemble = EnsembleDetector()
    # The 4 unsupervised detectors must always be present
    names = [d.name for d in ensemble.detectors]
    assert "IsolationForest" in names
    assert "Statistical_strict" in names
    assert "Statistical_sensitive" in names
    assert "LSTM-Autoencoder" in names
    # LightGBM is NOT in the ensemble when no model file exists
    assert "LightGBM" not in names


# ── Trained-detector behavior ─────────────────────────────────────────────────

def test_detector_scores_higher_for_obvious_anomaly():
    pytest.importorskip("lightgbm")

    np.random.seed(42)
    data, labels = generate_synthetic_data(n_points=2000, anomaly_rate=0.05)

    detector = LightGBMDetector(window=10, model_path="/dev/null")
    detector.fit(data, labels)
    assert detector.trained is True

    # Warm up the history with normal data
    for v in data[:50]:
        detector.get_score(float(v))

    # Score the typical normal level vs a strong outlier; outlier should score higher
    normal_value = float(np.mean(data[:50]))
    outlier_value = normal_value + 100.0  # well outside the noise band

    normal_score = detector.get_score(normal_value)
    outlier_score = detector.get_score(outlier_value)

    assert outlier_score > normal_score


def test_detector_predict_returns_bool():
    pytest.importorskip("lightgbm")

    np.random.seed(7)
    data, labels = generate_synthetic_data(n_points=500, anomaly_rate=0.1)
    detector = LightGBMDetector(window=10, model_path="/dev/null")
    detector.fit(data, labels)

    result = detector.predict(50.0)
    assert isinstance(result, bool)
