"""
LightGBM supervised anomaly detector.

This is the 4th member of the ensemble (alongside Isolation Forest, two
StatisticalDetectors, and the LSTM autoencoder, all unsupervised). The
supervised LightGBM model is trained on labeled synthetic data with
engineered rolling-window features (see feature_engineering.py) and
loaded at runtime from a serialized .pkl so the ensemble can boot
without a training step.

The detector is API-compatible with the other BaseDetector subclasses:
fit(), predict(), get_score(). The fit method accepts labels (the other
detectors don't), and silently no-ops if no model is loaded.

Graceful degradation: if the model file is missing or LightGBM isn't
installed, the detector reports trained=False and the EnsembleDetector
in pipeline.py skips it. The ensemble continues with the original
three detectors.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np

from .pipeline import BaseDetector
from .feature_engineering import (
    FEATURE_NAMES,
    build_training_matrix,
    features_to_vector,
    make_features,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "lightgbm_anomaly.pkl"
)


class LightGBMDetector(BaseDetector):
    """
    Supervised anomaly detector using a trained LightGBM classifier.

    Args:
        window: history length for rolling-window features. Must match
                the value used during training.
        model_path: optional path to a serialized model. Defaults to
                models/lightgbm_anomaly.pkl. If absent, detector
                reports trained=False and predict()/get_score() return
                False/0.0 (graceful degradation).
        decision_threshold: probability threshold for the binary
                predict() decision. get_score() returns the raw
                probability regardless.
    """

    def __init__(
        self,
        window: int = 10,
        model_path: Optional[str] = None,
        decision_threshold: float = 0.5,
    ):
        super().__init__("LightGBM")
        self.window = window
        self.decision_threshold = decision_threshold
        self.history: Deque[float] = deque(maxlen=window)
        self.model = None

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if path.exists():
            try:
                import joblib
                self.model = joblib.load(path)
                self.trained = True
                log.info(f"LightGBMDetector loaded model from {path}")
            except Exception as e:
                log.warning(
                    f"LightGBMDetector failed to load {path}: {e}. "
                    f"Detector will be inert."
                )
                self.trained = False
        else:
            log.info(
                f"LightGBMDetector: no model at {path}. "
                f"Detector will be inert; ensemble degrades to 3 detectors."
            )
            self.trained = False

    # ── BaseDetector interface ──────────────────────────────────────

    def fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Train a new LightGBM model on (data, labels) and persist it.

        This is for ad-hoc retraining; the normal flow trains via
        `scripts/train_lightgbm.py` and loads at construction time.

        Args:
            data: 1-D numeric array.
            labels: 1-D boolean array, same length as data. If None,
                    fit is a no-op (we have no targets to learn).
        """
        if labels is None:
            log.warning(
                "LightGBMDetector.fit called without labels; skipping. "
                "Use train_lightgbm.py to train, or pass labels here."
            )
            return

        try:
            import lightgbm as lgb
        except ImportError:
            log.warning("lightgbm not installed; skipping fit.")
            return

        X, y = build_training_matrix(data, labels, window=self.window)
        if len(np.unique(y)) < 2:
            log.warning(
                "Training data contains only one class; skipping fit."
            )
            return

        pos = max(int(y.sum()), 1)
        neg = len(y) - pos
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            class_weight="balanced",
            scale_pos_weight=float(neg / pos),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X, y)
        self.trained = True

    def predict(self, value: float) -> bool:
        """Return True if the model predicts anomaly above the threshold."""
        if not self.trained or self.model is None:
            self.history.append(float(value))
            return False

        feats = make_features(float(value), self.history)
        x = np.asarray([features_to_vector(feats)], dtype=float)
        try:
            prob = float(self.model.predict_proba(x)[0, 1])
        except Exception as e:
            log.debug(f"LightGBM predict_proba failed: {e}")
            self.history.append(float(value))
            return False

        self.history.append(float(value))
        return prob >= self.decision_threshold

    def get_score(self, value: float) -> float:
        """Return the raw anomaly probability in [0, 1]."""
        if not self.trained or self.model is None:
            return 0.0

        feats = make_features(float(value), self.history)
        x = np.asarray([features_to_vector(feats)], dtype=float)
        try:
            prob = float(self.model.predict_proba(x)[0, 1])
            return max(0.0, min(prob, 1.0))
        except Exception:
            return 0.0
