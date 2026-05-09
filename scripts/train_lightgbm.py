"""
scripts/train_lightgbm.py — Train the supervised LightGBM detector.

Generates labeled synthetic data, engineers rolling-window features,
splits chronologically (NOT randomly — random splits leak future state
into past tests for time-series), trains a regularized LightGBM
classifier, runs permutation importance, and saves both the model and
the importance results.

Usage:
    python scripts/train_lightgbm.py
"""

from pathlib import Path
import json
import logging
import sys

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detectors.feature_engineering import FEATURE_NAMES, build_training_matrix
from src.detectors.pipeline import generate_synthetic_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("train_lightgbm")

MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "lightgbm_anomaly.pkl"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

WINDOW = 10
N_POINTS = 5000
ANOMALY_RATE = 0.05
TRAIN_FRACTION = 0.8


def main() -> int:
    log.info(f"Generating {N_POINTS} synthetic points (anomaly rate {ANOMALY_RATE})")
    np.random.seed(42)
    data, labels = generate_synthetic_data(
        n_points=N_POINTS, anomaly_rate=ANOMALY_RATE
    )

    log.info(f"Building feature matrix with window={WINDOW}")
    X, y = build_training_matrix(data, labels, window=WINDOW)
    log.info(f"X shape: {X.shape}, anomalies: {int(y.sum())} ({y.mean():.1%})")

    # Chronological split — never random for time series
    split_idx = int(len(X) * TRAIN_FRACTION)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    log.info(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    try:
        import lightgbm as lgb
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )
        import joblib
    except ImportError as e:
        log.error(f"Missing dependency: {e}. Install lightgbm + scikit-learn + joblib.")
        return 1

    pos = max(int(y_train.sum()), 1)
    neg = len(y_train) - pos
    log.info(f"Class balance — train pos: {pos}, neg: {neg}")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        class_weight="balanced",
        scale_pos_weight=float(neg / pos),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    # Test metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None,
        "test_anomaly_rate": float(y_test.mean()),
        "test_size": int(len(y_test)),
    }
    log.info("Test metrics:")
    for k, v in metrics.items():
        log.info(f"  {k}: {v}")

    # Permutation importance — read on the test split (no leakage)
    log.info("Computing permutation importance (this takes a few seconds)")
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    importance = {
        FEATURE_NAMES[i]: {
            "mean": float(perm.importances_mean[i]),
            "std": float(perm.importances_std[i]),
        }
        for i in range(len(FEATURE_NAMES))
    }
    log.info("Permutation importance (mean ± std):")
    ranked = sorted(importance.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    for name, stats in ranked:
        log.info(f"  {name:18s} {stats['mean']:+.4f} ± {stats['std']:.4f}")

    # Persist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info(f"Saved model -> {MODEL_PATH}")

    payload = {
        "model_path": str(MODEL_PATH.relative_to(REPO_ROOT)),
        "training_config": {
            "n_points": N_POINTS,
            "anomaly_rate": ANOMALY_RATE,
            "window": WINDOW,
            "train_fraction": TRAIN_FRACTION,
            "split_kind": "chronological",
            "random_seed": 42,
        },
        "test_metrics": metrics,
        "permutation_importance": importance,
        "feature_names": FEATURE_NAMES,
    }
    IMPORTANCE_PATH.write_text(json.dumps(payload, indent=2))
    log.info(f"Saved importance -> {IMPORTANCE_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
