'''
Tests for the concept-drift detector (Page-Hinkley + KS) and its
integration into the StreamingPipeline.

All streams use np.random.default_rng with fixed seeds so the
pass/fail outcome is deterministic.
'''

import numpy as np
import pytest

from detectors.drift_detector import DriftDetector, DriftEvent
from detectors.pipeline import StreamingPipeline


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_invalid_method_raises():
    with pytest.raises(ValueError):
        DriftDetector(method='not_a_method')


def test_default_thresholds_per_method():
    ph = DriftDetector(method='page_hinkley')
    ks = DriftDetector(method='ks')
    assert ph.threshold == 10.0          # CUSUM magnitude
    assert ks.threshold == 0.01          # p-value alpha


# ---------------------------------------------------------------------------
# Page-Hinkley (mean shift)
# ---------------------------------------------------------------------------

def test_page_hinkley_fires_on_mean_shift():
    '''A clear step change in the mean should produce at least one event.'''
    rng = np.random.default_rng(0)
    det = DriftDetector(method='page_hinkley', reference_window_size=100,
                        threshold=10.0, delta=0.5)

    events = []
    for v in rng.normal(0.0, 1.0, 150):      # establish baseline
        e = det.update(v)
        if e is not None:
            events.append(e)
    for v in rng.normal(5.0, 1.0, 150):      # step shift in the mean
        e = det.update(v)
        if e is not None:
            events.append(e)

    assert len(events) >= 1
    assert isinstance(events[0], DriftEvent)
    assert events[0].method == 'page_hinkley'
    # Fired during the shifted segment, not during warmup.
    assert events[0].index >= 150


def test_page_hinkley_does_not_fire_on_stationary_noise():
    '''Stationary noise must not trip the cumulative-sum threshold.'''
    rng = np.random.default_rng(0)
    det = DriftDetector(method='page_hinkley', reference_window_size=100,
                        threshold=10.0, delta=0.5)

    events = [det.update(v) for v in rng.normal(0.0, 1.0, 1000)]
    assert all(e is None for e in events)


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov (distributional change)
# ---------------------------------------------------------------------------

def test_ks_fires_for_different_distributions():
    '''Reference vs recent drawn from different distributions -> drift.'''
    rng = np.random.default_rng(0)
    det = DriftDetector(method='ks', reference_window_size=100,
                        recent_window_size=100, threshold=0.01)

    events = []
    for v in rng.normal(0.0, 1.0, 100):      # fills the reference half
        e = det.update(v)
        if e is not None:
            events.append(e)
    for v in rng.normal(5.0, 1.0, 150):      # recent half diverges
        e = det.update(v)
        if e is not None:
            events.append(e)

    assert len(events) >= 1
    assert events[0].method == 'ks'
    assert events[0].p_value is not None and events[0].p_value < 0.01


def test_ks_does_not_fire_for_same_distribution():
    '''Both windows from the same distribution -> no drift.'''
    rng = np.random.default_rng(0)
    det = DriftDetector(method='ks', reference_window_size=100,
                        recent_window_size=100, threshold=0.001)

    events = [det.update(v) for v in rng.normal(0.0, 1.0, 500)]
    assert all(e is None for e in events)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    rng = np.random.default_rng(0)
    det = DriftDetector(method='page_hinkley', reference_window_size=20,
                        threshold=5.0, delta=0.5)
    for v in rng.normal(0.0, 1.0, 60):
        det.update(v)

    det.reset()
    assert det._total_seen == 0
    assert det._n == 0
    assert det._cum_pos == 0.0
    assert det._cum_neg == 0.0
    assert len(det._buffer) == 0


# ---------------------------------------------------------------------------
# StreamingPipeline integration
# ---------------------------------------------------------------------------

def test_pipeline_emits_drift_alert_distinct_from_anomaly():
    '''
    A stream that's stationary then mean-shifted should produce both a
    point-anomaly alert (the injected spike) and a separate drift alert,
    tracked in distinct collections.
    '''
    rng = np.random.default_rng(0)
    train_data = rng.normal(100.0, 5.0, 300)

    drift = DriftDetector(method='page_hinkley', reference_window_size=50,
                          threshold=10.0, delta=0.5)
    pipeline = StreamingPipeline(detector_type='ensemble', drift_detector=drift)
    pipeline.train(train_data)

    stationary = rng.normal(100.0, 5.0, 200)
    shifted = rng.normal(130.0, 5.0, 200)        # baseline moves up
    stream = np.concatenate([stationary, shifted])
    stream[100] = 350.0                          # lone point spike

    results = pipeline.process_stream(stream)

    # Point anomaly and drift live in separate, populated collections.
    assert len(pipeline.anomalies) >= 1
    assert len(pipeline.drift_events) >= 1
    assert 'drift_events' in results

    # Every result record carries both flags, and they are independent keys.
    rec = results['results'][0]
    assert 'is_anomaly' in rec and 'drift_alert' in rec


def test_pipeline_without_drift_detector_has_no_drift_events():
    '''Default pipeline (no drift detector) stays backward-compatible.'''
    rng = np.random.default_rng(1)
    pipeline = StreamingPipeline(detector_type='statistical')
    pipeline.train(rng.normal(50.0, 2.0, 200))
    results = pipeline.process_stream(rng.normal(50.0, 2.0, 100))

    assert results['drift_events'] == []
    assert pipeline.drift_events == []
    assert all(rec['drift_alert'] is False for rec in results['results'])
