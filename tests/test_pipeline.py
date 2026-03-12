'''
Smoke tests for anomaly detection pipeline.
Verifies that all detectors and the streaming pipeline
can be instantiated, trained, and run without errors.
'''

import json
import numpy as np
import pytest
from pathlib import Path

from detectors.pipeline import (
    IsolationForestDetector,
    StatisticalDetector,
    LSTMAutoencoder,
    EnsembleDetector,
    StreamingPipeline,
    generate_synthetic_data,
    evaluate_detector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_data():
    '''Small normally-distributed training array.'''
    np.random.seed(42)
    return np.random.randn(200) * 5 + 100


@pytest.fixture
def stream_data():
    '''Short stream with obvious outliers at indices 50 and 75.'''
    np.random.seed(0)
    data = np.random.randn(100) * 2 + 50
    data[50] += 80   # spike
    data[75] -= 80   # dip
    return data


# ---------------------------------------------------------------------------
# generate_synthetic_data
# ---------------------------------------------------------------------------

def test_generate_synthetic_data_shape():
    data, labels = generate_synthetic_data(n_points=200, anomaly_rate=0.05)
    assert data.shape == (200,)
    assert labels.shape == (200,)
    assert labels.dtype == bool


def test_generate_synthetic_data_anomaly_count():
    np.random.seed(1)
    data, labels = generate_synthetic_data(n_points=500, anomaly_rate=0.1)
    assert labels.sum() == 50


def test_generate_synthetic_data_anomalies_are_outliers():
    np.random.seed(2)
    data, labels = generate_synthetic_data(n_points=300, anomaly_rate=0.05)
    normal_std = data[~labels].std()
    for idx in np.where(labels)[0]:
        deviation = abs(data[idx] - data[~labels].mean())
        assert deviation > normal_std, f'Injected anomaly at {idx} not far enough from normal'


# ---------------------------------------------------------------------------
# evaluate_detector
# ---------------------------------------------------------------------------

def test_evaluate_detector_perfect():
    labels = np.array([True, False, True, False])
    preds  = np.array([True, False, True, False])
    m = evaluate_detector(preds, labels)
    assert m['precision'] == 1.0
    assert m['recall'] == 1.0
    assert m['f1_score'] == 1.0
    assert m['true_positives'] == 2
    assert m['false_positives'] == 0
    assert m['false_negatives'] == 0


def test_evaluate_detector_all_wrong():
    labels = np.array([True, True, False, False])
    preds  = np.array([False, False, True, True])
    m = evaluate_detector(preds, labels)
    assert m['precision'] == 0.0
    assert m['recall'] == 0.0
    assert m['true_positives'] == 0
    assert m['false_positives'] == 2
    assert m['false_negatives'] == 2


def test_evaluate_detector_no_positives_predicted():
    labels = np.array([True, False, True])
    preds  = np.array([False, False, False])
    m = evaluate_detector(preds, labels)
    assert m['precision'] == 0.0
    assert m['recall'] == 0.0
    assert m['f1_score'] == 0.0


# ---------------------------------------------------------------------------
# IsolationForestDetector
# ---------------------------------------------------------------------------

def test_isolation_forest_predict_before_fit_returns_false():
    det = IsolationForestDetector()
    assert det.predict(999.0) is False


def test_isolation_forest_score_before_fit_returns_zero():
    det = IsolationForestDetector()
    assert det.get_score(999.0) == 0.0


def test_isolation_forest_fit_marks_trained(normal_data):
    det = IsolationForestDetector()
    det.fit(normal_data)
    assert det.trained is True


def test_isolation_forest_clear_outlier_detected(normal_data):
    det = IsolationForestDetector(contamination=0.05)
    det.fit(normal_data)
    # Normal values should not be flagged, extreme outlier should be
    assert not det.predict(normal_data.mean())
    assert det.predict(normal_data.mean() + 200)


def test_isolation_forest_score_range(normal_data):
    det = IsolationForestDetector()
    det.fit(normal_data)
    for val in [90.0, 100.0, 110.0, 300.0, -100.0]:
        score = det.get_score(val)
        assert 0.0 <= score <= 1.0, f'Score {score} out of [0,1] for value {val}'


# ---------------------------------------------------------------------------
# StatisticalDetector
# ---------------------------------------------------------------------------

def test_statistical_detector_returns_false_for_short_window(normal_data):
    det = StatisticalDetector()
    det.fit(normal_data)
    # Only 2 values fed — window < 3, should return False
    assert det.predict(100.0) is False
    assert det.predict(101.0) is False


def test_statistical_detector_catches_spike(normal_data):
    det = StatisticalDetector(threshold=2.0, window_size=10)
    det.fit(normal_data)
    # Seed window with slightly varied values so std > 0
    for v in [98.0, 100.0, 102.0, 99.0, 101.0, 100.5, 99.5, 100.0, 101.5, 98.5]:
        det.predict(v)
    # Now feed an extreme spike — should be flagged
    assert det.predict(200.0)


def test_statistical_detector_score_range(normal_data):
    det = StatisticalDetector()
    det.fit(normal_data)
    for v in [99.0, 100.0, 101.0]:
        det.predict(v)
    score = det.get_score(100.0)
    assert 0.0 <= score <= 1.0


def test_statistical_detector_custom_name():
    det = StatisticalDetector(name='MyDetector')
    assert det.name == 'MyDetector'


# ---------------------------------------------------------------------------
# LSTMAutoencoder
# ---------------------------------------------------------------------------

def test_lstm_predict_before_sequence_full(normal_data):
    det = LSTMAutoencoder(sequence_length=10)
    det.fit(normal_data)
    # Feed fewer values than sequence_length
    for _ in range(9):
        assert det.predict(100.0) is False


def test_lstm_catches_spike_after_warmup(normal_data):
    det = LSTMAutoencoder(sequence_length=5)
    det.fit(normal_data)
    for v in [100.0] * 5:
        det.predict(v)
    assert det.predict(500.0)


def test_lstm_score_range(normal_data):
    det = LSTMAutoencoder(sequence_length=5)
    det.fit(normal_data)
    for v in [100.0] * 5:
        det.predict(v)
    for val in [98.0, 105.0, 400.0]:
        score = det.get_score(val)
        assert 0.0 <= score <= 1.0, f'Score {score} out of [0,1]'


# ---------------------------------------------------------------------------
# EnsembleDetector
# ---------------------------------------------------------------------------

def test_ensemble_fit_trains_all_detectors(normal_data):
    ens = EnsembleDetector()
    ens.fit(normal_data)
    for det in ens.detectors:
        assert det.trained is True


def test_ensemble_predict_returns_tuple(normal_data):
    ens = EnsembleDetector()
    ens.fit(normal_data)
    result = ens.predict(100.0)
    assert isinstance(result, tuple) and len(result) == 2
    is_anomaly, details = result
    assert isinstance(is_anomaly, (bool, np.bool_))
    assert 'vote_fraction' in details
    assert 'detector_scores' in details


def test_ensemble_vote_fraction_range(normal_data):
    ens = EnsembleDetector()
    ens.fit(normal_data)
    for v in [100.0] * 10:
        _, details = ens.predict(v)
        assert 0.0 <= details['vote_fraction'] <= 1.0


def test_ensemble_flags_extreme_outlier(normal_data):
    ens = EnsembleDetector(voting_threshold=0.5)
    ens.fit(normal_data)
    for v in [100.0] * 15:
        ens.predict(v)
    is_anomaly, _ = ens.predict(normal_data.mean() + 500)
    assert is_anomaly


def test_ensemble_score_range(normal_data):
    ens = EnsembleDetector()
    ens.fit(normal_data)
    score = ens.get_ensemble_score(100.0)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# StreamingPipeline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('detector_type', ['isolation', 'statistical', 'lstm', 'ensemble'])
def test_pipeline_all_detector_types_run(normal_data, stream_data, detector_type):
    pipeline = StreamingPipeline(detector_type=detector_type)
    pipeline.train(normal_data)
    results = pipeline.process_stream(stream_data)
    assert 'results' in results
    assert 'anomalies' in results
    assert 'stats' in results
    assert len(results['results']) == len(stream_data)


def test_pipeline_stats_keys(normal_data, stream_data):
    pipeline = StreamingPipeline(detector_type='isolation')
    pipeline.train(normal_data)
    results = pipeline.process_stream(stream_data)
    stats = results['stats']
    for key in ('total_points', 'anomalies_detected', 'anomaly_rate', 'avg_score', 'max_score'):
        assert key in stats, f'Missing stats key: {key}'


def test_pipeline_result_record_keys(normal_data, stream_data):
    pipeline = StreamingPipeline(detector_type='statistical')
    pipeline.train(normal_data)
    results = pipeline.process_stream(stream_data)
    for rec in results['results']:
        for key in ('index', 'timestamp', 'value', 'is_anomaly', 'score'):
            assert key in rec


def test_pipeline_detects_at_least_one_anomaly(normal_data, stream_data):
    '''stream_data has two ±80-sigma spikes; pipeline should catch at least one.'''
    pipeline = StreamingPipeline(detector_type='ensemble')
    pipeline.train(normal_data)
    results = pipeline.process_stream(stream_data)
    assert results['stats']['anomalies_detected'] >= 1


def test_pipeline_anomaly_rate_between_0_and_1(normal_data, stream_data):
    pipeline = StreamingPipeline(detector_type='isolation')
    pipeline.train(normal_data)
    results = pipeline.process_stream(stream_data)
    rate = results['stats']['anomaly_rate']
    assert 0.0 <= rate <= 1.0


def test_pipeline_save_results_empty_anomalies(tmp_path, normal_data):
    '''save_results() succeeds when no anomalies were detected (avoids np.bool_ JSON bug).'''
    pipeline = StreamingPipeline(detector_type='isolation')
    pipeline.train(normal_data)
    # Don't call process_stream — anomalies list stays empty
    out = tmp_path / 'results' / 'test_output.json'
    pipeline.save_results(str(out))
    assert out.exists()
    with open(out) as f:
        data = json.load(f)
    assert data['detector_type'] == 'isolation'
    assert data['anomalies'] == []


@pytest.mark.xfail(
    raises=TypeError,
    reason='Anomaly records contain np.bool_ which json.dump cannot serialize — '
           'tracked in maintenance_patches.json (fix_save_results_stats_call)',
    strict=True,
)
def test_pipeline_save_results_with_anomalies_known_bug(tmp_path, normal_data, stream_data):
    '''Documents the known np.bool_ serialization bug in save_results().'''
    pipeline = StreamingPipeline(detector_type='isolation')
    pipeline.train(normal_data)
    pipeline.process_stream(stream_data)
    assert len(pipeline.anomalies) > 0, 'Need anomalies to trigger the bug'
    pipeline.save_results(str(tmp_path / 'out.json'))


def test_pipeline_timestamps_propagated(normal_data, stream_data):
    pipeline = StreamingPipeline(detector_type='statistical')
    pipeline.train(normal_data)
    timestamps = [f't{i}' for i in range(len(stream_data))]
    results = pipeline.process_stream(stream_data, timestamps=timestamps)
    for rec in results['results']:
        assert rec['timestamp'] == timestamps[rec['index']]
