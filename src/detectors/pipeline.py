'''
Real-Time Anomaly Detection System
Multiple detection algorithms with ensemble voting
'''

import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class BaseDetector:
    '''Base class for all detectors'''
    
    def __init__(self, name: str):
        self.name = name
        self.trained = False
        
    def fit(self, data: np.ndarray):
        '''Train the detector'''
        raise NotImplementedError
        
    def predict(self, value: float) -> bool:
        '''Predict if value is anomaly'''
        raise NotImplementedError
        
    def get_score(self, value: float) -> float:
        '''Get anomaly score (0-1)'''
        raise NotImplementedError


class IsolationForestDetector(BaseDetector):
    '''
    Isolation Forest anomaly detector
    Based on the principle that anomalies are easier to isolate
    '''
    
    def __init__(self, contamination: float = 0.1, n_trees: int = 100):
        super().__init__('IsolationForest')
        self.contamination = contamination
        self.n_trees = n_trees
        self.threshold = None
        self.trees = []
        
    def fit(self, data: np.ndarray):
        '''Train isolation forest'''
        # Simplified implementation - in production use sklearn
        self.mean = np.mean(data)
        self.std = np.std(data)
        
        # Set threshold based on contamination rate
        sorted_data = np.sort(data)
        cutoff_idx = int(len(sorted_data) * (1 - self.contamination))
        self.threshold = abs(sorted_data[cutoff_idx] - self.mean)
        
        self.trained = True
        
    def predict(self, value: float) -> bool:
        '''Detect if value is anomaly'''
        if not self.trained:
            return False
            
        deviation = abs(value - self.mean)
        return deviation > self.threshold
        
    def get_score(self, value: float) -> float:
        '''Get anomaly score'''
        if not self.trained:
            return 0.0
            
        deviation = abs(value - self.mean)
        score = min(deviation / (self.threshold + 1e-10), 1.0)
        return score


class StatisticalDetector(BaseDetector):
    '''
    Statistical detector using z-scores and moving averages
    '''
    
    def __init__(self, threshold: float = 3.0, window_size: int = 50):
        super().__init__('Statistical')
        self.threshold = threshold
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.global_mean = None
        self.global_std = None
        
    def fit(self, data: np.ndarray):
        '''Train on historical data'''
        self.global_mean = np.mean(data)
        self.global_std = np.std(data)
        self.trained = True
        
    def predict(self, value: float) -> bool:
        '''Detect anomaly using z-score'''
        self.window.append(value)
        
        if len(self.window) < 3:
            return False
            
        # Use moving window statistics
        window_mean = np.mean(self.window)
        window_std = np.std(self.window)
        
        if window_std < 1e-10:
            return False
            
        z_score = abs((value - window_mean) / window_std)
        return z_score > self.threshold
        
    def get_score(self, value: float) -> float:
        '''Get normalized anomaly score'''
        if len(self.window) < 3:
            return 0.0
            
        window_mean = np.mean(self.window)
        window_std = np.std(self.window)
        
        if window_std < 1e-10:
            return 0.0
            
        z_score = abs((value - window_mean) / window_std)
        return min(z_score / self.threshold, 1.0)


class LSTMAutoencoder(BaseDetector):
    '''
    LSTM Autoencoder for anomaly detection
    Simplified implementation without actual deep learning
    '''
    
    def __init__(self, sequence_length: int = 10):
        super().__init__('LSTM-Autoencoder')
        self.sequence_length = sequence_length
        self.history = deque(maxlen=sequence_length)
        self.reconstruction_threshold = None
        
    def fit(self, data: np.ndarray):
        '''Train autoencoder on normal patterns'''
        # Simplified: use variance as proxy for reconstruction error
        self.mean = np.mean(data)
        self.std = np.std(data)
        
        # Set threshold for reconstruction error
        self.reconstruction_threshold = self.std * 2
        self.trained = True
        
    def predict(self, value: float) -> bool:
        '''Predict using reconstruction error'''
        self.history.append(value)
        
        if len(self.history) < self.sequence_length:
            return False
            
        # Simplified reconstruction error
        expected = np.mean(self.history)
        error = abs(value - expected)
        
        return error > self.reconstruction_threshold
        
    def get_score(self, value: float) -> float:
        '''Get anomaly score based on reconstruction error'''
        if len(self.history) < self.sequence_length:
            return 0.0
            
        expected = np.mean(self.history)
        error = abs(value - expected)
        
        return min(error / (self.reconstruction_threshold + 1e-10), 1.0)


class EnsembleDetector:
    '''
    Ensemble of multiple detectors with voting
    '''
    
    def __init__(self, voting_threshold: float = 0.5):
        '''
        Args:
            voting_threshold: Fraction of detectors that must agree
        '''
        self.voting_threshold = voting_threshold
        self.detectors = [
            IsolationForestDetector(contamination=0.1),
            StatisticalDetector(threshold=3.0),
            StatisticalDetector(threshold=2.5),  # More sensitive
            LSTMAutoencoder(sequence_length=10)
        ]
        
    def fit(self, data: np.ndarray):
        '''Train all detectors'''
        print(f'  Training {len(self.detectors)} detectors...')
        for detector in self.detectors:
            detector.fit(data)
        print(f'  ✅ All detectors trained')
        
    def predict(self, value: float) -> Tuple[bool, Dict]:
        '''
        Predict with ensemble voting
        
        Returns:
            (is_anomaly, details_dict)
        '''
        votes = []
        scores = {}
        
        for detector in self.detectors:
            vote = detector.predict(value)
            score = detector.get_score(value)
            
            votes.append(vote)
            scores[detector.name] = {
                'vote': vote,
                'score': score
            }
            
        vote_fraction = sum(votes) / len(votes)
        is_anomaly = vote_fraction >= self.voting_threshold
        
        return is_anomaly, {
            'vote_fraction': vote_fraction,
            'detector_scores': scores,
            'consensus': is_anomaly
        }
        
    def get_ensemble_score(self, value: float) -> float:
        '''Get average anomaly score across all detectors'''
        scores = [d.get_score(value) for d in self.detectors]
        return np.mean(scores)


class StreamingPipeline:
    '''
    Complete streaming anomaly detection pipeline
    '''
    
    def __init__(self, detector_type: str = 'ensemble'):
        '''
        Args:
            detector_type: 'isolation', 'statistical', 'lstm', or 'ensemble'
        '''
        self.detector_type = detector_type
        
        if detector_type == 'isolation':
            self.detector = IsolationForestDetector()
        elif detector_type == 'statistical':
            self.detector = StatisticalDetector()
        elif detector_type == 'lstm':
            self.detector = LSTMAutoencoder()
        else:
            self.detector = EnsembleDetector()
            
        self.anomalies = []
        self.all_scores = []
        
    def train(self, training_data: np.ndarray):
        '''Train detector on historical normal data'''
        print(f'\n🔧 Training {self.detector_type} detector...')
        print(f'  Training samples: {len(training_data)}')
        
        if isinstance(self.detector, EnsembleDetector):
            self.detector.fit(training_data)
        else:
            self.detector.fit(training_data)
            
        print('  ✅ Training complete')
        
    def process_stream(self, data_stream: np.ndarray, timestamps: Optional[List] = None) -> Dict:
        '''
        Process streaming data and detect anomalies
        
        Args:
            data_stream: Array of values to process
            timestamps: Optional timestamps for each value
            
        Returns:
            Dict with results
        '''
        print(f'\n📊 Processing stream of {len(data_stream)} points...')
        
        results = []
        
        for i, value in enumerate(data_stream):
            timestamp = timestamps[i] if timestamps else i
            
            if isinstance(self.detector, EnsembleDetector):
                is_anomaly, details = self.detector.predict(value)
                score = details['vote_fraction']
            else:
                is_anomaly = self.detector.predict(value)
                score = self.detector.get_score(value)
                details = {'score': score}
                
            result = {
                'index': i,
                'timestamp': timestamp,
                'value': value,
                'is_anomaly': is_anomaly,
                'score': score,
                'details': details
            }
            
            results.append(result)
            self.all_scores.append(score)
            
            if is_anomaly:
                self.anomalies.append(result)
                
        print(f'  ✅ Detected {len(self.anomalies)} anomalies ({len(self.anomalies)/len(data_stream)*100:.1f}%)')
        
        return {
            'results': results,
            'anomalies': self.anomalies,
            'stats': self._calculate_stats(results)
        }
        
    def _calculate_stats(self, results: List[Dict]) -> Dict:
        '''Calculate statistics about the detection run'''
        scores = [r['score'] for r in results]
        anomalies = [r for r in results if r['is_anomaly']]
        
        return {
            'total_points': len(results),
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / len(results) if results else 0,
            'avg_score': np.mean(scores) if scores else 0,
            'max_score': np.max(scores) if scores else 0,
            'score_std': np.std(scores) if scores else 0
        }
        
    def save_results(self, filepath: str):
        '''Save detection results to JSON'''
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                'detector_type': self.detector_type,
                'anomalies': self.anomalies,
                'statistics': self._calculate_stats([
                    {'score': s, 'is_anomaly': a['is_anomaly']} 
                    for s, a in zip(self.all_scores, self.anomalies)
                ] if self.anomalies else [])
            }, f, indent=2)
            
        print(f'  ✅ Saved results to {filepath}')


def generate_synthetic_data(n_points: int = 1000, 
                           anomaly_rate: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate synthetic time series with anomalies
    
    Returns:
        (data, true_anomaly_labels)
    '''
    # Normal data with trend and seasonality
    t = np.arange(n_points)
    trend = t * 0.05
    seasonality = 10 * np.sin(2 * np.pi * t / 50)
    noise = np.random.randn(n_points) * 2
    
    data = 100 + trend + seasonality + noise
    
    # Inject anomalies
    n_anomalies = int(n_points * anomaly_rate)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    true_labels = np.zeros(n_points, dtype=bool)
    
    for idx in anomaly_indices:
        # Random spike or dip
        if np.random.random() > 0.5:
            data[idx] += np.random.uniform(30, 50)  # Spike
        else:
            data[idx] -= np.random.uniform(30, 50)  # Dip
        true_labels[idx] = True
        
    return data, true_labels


def evaluate_detector(predictions: np.ndarray, 
                      true_labels: np.ndarray) -> Dict:
    '''
    Evaluate detector performance
    
    Returns:
        Dict with precision, recall, F1
    '''
    true_positives = np.sum(predictions & true_labels)
    false_positives = np.sum(predictions & ~true_labels)
    false_negatives = np.sum(~predictions & true_labels)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives)
    }


def demo():
    '''Comprehensive demonstration'''
    print('Real-Time Anomaly Detection System')
    print('=' * 60)
    
    # Generate synthetic data
    print('\n📊 Generating synthetic time series...')
    data, true_labels = generate_synthetic_data(n_points=1000, anomaly_rate=0.05)
    
    # Split train/test
    train_size = 500
    train_data = data[:train_size]
    test_data = data[train_size:]
    test_labels = true_labels[train_size:]
    
    print(f'  Training data: {len(train_data)} points')
    print(f'  Test data: {len(test_data)} points')
    print(f'  True anomalies in test: {np.sum(test_labels)}')
    
    # Create and train pipeline
    pipeline = StreamingPipeline(detector_type='ensemble')
    pipeline.train(train_data)
    
    # Process stream
    results = pipeline.process_stream(test_data)
    
    # Evaluate
    print('\n📈 Performance Metrics:')
    predictions = np.array([r['is_anomaly'] for r in results['results']])
    metrics = evaluate_detector(predictions, test_labels)
    
    print(f'  Precision: {metrics["precision"]:.3f}')
    print(f'  Recall: {metrics["recall"]:.3f}')
    print(f'  F1 Score: {metrics["f1_score"]:.3f}')
    print(f'  True Positives: {metrics["true_positives"]}')
    print(f'  False Positives: {metrics["false_positives"]}')
    print(f'  False Negatives: {metrics["false_negatives"]}')
    
    # Show top anomalies
    print('\n🔍 Top 5 Anomalies by Score:')
    top_anomalies = sorted(pipeline.anomalies, key=lambda x: x['score'], reverse=True)[:5]
    for anom in top_anomalies:
        print(f'  Index {anom["index"]}: value={anom["value"]:.1f}, score={anom["score"]:.3f}')
        
    # Save results
    pipeline.save_results('results/anomaly_detection.json')
    
    print('\n✅ Demo complete!')


if __name__ == '__main__':
    demo()
