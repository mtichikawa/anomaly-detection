'''Quick Anomaly Detection Demo'''
import sys
sys.path.append('../src')
from detectors.pipeline import StreamingPipeline, generate_synthetic_data, evaluate_detector
import numpy as np

print('Real-Time Anomaly Detection Demo')
print('=' * 50)

# Generate data
print('\nGenerating synthetic time series...')
data, labels = generate_synthetic_data(n_points=1000, anomaly_rate=0.05)
print(f'  Created {len(data)} data points with {np.sum(labels)} true anomalies')

# Split train/test
train_data = data[:500]
test_data = data[500:]
test_labels = labels[500:]

# Train detector
print('\nTraining ensemble detector...')
pipeline = StreamingPipeline(detector_type='ensemble')
pipeline.train(train_data)

# Detect anomalies
print('\nProcessing stream...')
results = pipeline.process_stream(test_data)

# Evaluate
predictions = np.array([r['is_anomaly'] for r in results['results']])
metrics = evaluate_detector(predictions, test_labels)

print(f'\n📊 Performance:')
print(f'  Precision: {metrics["precision"]:.3f}')
print(f'  Recall: {metrics["recall"]:.3f}')
print(f'  F1 Score: {metrics["f1_score"]:.3f}')

print(f'\n🔍 Detection Summary:')
print(f'  Anomalies detected: {len(pipeline.anomalies)}')
print(f'  True positives: {metrics["true_positives"]}')
print(f'  False positives: {metrics["false_positives"]}')

print('\n✅ Demo complete!')
