'''
Concept-drift demo.

Builds a synthetic stream that is stationary for ~1000 points and then
takes a step shift in its mean. A StreamingPipeline runs the point-anomaly
ensemble AND a Page-Hinkley drift detector at the same time, so you can see
the two layers do different jobs:

  - point anomalies  -> the injected one-off spikes/dips ("this value is weird")
  - concept drift     -> the sustained baseline shift ("the normal has moved")
'''
import sys
sys.path.append('../src')

import numpy as np
from detectors.pipeline import StreamingPipeline
from detectors.drift_detector import DriftDetector

print('Concept-Drift Detection Demo')
print('=' * 50)

# --- Build the stream --------------------------------------------------
rng = np.random.default_rng(0)
n_stationary = 1000
n_drifted = 500

stationary = rng.normal(50.0, 2.0, n_stationary)   # baseline regime
drifted = rng.normal(58.0, 2.0, n_drifted)         # baseline shifts up +8
stream = np.concatenate([stationary, drifted])

# Inject a handful of one-off point outliers in the stationary region.
# ~4 sigma spikes: the ensemble flags them, but a single spike does not
# trip the drift detector — that is the whole point of the separation.
spike_indices = [200, 450, 700]
for idx in spike_indices:
    stream[idx] += 12.0

print(f'\nStream: {len(stream)} points '
      f'(stationary 0-{n_stationary - 1}, drifted {n_stationary}-{len(stream) - 1})')
print(f'Injected point spikes at indices: {spike_indices}')
print(f'Baseline mean shifts ~50 -> ~58 at index {n_stationary}')

# --- Train on the early stationary regime ------------------------------
train_data = stationary[:500]

drift_detector = DriftDetector(
    method='page_hinkley',
    reference_window_size=200,   # warm up the baseline
    threshold=12.0,
    delta=0.5,
)

pipeline = StreamingPipeline(detector_type='ensemble',
                             drift_detector=drift_detector)
pipeline.train(train_data)

# --- Process the full stream -------------------------------------------
results = pipeline.process_stream(stream)

# --- Report ------------------------------------------------------------
print('\n' + '=' * 50)
print('Point anomalies (individual weird values):')
print(f'  {len(pipeline.anomalies)} flagged')
for anom in pipeline.anomalies[:6]:
    print(f'    index {anom["index"]:4d}: value={anom["value"]:.1f}, '
          f'score={anom["score"]:.2f}')

print('\nConcept-drift events (baseline shifted):')
if pipeline.drift_events:
    for ev in pipeline.drift_events:
        d = ev['drift']
        print(f'    index {ev["index"]:4d}: {d["detail"]}')
else:
    print('    (none)')

# A drift event firing shortly after index 1000 is the expected outcome.
drift_after_shift = [e for e in pipeline.drift_events if e['index'] >= n_stationary]
print('\n' + '=' * 50)
if drift_after_shift:
    first = drift_after_shift[0]['index']
    print(f'✅ Drift caught at index {first}, '
          f'{first - n_stationary} points after the shift began.')
else:
    print('⚠️  No drift event after the shift — check thresholds.')

print('\n✅ Demo complete!')
