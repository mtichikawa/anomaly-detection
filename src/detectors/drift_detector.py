"""
Concept drift detection for streaming data.

Where the point-anomaly detectors in pipeline.py ask "does this single
value look weird relative to the baseline?", the drift detector asks a
different question: "has the baseline itself moved?" Production models
decay because the underlying distribution changes over time (concept
drift) — what was normal last month silently becomes miscalibrated.

Two complementary univariate methods, selected via the `method` arg:

- **page_hinkley** — a cumulative-sum test for mean shift. It tracks the
  running deviation of each value from the stream mean and fires when the
  accumulated deviation crosses a threshold. Streaming-native, low memory,
  good for gradual or step changes in the mean. Deviations are
  standardized by the running std so the threshold is scale-free.

- **ks** — a two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp)
  comparing a reference window (older values) against a recent window
  (newer values). Fires when the two empirical distributions differ
  significantly. Catches shape changes a mean-shift test misses
  (variance changes, bimodality), at slightly higher cost.

Both return a `DriftEvent` dataclass when drift is detected, else None.
After firing, the detector re-baselines so it reports one clean event per
drift instead of alarming on every subsequent point.

Follow-ups (not implemented here, see README): automatic retraining of
the point-anomaly detectors on drift, and multivariate drift (e.g. MMD).
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Deque, Optional

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class DriftEvent:
    """
    Emitted when the drift detector decides the distribution has changed.

    Attributes:
        index: position in the stream (0-based) when drift fired.
        method: 'page_hinkley' or 'ks'.
        statistic: the test statistic that crossed the threshold
            (Page-Hinkley cumulative sum, or KS D statistic).
        threshold: the configured cutoff it crossed (a magnitude for
            page_hinkley, a p-value significance level for ks).
        p_value: the KS test p-value; None for page_hinkley.
        detail: human-readable one-liner for logs / demos.
    """

    index: int
    method: str
    statistic: float
    threshold: float
    p_value: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> dict:
        """Plain-dict view (JSON-friendly) for results payloads."""
        return asdict(self)


class DriftDetector:
    """
    Univariate concept-drift detector over a streaming sequence of values.

    Args:
        method: 'page_hinkley' (mean shift, CUSUM-based) or 'ks'
            (distributional change, two-sample KS).
        reference_window_size: for 'ks', the number of older values used
            as the reference distribution. For 'page_hinkley', the warmup
            length — the detector establishes the baseline mean/std over
            this many points before it can fire.
        recent_window_size: for 'ks', the number of newer values compared
            against the reference. Ignored by 'page_hinkley'.
        threshold: detection cutoff. For 'page_hinkley' this is the
            cumulative-sum magnitude (in standardized units) that the
            running deviation must exceed (default 10.0). For 'ks' this is
            the p-value significance level below which the two windows are
            judged different (default 0.01).
        delta: 'page_hinkley' slack term (the CUSUM "k" parameter, in std
            units). A small per-step allowance that absorbs noise so the
            cumulative sum doesn't random-walk into a false alarm.
            Ignored by 'ks'.
    """

    VALID_METHODS = ("page_hinkley", "ks")

    def __init__(
        self,
        method: str = "page_hinkley",
        reference_window_size: int = 100,
        recent_window_size: int = 50,
        threshold: Optional[float] = None,
        delta: float = 0.5,
    ):
        method = method.lower()
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of {self.VALID_METHODS}, got {method!r}"
            )

        self.method = method
        self.reference_window_size = reference_window_size
        self.recent_window_size = recent_window_size
        self.delta = delta

        if threshold is None:
            threshold = 10.0 if method == "page_hinkley" else 0.01
        self.threshold = threshold

        # deque is created once; reset()/_rebaseline() clear it in place.
        self._buffer: Deque[float] = deque(
            maxlen=reference_window_size + recent_window_size
        )
        self.reset()

    # ── public interface ────────────────────────────────────────────

    def update(self, value: float) -> Optional[DriftEvent]:
        """
        Feed one value. Returns a DriftEvent if drift fired, else None.
        """
        value = float(value)
        self._total_seen += 1
        if self.method == "page_hinkley":
            return self._update_page_hinkley(value)
        return self._update_ks(value)

    def reset(self):
        """Clear all state, including the global stream counter."""
        self._total_seen = 0
        self._rebaseline()

    # ── internal ────────────────────────────────────────────────────

    def _rebaseline(self):
        """
        Reset the test statistics so a new baseline is learned, without
        zeroing the global stream counter. Called on construction and
        after a DriftEvent fires (re-calibrate to the post-drift regime).
        """
        # Page-Hinkley: Welford running mean/variance + two-sided CUSUM.
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._cum_pos = 0.0
        self._cum_neg = 0.0
        # KS: rolling buffer of the most recent reference+recent values.
        self._buffer.clear()

    def _update_page_hinkley(self, value: float) -> Optional[DriftEvent]:
        # Welford incremental mean/variance.
        self._n += 1
        delta1 = value - self._mean
        self._mean += delta1 / self._n
        self._m2 += delta1 * (value - self._mean)

        # Warm up the baseline before testing.
        if self._n <= self.reference_window_size:
            return None

        std = sqrt(self._m2 / self._n)
        if std < 1e-9:
            return None

        # Standardized deviation from the running mean.
        z = (value - self._mean) / std

        # Two-sided reflected CUSUM: one accumulator per drift direction.
        self._cum_pos = max(0.0, self._cum_pos + z - self.delta)
        self._cum_neg = max(0.0, self._cum_neg - z - self.delta)
        ph = max(self._cum_pos, self._cum_neg)

        if ph > self.threshold:
            direction = "increase" if self._cum_pos >= self._cum_neg else "decrease"
            event = DriftEvent(
                index=self._total_seen - 1,
                method="page_hinkley",
                statistic=float(ph),
                threshold=self.threshold,
                detail=(
                    f"Page-Hinkley mean {direction}: cumulative deviation "
                    f"{ph:.2f} exceeded threshold {self.threshold:.2f}"
                ),
            )
            self._rebaseline()
            return event
        return None

    def _update_ks(self, value: float) -> Optional[DriftEvent]:
        self._buffer.append(value)
        need = self.reference_window_size + self.recent_window_size
        if len(self._buffer) < need:
            return None

        buf = np.asarray(self._buffer, dtype=float)
        reference = buf[: self.reference_window_size]
        recent = buf[self.reference_window_size :]

        stat, p_value = ks_2samp(reference, recent)

        if p_value < self.threshold:
            event = DriftEvent(
                index=self._total_seen - 1,
                method="ks",
                statistic=float(stat),
                threshold=self.threshold,
                p_value=float(p_value),
                detail=(
                    f"KS distributional change: D={stat:.3f}, "
                    f"p={p_value:.2e} < alpha {self.threshold:g}"
                ),
            )
            self._rebaseline()
            return event
        return None
