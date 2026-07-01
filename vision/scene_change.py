"""
Scene-change detection — the key vision cost/latency optimisation.

Running a VLM on every camera frame is impossibly slow and expensive. Instead
we compute a cheap perceptual difference between consecutive frames (downscaled
grayscale + colour histogram) and only trigger a VLM call when the scene has
meaningfully changed. This cuts VLM calls by 10-100x in static scenes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class SceneChangeDetector:
    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None

    def _features(self, frame):
        import cv2
        small = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        return gray, hist

    def changed(self, frame) -> float:
        """Return a change score in [0,1]; caller compares to its threshold."""
        gray, hist = self._features(frame)
        if self._prev_gray is None:
            self._prev_gray, self._prev_hist = gray, hist
            return 1.0  # first frame always triggers

        # Mean absolute pixel difference (structural change).
        struct = float(np.mean(np.abs(gray - self._prev_gray)))
        # Histogram correlation (colour/content change); 1=identical.
        import cv2
        corr = cv2.compareHist(
            self._prev_hist.astype(np.float32),
            hist.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        color = 1.0 - max(0.0, corr)
        score = min(1.0, 0.6 * struct * 4 + 0.4 * color)

        self._prev_gray, self._prev_hist = gray, hist
        return score

    def is_significant(self, frame) -> bool:
        return self.changed(frame) >= self.threshold
