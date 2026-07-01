"""
Optional wake-word gate ("Hey Jarvis" etc.) via openWakeWord.

When disabled (default), the assistant is always-on: any transcribed speech is
treated as directed at it. When enabled, a spoken utterance only triggers a
response if the wake word fired recently — this is how real glasses avoid
answering every overheard conversation.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from core.config import WakeWordConfig
from core.logging_setup import get_logger
from speech.audio_stream import pcm_to_float

log = get_logger("audio.wakeword")


class WakeWord:
    def __init__(self, cfg: WakeWordConfig, sample_rate: int) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self._model = None
        self._last_fire = 0.0
        self._buf = np.zeros(0, dtype=np.float32)
        if cfg.enabled:
            self._load()

    def _load(self) -> None:
        try:
            from openwakeword.model import Model
            self._model = Model(wakeword_models=[self.cfg.model])
            log.info("wake word active: '%s'", self.cfg.model)
        except Exception as e:  # noqa
            log.warning("openWakeWord unavailable (%s); running always-on", e)
            self._model = None
            self.cfg.enabled = False

    def feed(self, frame: bytes) -> None:
        """Feed audio frames continuously so the detector stays warm."""
        if not self.cfg.enabled or self._model is None:
            return
        pcm = (pcm_to_float(frame) * 32768).astype(np.int16)
        try:
            scores = self._model.predict(pcm)
            if any(v >= self.cfg.threshold for v in scores.values()):
                self._last_fire = time.time()
                log.info("🔔 wake word detected")
        except Exception:
            pass

    def is_active(self, window_s: float = 6.0) -> bool:
        """True if the assistant should respond to the current utterance."""
        if not self.cfg.enabled:
            return True  # always-on
        return (time.time() - self._last_fire) <= window_s
