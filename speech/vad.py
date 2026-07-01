"""
Voice Activity Detection with graceful backend fallback.

Priority: Silero (torch, accurate) -> webrtcvad -> energy gate (always works).
The VAD turns a continuous frame stream into discrete speech *segments*: it
buffers frames while someone is talking and emits the full utterance once a
short silence hangover elapses. This is what lets STT run only on real speech.
"""
from __future__ import annotations

import collections
from typing import Iterator, Optional

import numpy as np

from core.config import AudioConfig, VADConfig
from core.logging_setup import get_logger
from speech.audio_stream import pcm_to_float

log = get_logger("audio.vad")


class _SileroVAD:
    def __init__(self, threshold: float, sample_rate: int) -> None:
        import torch  # noqa
        self.threshold = threshold
        self.sample_rate = sample_rate
        model, _ = _load_silero()
        self.model = model
        self._torch = __import__("torch")

    def is_speech(self, frame: bytes) -> bool:
        audio = pcm_to_float(frame)
        # Silero expects 512 samples @16k; pad/trim to keep it happy.
        need = 512 if self.sample_rate == 16000 else 256
        if len(audio) < need:
            audio = np.pad(audio, (0, need - len(audio)))
        else:
            audio = audio[:need]
        t = self._torch.from_numpy(audio)
        with self._torch.no_grad():
            prob = self.model(t, self.sample_rate).item()
        return prob >= self.threshold


_SILERO_CACHE = None


def _load_silero():
    global _SILERO_CACHE
    if _SILERO_CACHE is None:
        import torch
        _SILERO_CACHE = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True, onnx=False
        )
    return _SILERO_CACHE


class _WebrtcVAD:
    def __init__(self, sample_rate: int, aggressiveness: int = 2) -> None:
        import webrtcvad
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate

    def is_speech(self, frame: bytes) -> bool:
        return self.vad.is_speech(frame, self.sample_rate)


class _EnergyVAD:
    """Dependency-free adaptive energy gate. Works everywhere as a last resort."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.noise_floor = 0.01
        self.k = 2.5  # speech must exceed k * noise floor

    def is_speech(self, frame: bytes) -> bool:
        audio = pcm_to_float(frame)
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
        speech = rms > max(0.015, self.noise_floor * self.k)
        if not speech:  # slowly adapt the noise floor during silence
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms
        return speech


def _make_backend(vcfg: VADConfig, sample_rate: int):
    order = {
        "silero": [_try_silero, _try_webrtc, _try_energy],
        "webrtc": [_try_webrtc, _try_energy],
        "energy": [_try_energy],
    }.get(vcfg.backend, [_try_silero, _try_webrtc, _try_energy])
    for factory in order:
        backend = factory(vcfg, sample_rate)
        if backend is not None:
            log.info("VAD backend: %s", backend.__class__.__name__)
            return backend
    return _EnergyVAD(vcfg.threshold)


def _try_silero(vcfg, sr):
    try:
        return _SileroVAD(vcfg.threshold, sr)
    except Exception as e:  # noqa
        log.warning("Silero VAD unavailable (%s); falling back", e)
        return None


def _try_webrtc(vcfg, sr):
    try:
        return _WebrtcVAD(sr)
    except Exception:
        return None


def _try_energy(vcfg, sr):
    return _EnergyVAD(vcfg.threshold)


class VADSegmenter:
    """Turns a frame stream into complete speech segments (bytes)."""

    def __init__(self, acfg: AudioConfig, vcfg: VADConfig) -> None:
        self.acfg = acfg
        self.backend = _make_backend(vcfg, acfg.sample_rate)
        self.hangover_frames = max(1, acfg.silence_hangover_ms // acfg.frame_ms)
        self.min_frames = max(1, acfg.min_speech_ms // acfg.frame_ms)
        self.max_frames = max(1, acfg.max_speech_ms // acfg.frame_ms)

    def process(self, frame: bytes, state: dict) -> Optional[bytes]:
        """
        Feed one frame. Returns a completed segment (bytes) or None.
        `state` is a caller-owned dict carrying the in-progress buffer.
        """
        buf = state.setdefault("buf", [])
        silence = state.get("silence", 0)
        speaking = state.get("speaking", False)

        try:
            is_speech = self.backend.is_speech(frame)
        except Exception:
            is_speech = _EnergyVAD().is_speech(frame)

        if is_speech:
            if not speaking:
                state["speaking"] = True
                state["started"] = True  # signal for barge-in
            buf.append(frame)
            state["silence"] = 0
        elif speaking:
            buf.append(frame)
            silence += 1
            state["silence"] = silence

        segment = None
        end_by_silence = speaking and silence >= self.hangover_frames
        end_by_maxlen = len(buf) >= self.max_frames
        if end_by_silence or end_by_maxlen:
            if len(buf) >= self.min_frames:
                segment = b"".join(buf)
            state["buf"] = []
            state["speaking"] = False
            state["silence"] = 0
        return segment

    def just_started(self, state: dict) -> bool:
        if state.pop("started", False):
            return True
        return False
