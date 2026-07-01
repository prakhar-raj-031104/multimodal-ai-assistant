"""
Speech-to-text with a low-latency cloud path and local fallbacks.

Primary: Groq whisper-large-v3-turbo — sub-second for short utterances and it
reuses the Groq key already in use for the LLM, so no extra infra.
Fallbacks: faster-whisper (local, GPU/CPU) -> openai-whisper (local).

Segments arrive as raw int16 PCM bytes; we wrap them into an in-memory WAV so
no temp files touch disk on the hot path.
"""
from __future__ import annotations

import io
import wave
from typing import Optional

from core.config import STTConfig
from core.logging_setup import get_logger

log = get_logger("stt")


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class STT:
    def __init__(self, cfg: STTConfig, sample_rate: int, groq_client=None) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self._groq = groq_client
        self._local = None
        self._backend = self._select_backend()

    def _select_backend(self) -> str:
        if self.cfg.backend == "groq" and self._groq is not None:
            return "groq"
        if self.cfg.backend == "faster_whisper" and self._load_faster_whisper():
            return "faster_whisper"
        if self._load_whisper():
            return "whisper"
        if self._groq is not None:
            return "groq"
        raise RuntimeError("No STT backend available (need GROQ_API_KEY or local whisper)")

    def _load_faster_whisper(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            self._local = WhisperModel(self.cfg.local_model, device=device, compute_type=compute)
            log.info("STT: faster-whisper (%s/%s)", self.cfg.local_model, device)
            return True
        except Exception as e:  # noqa
            log.debug("faster-whisper unavailable: %s", e)
            return False

    def _load_whisper(self) -> bool:
        try:
            import whisper
            self._local = whisper.load_model(self.cfg.local_model)
            self._backend_local_kind = "whisper"
            log.info("STT: openai-whisper (%s)", self.cfg.local_model)
            return True
        except Exception as e:  # noqa
            log.debug("openai-whisper unavailable: %s", e)
            return False

    def transcribe(self, pcm: bytes) -> str:
        try:
            if self._backend == "groq":
                return self._transcribe_groq(pcm)
            if self._backend == "faster_whisper":
                return self._transcribe_faster(pcm)
            return self._transcribe_whisper(pcm)
        except Exception as e:  # noqa
            log.error("STT failed on %s: %s", self._backend, e)
            return ""

    def _transcribe_groq(self, pcm: bytes) -> str:
        wav = pcm_to_wav_bytes(pcm, self.sample_rate)
        resp = self._groq.audio.transcriptions.create(
            file=("audio.wav", wav, "audio/wav"),
            model=self.cfg.groq_model,
            language=self.cfg.language,
            response_format="text",
        )
        return (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()

    def _transcribe_faster(self, pcm: bytes) -> str:
        import numpy as np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._local.transcribe(audio, language=self.cfg.language, vad_filter=False)
        return " ".join(s.text for s in segments).strip()

    def _transcribe_whisper(self, pcm: bytes) -> str:
        import numpy as np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        result = self._local.transcribe(audio, language=self.cfg.language, fp16=False)
        return result["text"].strip()
