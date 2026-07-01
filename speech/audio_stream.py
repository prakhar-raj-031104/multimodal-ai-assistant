"""
Continuous microphone capture.

sounddevice's callback runs on a dedicated audio thread; we hand frames off to
the asyncio world through a thread-safe queue. This replaces the old blocking
`sd.rec(duration=5)` model — the mic is now always open and never blocks the
event loop.
"""
from __future__ import annotations

import asyncio
import queue
from typing import Optional

import numpy as np

from core.config import AudioConfig
from core.logging_setup import get_logger

log = get_logger("audio.stream")


class AudioStream:
    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._stream = None
        self._running = False

    def _callback(self, indata, frames, time_info, status) -> None:  # audio thread
        if status:
            log.debug("audio status: %s", status)
        try:
            self._q.put_nowait(bytes(indata))
        except queue.Full:
            # Drop the oldest frame to keep latency bounded under back-pressure.
            try:
                self._q.get_nowait()
                self._q.put_nowait(bytes(indata))
            except queue.Empty:
                pass

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.RawInputStream(
            samplerate=self.cfg.sample_rate,
            blocksize=self.cfg.frame_samples,
            device=self.cfg.input_device,
            channels=self.cfg.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        log.info("🎙️  mic open @ %d Hz, %dms frames", self.cfg.sample_rate, self.cfg.frame_ms)

    async def frames(self):
        """Async generator yielding raw int16 PCM frames as bytes."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                frame = await loop.run_in_executor(None, self._q.get, True, 0.5)
            except queue.Empty:
                continue
            yield frame

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("mic closed")


def pcm_to_float(frame: bytes) -> np.ndarray:
    return np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
