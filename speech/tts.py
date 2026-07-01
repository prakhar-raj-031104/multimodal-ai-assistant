"""
Text-to-speech with barge-in and automatic fallback.

The assistant speaks its replies. To keep perceived latency low we speak
*sentence by sentence* as the LLM streams tokens, rather than waiting for the
full answer. `interrupt()` stops playback immediately when the user starts
talking (barge-in) — essential for a natural, glasses-style conversation.

Backends (in preference order, each degrades gracefully):
  groq     — cloud neural TTS (Orpheus). Natural, but needs terms acceptance.
  pyttsx3  — offline, cross-platform (wraps espeak on Linux).
  spd-say  — offline, Linux speech-dispatcher (usually preinstalled).
  none     — text only.

Crucially: if the primary (cloud) backend fails at *runtime* — e.g. the model
was decommissioned or terms aren't accepted — we log a clear warning and
permanently switch to the best available local backend, so you never silently
lose voice output.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading

from core.config import TTSConfig
from core.logging_setup import get_logger

log = get_logger("tts")

_SENTENCE_END = re.compile(r"(.+?[.!?…]+[\s\"')\]]*)", re.S)


class TTS:
    def __init__(self, cfg: TTSConfig, groq_client=None) -> None:
        self.cfg = cfg
        self._groq = groq_client
        self._stop = threading.Event()
        self._pyttsx = None
        self._proc = None  # current spd-say subprocess (for barge-in)
        self._local_backend = self._detect_local()  # "pyttsx3" | "spd" | None
        self._backend = self._select_backend()
        log.info("TTS backend: %s (local fallback: %s)", self._backend, self._local_backend)

    # -- backend selection --------------------------------------------------
    def _select_backend(self) -> str:
        if not self.cfg.enabled:
            return "none"
        if self.cfg.backend == "groq" and self._groq is not None:
            return "groq"                       # primary; auto-falls back at runtime
        if self.cfg.backend in ("pyttsx3", "spd") and self._local_backend:
            return self._local_backend
        if self._local_backend:
            return self._local_backend
        if self._groq is not None:
            return "groq"
        log.warning("No TTS backend available; replies will be text-only")
        return "none"

    def _detect_local(self) -> "str | None":
        # pyttsx3 gives us clean stop()/barge-in; spd-say is the reliable
        # preinstalled Linux option.
        try:
            import pyttsx3
            self._pyttsx = pyttsx3.init()
            return "pyttsx3"
        except Exception:
            self._pyttsx = None
        if shutil.which("spd-say"):
            return "spd"
        return None

    # -- barge-in -----------------------------------------------------------
    def interrupt(self) -> None:
        self._stop.set()
        # Stop cloud playback
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        # Stop offline engines
        if self._pyttsx is not None:
            try:
                self._pyttsx.stop()
            except Exception:
                pass
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if shutil.which("spd-say"):
            subprocess.run(["spd-say", "-C"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

    # -- streaming speak ----------------------------------------------------
    def speak_stream(self, token_iter, on_token=None) -> str:
        """
        Consume an iterator of text chunks (LLM tokens), speaking each complete
        sentence as soon as it's ready. `on_token(chunk)` is called for every
        chunk as it arrives so the caller can render the text live (voice+text).
        Returns the full spoken text.
        """
        self._stop.clear()
        full, pending = [], ""
        for chunk in token_iter:
            if self._stop.is_set():
                break
            if not chunk:
                continue
            if on_token is not None:
                on_token(chunk)
            full.append(chunk)
            pending += chunk
            pending = self._flush_sentences(pending)
        if pending.strip() and not self._stop.is_set():
            self._say(pending.strip())
        return "".join(full)

    def _flush_sentences(self, pending: str) -> str:
        last = 0
        for m in _SENTENCE_END.finditer(pending):
            if self._stop.is_set():
                break
            self._say(m.group(1).strip())
            last = m.end()
        return pending[last:]

    def speak(self, text: str) -> None:
        if text.strip():
            self._stop.clear()
            self._say(text.strip())

    # -- dispatch (audio only; text rendering is owned by the caller) -------
    def _say(self, text: str) -> None:
        if self._stop.is_set() or not text:
            return
        try:
            self._dispatch(self._backend, text)
        except Exception as e:  # noqa
            # Primary failed at runtime — switch to local backend permanently.
            if self._backend == "groq" and self._local_backend:
                log.warning("cloud TTS failed (%s); switching to local '%s'",
                            str(e)[:120], self._local_backend)
                self._backend = self._local_backend
                try:
                    self._dispatch(self._backend, text)
                except Exception as e2:  # noqa
                    log.warning("local TTS also failed: %s", e2)
            else:
                log.warning("TTS error: %s", str(e)[:120])

    def _dispatch(self, backend: str, text: str) -> None:
        if backend == "groq":
            self._say_groq(text)
        elif backend == "pyttsx3":
            self._say_pyttsx(text)
        elif backend == "spd":
            self._say_spd(text)
        # "none" -> nothing

    def _say_groq(self, text: str) -> None:
        resp = self._groq.audio.speech.create(
            model=self.cfg.groq_model,
            voice=self.cfg.voice,
            input=text,
            response_format="wav",
        )
        audio = resp.read() if hasattr(resp, "read") else bytes(resp)
        self._play_wav_bytes(audio)

    def _say_pyttsx(self, text: str) -> None:
        self._pyttsx.say(text)
        self._pyttsx.runAndWait()

    def _say_spd(self, text: str) -> None:
        # -w waits until the utterance finishes so sentences don't overlap.
        self._proc = subprocess.Popen(
            ["spd-say", "-w", "-t", "female1", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._proc.wait()

    def _play_wav_bytes(self, wav: bytes) -> None:
        if self._stop.is_set():
            return
        try:
            import io, wave
            import numpy as np
            import sounddevice as sd
            with wave.open(io.BytesIO(wav), "rb") as wf:
                sr = wf.getframerate()
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            sd.play(data, sr)
            sd.wait()
        except Exception:
            import tempfile
            player = next((p for p in ("aplay", "ffplay", "afplay") if shutil.which(p)), None)
            if not player:
                raise
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                f.write(wav)
                f.flush()
                args = [player, f.name] if player != "ffplay" else [player, "-nodisp", "-autoexit", f.name]
                subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
