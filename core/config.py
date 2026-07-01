"""
Central configuration for the real-time multimodal second-brain assistant.

All tunables live here. Values can be overridden via environment variables
(loaded from a local .env file) so the same code runs in dev and prod without
edits. Everything is a plain dataclass so it is trivial to serialise/inspect.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from the project root once, on import.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AudioConfig:
    sample_rate: int = _env_int("AUDIO_SAMPLE_RATE", 16000)
    channels: int = 1
    # Frame size fed to VAD (ms). 20/30ms are typical for webrtc-style VAD.
    frame_ms: int = _env_int("AUDIO_FRAME_MS", 30)
    # Silence (ms) after speech before we close a segment and transcribe.
    silence_hangover_ms: int = _env_int("AUDIO_SILENCE_HANGOVER_MS", 600)
    # Ignore segments shorter than this (ms) — filters coughs/clicks.
    min_speech_ms: int = _env_int("AUDIO_MIN_SPEECH_MS", 300)
    # Hard cap on a single utterance (ms) so STT calls stay bounded.
    max_speech_ms: int = _env_int("AUDIO_MAX_SPEECH_MS", 15000)
    input_device: Optional[int] = None  # None = system default mic

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)


@dataclass
class VADConfig:
    # "silero" (accurate, needs torch) | "webrtc" | "energy" (always available)
    backend: str = _env("VAD_BACKEND", "silero")
    # Probability threshold for silero; energy threshold auto-calibrates.
    threshold: float = _env_float("VAD_THRESHOLD", 0.5)


@dataclass
class WakeWordConfig:
    # If disabled, the assistant answers any transcribed speech (always-on).
    enabled: bool = _env_bool("WAKEWORD_ENABLED", False)
    model: str = _env("WAKEWORD_MODEL", "hey_jarvis")
    threshold: float = _env_float("WAKEWORD_THRESHOLD", 0.5)


@dataclass
class STTConfig:
    # "groq" (low latency, cloud) | "faster_whisper" | "whisper" (local)
    backend: str = _env("STT_BACKEND", "groq")
    groq_model: str = _env("STT_GROQ_MODEL", "whisper-large-v3-turbo")
    local_model: str = _env("STT_LOCAL_MODEL", "base")
    language: Optional[str] = os.getenv("STT_LANGUAGE") or None


@dataclass
class TTSConfig:
    enabled: bool = _env_bool("TTS_ENABLED", True)
    # "groq" (streaming, cloud) | "piper" | "pyttsx3" (offline) | "none"
    backend: str = _env("TTS_BACKEND", "groq")
    groq_model: str = _env("TTS_GROQ_MODEL", "canopylabs/orpheus-v1-english")
    voice: str = _env("TTS_VOICE", "tara")


@dataclass
class VisionConfig:
    enabled: bool = _env_bool("VISION_ENABLED", True)
    camera_index: int = _env_int("VISION_CAMERA_INDEX", 0)
    # Max VLM calls per second (rate limit / cost guard).
    max_fps: float = _env_float("VISION_MAX_FPS", 0.5)
    # Only call the VLM when the scene changes by more than this (0..1).
    scene_change_threshold: float = _env_float("VISION_SCENE_CHANGE_THRESHOLD", 0.35)
    base_url: str = _env("VISION_BASE_URL", "https://router.huggingface.co/v1")
    model: str = _env("VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    max_tokens: int = _env_int("VISION_MAX_TOKENS", 600)
    timeout: float = _env_float("VISION_TIMEOUT", 30.0)


@dataclass
class MemoryConfig:
    persist_dir: str = _env("MEMORY_DIR", str(_PROJECT_ROOT / "data" / "memory"))
    # "sentence_transformers" | "fastembed" | "hash" (dependency-free fallback)
    embedding_backend: str = _env("EMBEDDING_BACKEND", "sentence_transformers")
    embedding_model: str = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dim: int = _env_int("EMBEDDING_DIM", 384)
    # Rolling short-term perception buffer (seconds of "what just happened").
    episodic_window_s: float = _env_float("EPISODIC_WINDOW_S", 120.0)
    top_k: int = _env_int("MEMORY_TOP_K", 5)
    # After this many raw perceptions, consolidate them into a durable summary.
    consolidate_every: int = _env_int("MEMORY_CONSOLIDATE_EVERY", 20)


@dataclass
class LLMConfig:
    # "groq" (free, fast) | "gemini" (free, strong — best free tier) | "anthropic" (Claude, paid)
    provider: str = _env("LLM_PROVIDER", "groq")
    api_key_env: str = "GROQ_API_KEY"
    model: str = _env("LLM_MODEL", "llama-3.3-70b-versatile")
    # Small/fast model for routing + consolidation (latency-sensitive paths).
    fast_model: str = _env("LLM_FAST_MODEL", "llama-3.1-8b-instant")
    # Gemini settings — used when provider == "gemini" (free tier).
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")
    # Anthropic (Claude) settings — used when provider == "anthropic".
    anthropic_model: str = _env("ANTHROPIC_MODEL", "claude-opus-4-8")
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    temperature: float = _env_float("LLM_TEMPERATURE", 0.3)
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 1024)
    max_history_turns: int = _env_int("LLM_MAX_HISTORY_TURNS", 8)
    enable_tools: bool = _env_bool("LLM_ENABLE_TOOLS", True)
    # Only treat perceptions newer than this (s) as "what you currently see".
    scene_freshness_s: float = _env_float("SCENE_FRESHNESS_S", 45.0)


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    log_level: str = _env("LOG_LEVEL", "INFO")

    def groq_api_key(self) -> Optional[str]:
        return os.getenv(self.llm.api_key_env)

    def hf_token(self) -> Optional[str]:
        return os.getenv("HF_TOKEN")

    def as_dict(self) -> dict:
        return asdict(self)


# A ready-to-use default instance. Import this everywhere.
config = Config()
