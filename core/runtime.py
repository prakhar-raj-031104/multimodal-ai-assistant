"""
The always-on runtime — the heart of the real-time assistant.

Wires the independent async producers/consumers together:

    mic ─▶ VAD segmenter ─▶ STT ─▶ router ─▶ brain ─▶ TTS  (barge-in aware)
    cam ─▶ scene-change ─▶ VLM ─▶ perception ─▶ memory
                                         └────────────▶ episodic/long-term

Everything runs concurrently on one asyncio loop. Blocking work (LLM/STT/TTS/
VLM network + audio playback) is pushed to a thread pool so the loop stays
responsive — crucially, so it can detect barge-in while the assistant is
speaking and cut playback instantly.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from core.config import Config
from core.events import Event, EventType, Perception
from core.latency import Trace, stopwatch
from core.logging_setup import get_logger

from speech.audio_stream import AudioStream
from speech.vad import VADSegmenter
from speech.wake_word import WakeWord
from speech.stt import STT
from speech.tts import TTS

from vision.vision_stream import VisionStream

from memory.memory_manager import MemoryManager

from brain.llm_engine import get_groq_client
from brain.providers import build_provider
from brain.router import Router, Intent
from brain.processor import Brain

from agents.tools import build_default_registry

log = get_logger("runtime")


class Assistant:
    """Composition root. Build once, then `await run()` or call `handle_text()`."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.bus = None  # set in run()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._speaking = False
        self._running = False

        # Shared Groq client powers LLM + STT + TTS.
        self.groq = get_groq_client(cfg.groq_api_key())

        # Memory
        self.memory = MemoryManager(cfg.memory, cfg.llm, groq_client=self.groq)

        # Brain — provider is chosen from config (groq | anthropic/Claude).
        self.tools = build_default_registry(self.memory)
        provider = build_provider(cfg.llm, groq_client=self.groq, tools=self.tools)
        self.router = Router(cfg=cfg.llm)
        self.brain = Brain(provider, cfg.llm, memory=self.memory)

        # Speech I/O
        self.tts = TTS(cfg.tts, groq_client=self.groq)
        self.stt = STT(cfg.stt, cfg.audio.sample_rate, groq_client=self.groq)

        # Vision (engine injected lazily so a missing HF token doesn't crash boot)
        self.vision = VisionStream(cfg.vision, bus=None, engine=self._make_vision_engine())

        log.info("assistant assembled (provider=%s, tools=%d, memory=%d facts)",
                 cfg.llm.provider, len(self.tools), len(self.memory.store))

    def _make_vision_engine(self):
        if not self.cfg.vision.enabled:
            return None
        try:
            from vision.vision_engine import VisionEngine
            return VisionEngine(
                base_url=self.cfg.vision.base_url,
                model=self.cfg.vision.model,
                max_tokens=self.cfg.vision.max_tokens,
                timeout=self.cfg.vision.timeout,
            )
        except Exception as e:  # noqa
            log.warning("vision engine disabled: %s", e)
            return None

    # ======================================================================
    # Live (mic + camera) mode
    # ======================================================================
    async def run(self) -> None:
        from core.events import EventBus
        self.bus = EventBus()
        self.vision.bus = self.bus
        self._running = True

        tasks = [
            asyncio.create_task(self._audio_loop(), name="audio"),
            asyncio.create_task(self.vision.run(), name="vision"),
            asyncio.create_task(self._perception_consumer(), name="perception"),
        ]
        log.info("🟢 assistant live — speak to it (Ctrl-C to stop)")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.shutdown()

    async def _audio_loop(self) -> None:
        mic = AudioStream(self.cfg.audio)
        try:
            mic.start()
        except Exception as e:  # noqa
            log.error("microphone unavailable (%s) — use text mode instead", e)
            return
        seg = VADSegmenter(self.cfg.audio, self.cfg.vad)
        wake = WakeWord(self.cfg.wake_word, self.cfg.audio.sample_rate)
        state: dict = {}
        loop = asyncio.get_running_loop()

        async for frame in mic.frames():
            if not self._running:
                break
            wake.feed(frame)
            # Barge-in: user starts talking while we're speaking -> cut TTS.
            if self._speaking and seg.just_started(state):
                self.tts.interrupt()

            segment = seg.process(frame, state)
            seg.just_started(state)  # clear one-shot flag if not consumed
            if segment is None:
                continue
            if not wake.is_active():
                continue
            asyncio.create_task(self._on_segment(segment, loop))

        mic.stop()

    async def _on_segment(self, pcm: bytes, loop) -> None:
        trace = Trace("turn")
        with stopwatch("stt", trace):
            text = await loop.run_in_executor(self._executor, self.stt.transcribe, pcm)
        text = (text or "").strip()
        if not text:
            return
        log.info("🗣️  %s", text)
        await self._handle(text, trace)

    async def _handle(self, text: str, trace: Optional[Trace] = None) -> None:
        intent = self.router.route(text)
        self.memory.observe("utterance", text)

        if intent == Intent.AMBIENT:
            log.debug("ambient — not responding")
            await self._background_consolidate()
            return

        scene = None
        if intent == Intent.VISUAL:
            with stopwatch("vision_ondemand", trace) if trace else _null():
                p = await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._analyze_now_sync)
                scene = p.summary if p else None
        else:
            scene = self._current_scene()

        await self._respond(text, scene, trace)
        await self._background_consolidate()

    def _analyze_now_sync(self):
        # Bridge async analyze into a thread; simplest is a fresh event loop call.
        frame = self.vision.latest_frame() if self.vision else None
        if frame is None or self.vision.engine is None:
            return None
        try:
            raw = self.vision.engine.analyze_frame(frame)
            return self.vision._to_perception(raw)
        except Exception:
            return None

    async def _respond(self, text: str, scene: Optional[str], trace: Optional[Trace]) -> None:
        self._speaking = True
        loop = asyncio.get_running_loop()

        def _work() -> str:
            gen = self.brain.answer_stream(text, scene_summary=scene)
            # Render text live (token-by-token) AND speak it. speak_stream drives
            # the audio; on_token prints each chunk so you see + hear the reply.
            print("\033[38;5;250m🤖 assistant ›\033[0m ", end="", flush=True)
            reply = self.tts.speak_stream(
                gen, on_token=lambda t: print(t, end="", flush=True))
            print()
            return reply

        try:
            with stopwatch("respond", trace) if trace else _null():
                reply = await loop.run_in_executor(self._executor, _work)
            if self.bus:
                await self.bus.publish(Event(EventType.ASSISTANT_REPLY, {"text": reply}))
        finally:
            self._speaking = False
            if trace:
                log.info(trace.summary())

    def _current_scene(self) -> Optional[str]:
        """Most recent perception, but only if fresh enough to call 'current'.

        Stale perceptions are the main source of visual hallucination ("I see X"
        when X was minutes ago), so we drop anything older than the freshness
        window rather than pass it off as the current view.
        """
        recent = self.memory.episodic.recent(limit=20)
        for ep in reversed(recent):
            if ep.kind == "perception":
                if ep.age() <= self.cfg.llm.scene_freshness_s:
                    return ep.text
                return None
        return None

    async def _perception_consumer(self) -> None:
        if self.bus is None:
            return
        q = self.bus.subscribe()
        while self._running:
            try:
                event: Event = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if event.type == EventType.PERCEPTION:
                p: Perception = event.data["perception"]
                self.memory.observe("perception", p.summary)

    async def _background_consolidate(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.memory.maybe_consolidate)

    # ======================================================================
    # Text mode (no hardware) — great for testing & headless deploys
    # ======================================================================
    async def handle_text(self, text: str) -> str:
        """Process a typed query end-to-end; returns the reply text."""
        intent = self.router.route(text)
        self.memory.observe("utterance", text)
        if intent == Intent.AMBIENT:
            intent = Intent.ANSWER  # in text mode, always answer typed input
        scene = self._current_scene()
        # Both: stream text to stdout live AND speak it (if TTS enabled).
        loop = asyncio.get_running_loop()

        def _work() -> str:
            gen = self.brain.answer_stream(text, scene_summary=scene)
            return self.tts.speak_stream(
                gen, on_token=lambda t: print(t, end="", flush=True))

        reply = await loop.run_in_executor(self._executor, _work)
        print()
        await self._background_consolidate()
        return reply

    def shutdown(self) -> None:
        self._running = False
        try:
            self.vision.stop()
        except Exception:
            pass
        self._executor.shutdown(wait=False)
        log.info("assistant stopped")


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False
