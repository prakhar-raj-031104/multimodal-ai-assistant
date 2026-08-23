"""
Continuous visual perception loop.

Captures frames from the camera on a worker thread, gates them through the
scene-change detector + a hard FPS cap, then runs the VLM asynchronously and
publishes a `Perception` event. It also keeps the latest raw frame so the brain
can do an on-demand "look right now" when the user asks a visual question.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Optional

from core.config import VisionConfig
from core.events import Event, EventType, Perception
from core.latency import stopwatch
from core.logging_setup import get_logger
from vision.scene_change import SceneChangeDetector

log = get_logger("vision.stream")


class VisionStream:
    def __init__(self, cfg: VisionConfig, bus, engine=None) -> None:
        self.cfg = cfg
        self.bus = bus
        self.engine = engine  # a VisionEngine; injected so it can be mocked
        self.scene = SceneChangeDetector(cfg.scene_change_threshold)
        self._cap = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self.available = False

    def _open_camera(self) -> bool:
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.cfg.camera_index)
            if not self._cap.isOpened():
                log.warning("camera %d not accessible; vision disabled", self.cfg.camera_index)
                return False
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            log.info("📷 camera %d open", self.cfg.camera_index)
            return True
        except Exception as e:  # noqa
            log.warning("camera init failed (%s); vision disabled", e)
            return False

    def _grab_loop(self) -> None:
        """Continuously read frames on a worker thread; keep only the latest."""
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._frame_lock:
                self._latest_frame = frame

    def latest_frame(self):
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    async def run(self) -> None:
        if not self.cfg.enabled or self.engine is None:
            log.info("vision stream disabled")
            return
        if not self._open_camera():
            return
        self.available = True
        self._running = True
        threading.Thread(target=self._grab_loop, daemon=True).start()

        min_interval = 1.0 / max(self.cfg.max_fps, 0.01)
        loop = asyncio.get_running_loop()
        last_call = 0.0
        while self._running:
            await asyncio.sleep(0.1)
            frame = self.latest_frame()
            if frame is None:
                continue
            if (time.time() - last_call) < min_interval:
                continue
            if not self.scene.is_significant(frame):
                continue
            last_call = time.time()
            try:
                with stopwatch("vision_vlm"):
                    raw = await loop.run_in_executor(None, self.engine.analyze_frame, frame)
                perception = self._to_perception(raw)
                await self.bus.publish(Event(EventType.PERCEPTION, {"perception": perception}))
                log.info("👁️  %s", perception.summary[:90])
            except Exception as e:  # noqa
                log.debug("vision analyze failed: %s", e)

    async def analyze_now(self) -> Optional[Perception]:
        """On-demand single-shot analysis of the current frame (for questions)."""
        if not self.available or self.engine is None:
            return None
        frame = self.latest_frame()
        if frame is None:
            return None
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, self.engine.analyze_frame, frame)
            return self._to_perception(raw)
        except Exception:
            return None

    def _to_perception(self, raw: str) -> Perception:
        data = _extract_json(raw)
        summary = data.get("scene_summary") if isinstance(data, dict) else None
        if isinstance(summary, (list, dict)):
            summary = json.dumps(summary)
        if not summary:
            summary = _clean_summary(raw)[:400]
        return Perception(summary=str(summary).strip(),
                          raw=data if isinstance(data, dict) else {"text": raw})

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _extract_json(text: str) -> dict:
    """Parse the VLM's JSON, tolerating the three ways it usually arrives dirty:
    wrapped in a ```json fence, prefixed with prose, or cut off mid-object by
    max_tokens. Without the truncation salvage, a clipped response fell through
    to the raw-text path and a 400-char JSON blob got stored as a "perception" —
    which is both unreadable to the LLM and a direct hallucination source."""
    if not text:
        return {}
    t = text.strip()

    # ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Largest brace-delimited span
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    # Truncated output: pull the fields we actually use straight out of the text.
    out = {}
    sm = re.search(r'"scene_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', t, re.S)
    if sm:
        try:
            out["scene_summary"] = json.loads(f'"{sm.group(1)}"')
        except Exception:
            out["scene_summary"] = sm.group(1)
    return out


def _clean_summary(text: str) -> str:
    """Last-resort summary from non-JSON output: strip fences/braces so we never
    hand the brain a wall of syntax."""
    t = re.sub(r"```(?:json)?", " ", text or "")
    t = re.sub(r'^\s*[\{\[]', " ", t)
    t = re.sub(r'"\w+"\s*:', " ", t)
    t = re.sub(r"[\{\}\[\]\"]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
