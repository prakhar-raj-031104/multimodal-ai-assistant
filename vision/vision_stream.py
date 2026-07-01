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
        if not summary:
            summary = (raw or "").strip()[:400]
        return Perception(summary=summary, raw=data if isinstance(data, dict) else {"text": raw})

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}
