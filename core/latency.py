"""
Lightweight latency instrumentation.

Every hot-path stage wraps itself in `stopwatch(...)` so we can print an
end-to-end latency breakdown (mic -> VAD -> STT -> LLM -> TTS). Keeping this
visible is the single most useful thing for driving real-time latency down.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

from core.logging_setup import get_logger

log = get_logger("latency")


@dataclass
class Trace:
    """Accumulates named stage durations for one end-to-end interaction."""
    name: str
    t0: float = field(default_factory=time.perf_counter)
    stages: List[tuple] = field(default_factory=list)

    def mark(self, stage: str, seconds: float) -> None:
        self.stages.append((stage, seconds))

    def total(self) -> float:
        return time.perf_counter() - self.t0

    def summary(self) -> str:
        parts = " ".join(f"{s}={d*1000:.0f}ms" for s, d in self.stages)
        return f"[{self.name}] total={self.total()*1000:.0f}ms | {parts}"


# Aggregate stats for a simple production dashboard / health endpoint.
_STATS: Dict[str, List[float]] = {}


@contextmanager
def stopwatch(stage: str, trace: "Trace | None" = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        _STATS.setdefault(stage, []).append(dur)
        if trace is not None:
            trace.mark(stage, dur)
        log.debug("%s took %.0fms", stage, dur * 1000)


def stats_snapshot() -> Dict[str, dict]:
    out = {}
    for stage, xs in _STATS.items():
        if not xs:
            continue
        s = sorted(xs)
        out[stage] = {
            "count": len(s),
            "avg_ms": sum(s) / len(s) * 1000,
            "p50_ms": s[len(s) // 2] * 1000,
            "p95_ms": s[min(len(s) - 1, int(len(s) * 0.95))] * 1000,
        }
    return out
