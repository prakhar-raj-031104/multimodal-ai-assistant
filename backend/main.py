"""
Entrypoint for the real-time multimodal second-brain assistant.

Modes:
  python backend/main.py            # live: mic + camera, always-on
  python backend/main.py --text     # typed queries (no hardware needed)
  python backend/main.py --selftest # boot every subsystem, no mic/camera
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import config
from core.logging_setup import setup_logging, get_logger
from core.latency import stats_snapshot

log = get_logger("main")


async def _run_live() -> None:
    from core.runtime import Assistant
    assistant = Assistant(config)
    try:
        await assistant.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        assistant.shutdown()


async def _run_text() -> None:
    from core.runtime import Assistant
    assistant = Assistant(config)
    print("\n💬 Text mode. Type a query (or 'quit').\n")
    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, input, "you › ")
            if line.strip().lower() in {"quit", "exit", "q"}:
                break
            if not line.strip():
                continue
            print("assistant › ", end="", flush=True)
            await assistant.handle_text(line.strip())
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        assistant.shutdown()
        _print_latency()


def _run_selftest() -> None:
    """Boot every subsystem and report readiness — no mic/camera required."""
    setup_logging(config.log_level)
    from core.runtime import Assistant
    print("\n=== SELF TEST ===")
    a = Assistant(config)
    checks = {
        "LLM provider": config.llm.provider,
        "Groq client (STT/TTS)": a.groq is not None,
        "STT backend": getattr(a.stt, "_backend", None),
        "TTS backend": getattr(a.tts, "_backend", None),
        "Embeddings": a.memory.embeddings._kind,
        "Vector store facts": len(a.memory.store),
        "Vision engine": a.vision.engine is not None,
        "Agent tools": len(a.tools),
    }
    for k, v in checks.items():
        print(f"  {k:32} : {v}")
    if a.groq is not None:
        print("\n  LLM smoke test:")
        reply = "".join(a.brain.answer_stream("Say 'ready' in one word."))
        print(f"    -> {reply.strip()[:80]}")
    print("=== OK ===\n")
    a.shutdown()


def _print_latency() -> None:
    stats = stats_snapshot()
    if not stats:
        return
    print("\n--- latency (p50/p95 ms) ---")
    for stage, s in stats.items():
        print(f"  {stage:16} p50={s['p50_ms']:.0f}  p95={s['p95_ms']:.0f}  n={s['count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time multimodal second-brain assistant")
    parser.add_argument("--text", action="store_true", help="typed input, no hardware")
    parser.add_argument("--selftest", action="store_true", help="boot & health-check only")
    args = parser.parse_args()

    setup_logging(config.log_level)

    if args.selftest:
        _run_selftest()
    elif args.text:
        asyncio.run(_run_text())
    else:
        asyncio.run(_run_live())


if __name__ == "__main__":
    main()
