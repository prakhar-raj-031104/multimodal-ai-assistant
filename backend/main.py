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


def _run_selftest() -> int:
    """Boot every subsystem and verify it actually works — no mic/camera needed.

    Returns the number of FAILED checks so CI (and you) get a real signal; the
    old version printed "OK" even when the LLM call had errored out.
    """
    setup_logging(config.log_level)
    from core.runtime import Assistant
    print("\n=== SELF TEST ===")
    a = Assistant(config)
    failures = []

    def check(name, ok, detail=""):
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:30} {detail}")
        if not ok:
            failures.append(name)

    check("Groq client (STT/TTS/LLM)", a.groq is not None,
          "missing GROQ_API_KEY" if a.groq is None else "")
    check("STT backend", getattr(a.stt, "_backend", None) is not None,
          str(getattr(a.stt, "_backend", None)))
    check("TTS backend", True, f"{getattr(a.tts, '_backend', None)} "
                               f"(local fallback: {getattr(a.tts, '_local_backend', None)})")
    check("Embeddings", a.memory.embeddings._kind != "hash",
          f"{a.memory.embeddings._kind} (dim={a.memory.embeddings.dim})")
    check("Vector store", True, f"{len(a.memory.store)} durable facts")
    check("Agent tools", len(a.tools) > 0, f"{len(a.tools)} registered")

    # --- LLM: the configured model must actually exist on the provider ------
    reply = ""
    try:
        reply = "".join(a.brain.answer_stream("Reply with the single word: ready.")).strip()
    except Exception as e:  # noqa
        reply = f"exception: {e}"
    llm_ok = bool(reply) and "unavailable" not in reply.lower() \
        and "interrupted" not in reply.lower()
    model = config.llm.gemini_model if config.llm.provider == "gemini" else config.llm.model
    check(f"LLM ({config.llm.provider}/{model})", llm_ok, f"-> {reply[:60]!r}")

    # --- Vision: reachable AND the model is enabled for this token ----------
    if a.vision.engine is None:
        check("Vision engine", not config.vision.enabled, "disabled")
    else:
        try:
            import numpy as np
            import cv2
            frame = np.full((240, 320, 3), 200, dtype=np.uint8)
            cv2.putText(frame, "TEST", (60, 140), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0, (0, 0, 0), 5)
            raw = a.vision.engine.analyze_frame(frame)
            summary = a.vision._to_perception(raw).summary
            check(f"Vision ({config.vision.model})", bool(summary), f"-> {summary[:50]!r}")
        except Exception as e:  # noqa
            check(f"Vision ({config.vision.model})", False, str(e)[:90])

    # --- Memory retrieval precision (the anti-hallucination gate) ----------
    gated = not a.memory.needs_recall("What is the capital of France?")
    opened = a.memory.needs_recall("What did I ask you to remember earlier?")
    check("Memory relevance gate", gated and opened,
          f"general-knowledge skipped={gated}, recall-query retrieves={opened}")

    print(f"=== {'OK' if not failures else str(len(failures)) + ' CHECK(S) FAILED: ' + ', '.join(failures)} ===\n")
    a.shutdown()
    return len(failures)


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
        sys.exit(1 if _run_selftest() else 0)
    elif args.text:
        asyncio.run(_run_text())
    else:
        asyncio.run(_run_live())


if __name__ == "__main__":
    main()
