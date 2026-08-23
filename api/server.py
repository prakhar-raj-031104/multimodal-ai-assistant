"""
HTTP API for the second-brain assistant — the backend behind the web console.

Design is driven by one number: how long until the user sees a reply.

  * The Assistant is built ONCE at startup (embeddings + vector store + clients),
    so a request never pays boot cost.
  * /api/chat streams tokens as Server-Sent Events, so text renders at
    time-to-first-token (~300ms) instead of time-to-full-answer.
  * Vision NEVER blocks chat. The browser pushes frames to /api/vision/frame;
    they are scene-gated and analysed on a background thread, and the newest
    perception is what chat reads. A VLM call is 3-5s — putting it on the reply
    path would make every grounded answer feel broken.
  * Voice output is synthesised in the browser (SpeechSynthesis), so it starts
    speaking the first sentence while the rest is still streaming.

    uvicorn api.server:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.config import config
from core.latency import stats_snapshot, stopwatch
from core.logging_setup import setup_logging, get_logger

log = get_logger("api")

_WEB = Path(__file__).resolve().parent.parent / "web"

# Populated at startup. Module-level so the SSE generator can reach it.
STATE: dict = {
    "assistant": None,
    "perception": None,      # {"summary": str, "ts": float}
    "vision_busy": False,
    "scene": None,           # SceneChangeDetector
    "turns": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(config.log_level)
    t0 = time.perf_counter()
    from core.runtime import Assistant
    from vision.scene_change import SceneChangeDetector

    STATE["assistant"] = Assistant(config)
    STATE["scene"] = SceneChangeDetector(config.vision.scene_change_threshold)
    log.info("🌐 web console ready in %.1fs — http://127.0.0.1:%s",
             time.perf_counter() - t0, os.getenv("PORT", "8100"))
    yield
    a = STATE.get("assistant")
    if a is not None:
        a.shutdown()


app = FastAPI(title="Second Brain Assistant", lifespan=lifespan)


class NoCompressSSE(BaseHTTPMiddleware):
    """Mark SSE responses so gzip leaves them alone.

    Compressing an event stream buffers it — the tokens arrive in one lump at
    the end, which silently destroys the whole point of streaming. Setting an
    explicit Content-Encoding makes GZipMiddleware skip the response.
    """

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            response.headers["Content-Encoding"] = "identity"
        return response


class CacheStatic(BaseHTTPMiddleware):
    """Long-lived caching for /assets, never for the API."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=3600"
        return response


# Order matters: gzip is added first so it runs OUTERMOST, letting the
# Content-Encoding set by NoCompressSSE (inner) suppress it.
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(NoCompressSSE)
app.add_middleware(CacheStatic)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ==========================================================================
# Status / telemetry
# ==========================================================================
@app.get("/api/status")
async def status():
    a = STATE["assistant"]
    if a is None:
        return JSONResponse({"ready": False}, status_code=503)
    p = STATE["perception"]
    return {
        "ready": True,
        "provider": config.llm.provider,
        "model": (config.llm.gemini_model if config.llm.provider == "gemini"
                  else config.llm.model),
        "stt": getattr(a.stt, "_backend", None),
        "tts": getattr(a.tts, "_backend", None),
        "embeddings": a.memory.embeddings._kind,
        "embedding_dim": a.memory.embeddings.dim,
        "vision_model": config.vision.model if a.vision.engine else None,
        "vision_online": a.vision.engine is not None,
        "tools": [s["function"]["name"] for s in a.tools.schemas()],
        "memory_facts": len(a.memory.store),
        "episodic": len(a.memory.episodic),
        "turns": STATE["turns"],
        "perception": ({"summary": p["summary"], "age_s": round(time.time() - p["ts"], 1)}
                       if p else None),
        "latency": stats_snapshot(),
    }


# ==========================================================================
# Chat — SSE token stream
# ==========================================================================
def _fresh_scene() -> Optional[str]:
    """Latest perception, but only while it can honestly be called 'current'."""
    p = STATE["perception"]
    if not p:
        return None
    if (time.time() - p["ts"]) > config.llm.scene_freshness_s:
        return None
    return p["summary"]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat")
async def chat(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    a = STATE["assistant"]
    if a is None:
        return JSONResponse({"error": "not ready"}, status_code=503)

    use_scene = bool(payload.get("use_scene", True))
    scene = _fresh_scene() if use_scene else None

    async def gen():
        loop = asyncio.get_running_loop()
        q: "queue.Queue" = queue.Queue()
        started = time.perf_counter()

        def produce():
            """Run the blocking token generator on a worker thread."""
            try:
                for chunk in a.brain.answer_stream(text, scene_summary=scene):
                    q.put(("token", chunk))
            except Exception as e:  # noqa
                log.error("chat stream failed: %s", e)
                q.put(("error", str(e)[:200]))
            finally:
                q.put((None, None))

        threading.Thread(target=produce, daemon=True).start()

        yield _sse("start", {"scene_used": bool(scene), "scene": scene})
        first = None
        full = []
        while True:
            kind, value = await loop.run_in_executor(None, q.get)
            if kind is None:
                break
            if kind == "error":
                yield _sse("error", {"message": value})
                break
            if first is None:
                first = (time.perf_counter() - started) * 1000
                yield _sse("ttft", {"ms": round(first)})
            full.append(value)
            yield _sse("token", {"t": value})

        reply = "".join(full).strip()
        STATE["turns"] += 1
        # Consolidation is off the hot path — do it after the answer is delivered.
        await loop.run_in_executor(None, a.memory.maybe_consolidate)
        yield _sse("done", {
            "text": reply,
            "total_ms": round((time.perf_counter() - started) * 1000),
            "ttft_ms": round(first) if first else None,
            "memory_facts": len(a.memory.store),
        })

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",       # stop nginx buffering the stream
        "Connection": "keep-alive",
    })


# ==========================================================================
# Speech-to-text — browser mic clip
# ==========================================================================
@app.post("/api/stt")
async def stt(audio: UploadFile = File(...)):
    a = STATE["assistant"]
    if a is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    import tempfile
    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    loop = asyncio.get_running_loop()
    try:
        with stopwatch("stt"):
            text = await loop.run_in_executor(None, a.stt.transcribe_file, path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return {"text": text or ""}


# ==========================================================================
# Vision — background analysis, never on the reply path
# ==========================================================================
def _analyze_bg(frame) -> None:
    """Runs on a worker thread. Stores the newest perception + feeds memory."""
    a = STATE["assistant"]
    try:
        with stopwatch("vision_vlm"):
            raw = a.vision.engine.analyze_frame(frame)
        perception = a.vision._to_perception(raw)
        STATE["perception"] = {"summary": perception.summary, "ts": time.time(),
                               "raw": perception.raw}
        a.memory.observe("perception", perception.summary)
        log.info("👁️  %s", perception.summary[:90])
    except Exception as e:  # noqa
        log.debug("vision analyze failed: %s", e)
    finally:
        STATE["vision_busy"] = False


@app.post("/api/vision/frame")
async def vision_frame(frame: UploadFile = File(...), force: str = Form("false")):
    """Accept a webcam frame. Returns immediately; analysis happens in the
    background so the UI never stalls on a 3-5s VLM call."""
    a = STATE["assistant"]
    if a is None or a.vision.engine is None:
        return {"accepted": False, "reason": "vision offline"}

    import cv2
    import numpy as np
    buf = np.frombuffer(await frame.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return {"accepted": False, "reason": "bad frame"}

    forced = str(force).lower() in ("1", "true", "yes")
    if STATE["vision_busy"]:
        return {"accepted": False, "reason": "busy"}

    # The scene-change gate is the whole cost story: without it every frame is a
    # VLM call. Always feed the detector so its baseline tracks what we actually
    # analysed — a forced frame that skipped the update left the baseline stale,
    # making the very next frame read as a big change and fire a needless call.
    score = STATE["scene"].changed(img)
    if not forced and score < config.vision.scene_change_threshold:
        return {"accepted": False, "reason": "no change", "score": round(score, 3)}

    STATE["vision_busy"] = True
    threading.Thread(target=_analyze_bg, args=(img,), daemon=True).start()
    return {"accepted": True, "forced": forced, "score": round(score, 3)}


@app.get("/api/vision/latest")
async def vision_latest():
    p = STATE["perception"]
    return {
        "busy": STATE["vision_busy"],
        "perception": ({"summary": p["summary"],
                        "age_s": round(time.time() - p["ts"], 1),
                        "raw": p.get("raw")} if p else None),
    }


# ==========================================================================
# Memory
# ==========================================================================
@app.get("/api/memory")
async def memory_list(limit: int = 50):
    a = STATE["assistant"]
    if a is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    rows = sorted(a.memory.store.all(), key=lambda r: r.ts, reverse=True)[:limit]
    return {
        "count": len(a.memory.store),
        "records": [{"id": r.id[:8], "text": r.text, "kind": r.kind,
                     "age_s": round(time.time() - r.ts)} for r in rows],
        "episodic": [{"kind": e.kind, "text": e.text, "age_s": round(e.age())}
                     for e in a.memory.episodic.recent(limit=12)],
    }


@app.post("/api/memory/search")
async def memory_search(payload: dict):
    a = STATE["assistant"]
    q = (payload.get("query") or "").strip()
    if a is None or not q:
        return JSONResponse({"error": "bad request"}, status_code=400)
    qv = a.memory.embeddings.encode([q])[0]
    hits = a.memory.store.search(qv, 8)
    thr = config.memory.relevance_threshold
    return {
        "query": q,
        "gate_open": a.memory.needs_recall(q),
        "threshold": thr,
        "hits": [{"text": r.text, "score": round(float(s), 3),
                  "shown": bool(s >= thr), "kind": r.kind} for r, s in hits],
        "context": a.memory.format_context(q),
    }


@app.delete("/api/memory/{record_id}")
async def memory_delete(record_id: str):
    a = STATE["assistant"]
    if a is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    for r in a.memory.store.all():
        if r.id.startswith(record_id):
            a.memory.store.delete(r.id)
            return {"deleted": r.id[:8], "count": len(a.memory.store)}
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/reset")
async def reset_conversation():
    """Clear the chat history without touching durable memory."""
    a = STATE["assistant"]
    if a is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    a.brain._history.clear()
    STATE["turns"] = 0
    return {"ok": True}


# ==========================================================================
# Static frontend
# ==========================================================================
if _WEB.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_WEB)), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(str(_WEB / "index.html"))


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8100")), log_level="warning")


if __name__ == "__main__":
    main()
