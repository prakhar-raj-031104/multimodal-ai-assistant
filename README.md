# 🧠👓 Real-time Multimodal Second-Brain Assistant

An always-on, "Meta AI glasses"-style assistant. It **continuously sees** your
surroundings and **hears** what's said around you, fuses that into a temporal
memory, and answers your spoken questions in real time — grounded in what it
just saw, heard, and remembers.

Built for **low latency**: the entire hot path (speech-to-text → LLM →
text-to-speech) rides a single fast cloud provider (**Groq**), the LLM answer is
**streamed and spoken sentence-by-sentence**, and the VLM only fires when the
scene actually changes. Every heavy dependency has a graceful fallback, so it
runs on a bare install and gets better as you add optional models.

---

## Architecture

```
   ┌─ camera ──▶ scene-change gate ──▶ VLM (Qwen2.5-VL) ──▶ Perception ─┐
   │                                                                     ▼
   │   mic ──▶ VAD segmenter ──▶ STT (Whisper turbo) ─┐         ┌──── Memory ────┐
   │              │                                    ▼         │  episodic (2m) │
   │         wake word ─────────────────────────▶ Router ──▶ Brain ── vector RAG │
   │              │                                    │       │  consolidation │
   └─ TTS ◀── streamed LLM answer ◀── Reasoner+Tools ◀─┘         └────────────────┘
              (barge-in aware)
```

Everything runs concurrently on one `asyncio` loop; blocking work (network,
audio playback) is offloaded to a thread pool so the loop stays free to detect
**barge-in** and cut playback the instant you start talking.

| Layer | What it does | Key files |
|-------|--------------|-----------|
| **Core** | async event bus, config, logging, latency tracing | [core/](core/) |
| **Audio in** | continuous capture → VAD segments → wake word | [speech/audio_stream.py](speech/audio_stream.py), [speech/vad.py](speech/vad.py), [speech/wake_word.py](speech/wake_word.py) |
| **STT** | Groq `whisper-large-v3-turbo` (+ local fallback) | [speech/stt.py](speech/stt.py) |
| **Vision** | camera loop, scene-change gating, VLM | [vision/vision_stream.py](vision/vision_stream.py), [vision/scene_change.py](vision/scene_change.py), [vision/vision_engine.py](vision/vision_engine.py) |
| **Memory** | episodic buffer + vector RAG + LLM consolidation | [memory/](memory/) |
| **Brain** | router, context fusion, streaming LLM, agentic tools | [brain/](brain/), [agents/tools.py](agents/tools.py) |
| **TTS** | Groq PlayAI streaming TTS (+ offline fallback), barge-in | [speech/tts.py](speech/tts.py) |
| **Runtime** | wires it all into an always-on loop | [core/runtime.py](core/runtime.py) |

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # then add your GROQ_API_KEY and HF_TOKEN
```

`GROQ_API_KEY` powers the LLM, STT, and TTS (one key, one fast path). `HF_TOKEN`
powers the vision model. Get a Groq key at **console.groq.com**.

> **Recommended installs** (big quality/latency wins): `sentence-transformers`
> for real semantic memory and `torch` + `silero-vad` for accurate speech
> detection. Without them the system still runs on hash-based memory + an energy
> VAD.

---

## Run

```bash
uvicorn api.server:app --port 8100  # 🌐 web console — camera + mic in the browser
python backend/main.py             # 🟢 LIVE — server mic + camera, always-on
python backend/main.py --text      # 💬 typed queries, no hardware needed
python backend/main.py --selftest  # ✅ boot & health-check every subsystem (exit 1 on failure)
python tools_memory.py audit       # 🧠 inspect / clean long-term memory
python ui/gradio_app.py            # 🖥️  legacy Gradio dashboard
```

### The web console

`uvicorn api.server:app --port 8100` → **http://127.0.0.1:8100**

A custom frontend ([web/](web/)) over a streaming API ([api/server.py](api/server.py)).
Camera and microphone come from the *browser*, so the server needs no hardware:

- **Streaming answers** — tokens render at time-to-first-token (~300ms) over SSE,
  not when the full reply lands. Voice output speaks each sentence as it
  completes, so it starts talking while the model is still generating.
- **Vision never blocks chat** — the browser pushes frames, the server
  scene-gates them and analyses in the background. A vision call takes 3–5s; on
  the reply path it would make every grounded answer feel broken.
- **Live retrieval probe** — search any query and see exactly which memories
  clear the relevance bar and what text the model actually receives.

Motion is GSAP + Lenis, vendored locally (no CDN round-trip). `?static=1`
disables it. Deployment lives in **[DEPLOY.md](DEPLOY.md)**.

Step-by-step manual verification (what to type, what must come back, and how to
tell a real bug from a rate limit) lives in **[TESTING.md](TESTING.md)**.

In live mode: just talk. Ask *"what am I looking at?"*, *"remind me to leave in
10 minutes"*, *"what did she just say?"* — it answers out loud. Set
`WAKEWORD_ENABLED=true` to make it respond only after "Hey Jarvis".

**Latency:** every turn prints a breakdown, e.g.
`[turn] total=980ms | stt=310ms respond=640ms`. `--selftest` and text mode print
p50/p95 per stage on exit.

---

## Choosing the brain — Groq or Claude

The LLM is a swappable provider ([brain/providers.py](brain/providers.py)):

> **Model IDs go stale.** Groq retired the `llama-3.x` families — a config
> pointing at `llama-3.3-70b-versatile` now fails with a 404 and the assistant
> answers `(…response interrupted.)`. `--selftest` catches this. To see what
> your key can actually use:
> `curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models`

```bash
# Fast & cheap (default) — GPT-OSS on Groq
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b      # or openai/gpt-oss-20b / qwen/qwen3.6-27b
LLM_REASONING_EFFORT=low           # gpt-oss thinks before speaking: low ≈ 120ms TTFT, medium ≈ 490ms

# Strongest reasoning — Claude (least hallucination)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...        # separate pay-as-you-go key; Claude Pro does NOT include API access
ANTHROPIC_MODEL=claude-opus-4-8     # or claude-sonnet-5 (cheaper)
```
Tool-calling and streaming work identically on both. If `LLM_PROVIDER=anthropic`
but no key is set, it falls back to Groq automatically.

## Memory — how it avoids confident nonsense

Long-term memory is the part of a system like this that quietly rots: a
consolidation pass promotes "a ceiling fan and curtains were visible" to a
permanent fact, and a week later the assistant asserts it as though it were
looking at it. The retrieval path is built around five rules
([memory/memory_manager.py](memory/memory_manager.py)):

1. **Retrieve only when the question needs it.** "What is the capital of
   France?" must not drag in "the user's favourite colour is teal". A lexical
   gate decides; the model can always force a lookup with the `recall_memory`
   tool (`MEMORY_ALWAYS_RETRIEVE=true` restores the old always-on behaviour).
2. **High relevance bar.** Only facts clearly above `MEMORY_RELEVANCE_THRESHOLD`
   (0.45) reach the prompt. Loosely-related facts injected as "MEMORY" are the
   single biggest source of confident nonsense.
3. **Memory is never the present.** Every fact is rendered with its age and
   flagged past `MEMORY_STALE_AFTER_S`, under a heading that says these are past
   events — so a remembered thing can't be reported as something happening now.
4. **The conversation is not memory.** Dialogue reaches the model as native chat
   turns, so it is excluded from the context block instead of being pasted in
   twice (duplication is what makes models blend unrelated earlier turns into
   the current answer).
5. **Write-side hygiene.** Near-duplicate facts are dropped on write
   (`MEMORY_DEDUPE_THRESHOLD`), and consolidation is instructed to keep only
   what is still true days later — names, preferences, commitments — and discard
   scene description, which is most of what it sees.

Inspect and clean what has accumulated:

```bash
python tools_memory.py list                 # everything, newest first
python tools_memory.py search "my demo"     # exactly what a query would retrieve
python tools_memory.py audit                # flag transient/duplicate junk
python tools_memory.py prune                # delete what audit flagged
```

**Elsewhere:** history is passed as native chat turns, stale perceptions aren't
presented as the current view (`SCENE_FRESHNESS_S`), and the system prompt
separates *general world knowledge* (answer it directly) from *claims about the
user's surroundings and history* (context only, never invented). Weaker models
hallucinate more — `openai/gpt-oss-120b` or Claude give the biggest gains.

## Configuration

Everything is tunable via environment variables (see [.env.example](.env.example))
and typed dataclasses in [core/config.py](core/config.py) — LLM provider/model,
STT/TTS/VAD/embedding backends, scene-change sensitivity, VLM rate cap, memory
window, and more.

---

## Design notes & production behaviour

- **Graceful degradation** — missing optional deps or a dead camera/mic never
  crash the system; the affected subsystem logs a warning and disables itself.
- **Cost/latency guards** — VLM calls are gated by scene change *and* a hard FPS
  cap; STT runs only on VAD-detected speech, not the raw mic stream.
- **Persistent memory** — durable facts survive restarts in `data/memory/`, are
  deduplicated on write, and are inspectable/prunable via `tools_memory.py`.
- **One LLM call per turn** — the reasoner streams with tool schemas attached, so
  a reply that needs no tool costs a single call. (It used to make a blocking
  tool-probe call, throw that answer away, and regenerate it — every turn paid
  for two full generations.)
- **Free-tier note** — Groq's free tier caps tokens-per-minute (8k on this key).
  Past the cap, requests are *queued*, not rejected, which shows up as
  multi-second stalls before the first token rather than an error.
- **Privacy** — this records audio/video of people. Keep processing local where
  possible, be transparent, and comply with local consent laws before deploying
  on real glasses.

## Roadmap

Speaker diarization · ambient (non-speech) audio understanding · face
re-identification · proactive assistance · WebRTC glasses↔edge↔server split.
