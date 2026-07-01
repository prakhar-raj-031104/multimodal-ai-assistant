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
python backend/main.py             # 🟢 LIVE — mic + camera, always-on
python backend/main.py --text      # 💬 typed queries, no hardware needed
python backend/main.py --selftest  # ✅ boot & health-check every subsystem
python ui/gradio_app.py            # 🖥️  optional web dashboard (needs gradio)
```

In live mode: just talk. Ask *"what am I looking at?"*, *"remind me to leave in
10 minutes"*, *"what did she just say?"* — it answers out loud. Set
`WAKEWORD_ENABLED=true` to make it respond only after "Hey Jarvis".

**Latency:** every turn prints a breakdown, e.g.
`[turn] total=980ms | stt=310ms respond=640ms`. `--selftest` and text mode print
p50/p95 per stage on exit.

---

## Choosing the brain — Groq or Claude

The LLM is a swappable provider ([brain/providers.py](brain/providers.py)):

```bash
# Fast & cheap (default) — Llama/GPT-OSS on Groq
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant     # or llama-3.3-70b-versatile / openai/gpt-oss-120b

# Strongest reasoning — Claude (least hallucination)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...        # separate pay-as-you-go key; Claude Pro does NOT include API access
ANTHROPIC_MODEL=claude-opus-4-8     # or claude-sonnet-5 (cheaper)
```
Tool-calling and streaming work identically on both. If `LLM_PROVIDER=anthropic`
but no key is set, it falls back to Groq automatically.

**Anti-hallucination:** history is passed as native chat turns (not flattened
into the prompt), stale perceptions aren't presented as the current view
(`SCENE_FRESHNESS_S`), memory retrieval uses a relevance threshold, and the
system prompt forbids inventing unseen details. Weaker models (Groq 8B)
hallucinate more — switch to a 70B model or Claude for the biggest gains.

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
- **Persistent memory** — durable facts survive restarts in `data/memory/`.
- **Privacy** — this records audio/video of people. Keep processing local where
  possible, be transparent, and comply with local consent laws before deploying
  on real glasses.

## Roadmap

Speaker diarization · ambient (non-speech) audio understanding · face
re-identification · proactive assistance · WebRTC glasses↔edge↔server split.
