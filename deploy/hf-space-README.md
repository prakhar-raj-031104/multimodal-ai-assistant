---
title: OPTIC — Multimodal Second-Brain Assistant
emoji: 👁️
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Real-time assistant that sees, hears and remembers.
---

# 👁️ OPTIC

A real-time multimodal assistant. Share your camera and microphone and ask it
anything — it grounds every answer in what it can actually see and hear, and
remembers what matters across sessions.

**How to use:** open the console, click *Enable camera*, then type or hold the
mic button and speak. Ask *"what do you see?"*, *"remember that…"*, *"what time
is it?"*.

Everything runs in your browser for capture and playback; the server only does
reasoning, transcription and vision.

### Under the hood

| Stage | What it does |
|---|---|
| Scene-change gate | A perceptual diff decides whether a frame is worth a vision call — a still room sends nothing |
| Voice activity | The browser captures speech; Groq Whisper transcribes it |
| Relevance gate | Long-term memory is searched only when the question needs it, and only above-threshold facts reach the prompt |
| Single-pass stream | One LLM call with tools attached; tokens stream straight to the page and to speech synthesis |

Built with FastAPI, GSAP and Lenis. Source: see the repository linked below.
