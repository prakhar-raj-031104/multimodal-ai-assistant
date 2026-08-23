# Manual test guide

Everything below was run against this machine on 2026-08-23. Times are what you
should roughly expect; anything wildly slower is almost always the Groq free-tier
token-per-minute cap (see [Troubleshooting](#troubleshooting)).

```bash
cd /home/mahadev/Documents/Multimodal-ai-assistance-project/multimodal-ai-assistant
source venv/bin/activate
```

---

## 0. Automated health check (start here — 30s)

```bash
python backend/main.py --selftest ; echo "exit=$?"
```

Every line must read `[PASS]` and `exit=0`. This boots the real assistant and
makes real API calls, so it catches dead model IDs, a bad key, and a vision model
your HF account can't reach. A `[FAIL]` line names the broken subsystem.

Expected:

```
[PASS] Groq client (STT/TTS/LLM)
[PASS] STT backend                    groq
[PASS] TTS backend                    spd (local fallback: spd)
[PASS] Embeddings                     sentence_transformers (dim=384)
[PASS] Vector store                   40 durable facts
[PASS] Agent tools                    5 registered
[PASS] LLM (groq/openai/gpt-oss-120b) -> 'ready'
[PASS] Vision (Qwen/Qwen3-VL-30B-A3B-Instruct) -> 'A plain gray background with...'
[PASS] Memory relevance gate          general-knowledge skipped=True, recall-query retrieves=True
=== OK ===
```

---

## 1. Text mode — brain, tools and memory (no hardware)

```bash
python backend/main.py --text
```

Type these in order. The right-hand column is the behaviour under test — if you
get something else, that's the bug.

| Type this | Must do this |
|---|---|
| `What is 12 squared?` | `144.` — answers general knowledge directly, **without** refusing |
| `What is the capital of Japan?` | `Tokyo.` — no memory lookup, no "I can't see that" |
| `Remember I like oat milk.` | logs `🛠️ save_memory(...)`, replies **one** short sentence |
| `What milk do I like?` | `You like oat milk.` |
| `What did I ask you to remember?` | recalls the oat milk (meta-question → recency fallback) |
| `What time is it?` | logs `🛠️ get_current_time()`, gives the real time |
| `Set a reminder to stretch in 15 minutes.` | logs `🛠️ set_reminder(...)`, confirms the clock time |
| `What colour is the mug on my desk?` | *"I can't see that right now"* — must **not** invent a colour |
| `quit` | prints the p50/p95 latency table |

**Three failure modes to watch for specifically:**

- **Filler / repeats.** Each reply must be *one* sentence. If you see
  `Got it.Sure thing!You're welcome!...` the tool loop is re-entering.
- **Refusing general knowledge.** "What is the capital of Japan?" answering
  "I can't see that right now" means the grounding prompt is over-clamped.
- **Leaked memory.** Ask `What is 17 times 3?` — the answer must be `51.` with no
  mention of anything you told it earlier.

Latency: replies should land in **0.3–2s**. Watch the `[turn]` breakdown lines.

---

## 2. Memory inspection — what it actually stored

```bash
python tools_memory.py list                  # every durable fact, newest first
python tools_memory.py search "oat milk"     # scores + exactly what the LLM receives
python tools_memory.py search "capital of France"
python tools_memory.py audit                 # flags transient/duplicate junk
```

`search "oat milk"` should print `gate: needs_recall=True` and show the fact
above the threshold. `search "capital of France"` should print
`needs_recall=False` and end with `(nothing — no memory injected)`.

**Your store currently has 15 junk records** (8 transient scene descriptions like
*"A ceiling fan and curtains were visible"*, 7 duplicate reminders) written by the
old consolidation prompt. They are the direct cause of the assistant asserting
stale room details. Clean them out — this deletes data, so it's yours to run:

```bash
cp -r data/memory data/memory.bak    # keep a copy first
python tools_memory.py prune
```

Verify write-side dedupe still holds afterwards:

```bash
python -c "
from memory.memory_manager import MemoryManager
from core.config import config
m=MemoryManager(config.memory, config.llm)
print('new      :', m.remember_fact('The user commutes by train.'))   # True
print('duplicate:', m.remember_fact('The user commutes by train.'))   # False
print('paraphrase:', m.remember_fact('The user takes the train to work.'))  # False
"
```

---

## 3. Web dashboard — camera + mic in the browser

```bash
python ui/gradio_app.py     # open http://127.0.0.1:7860
```

1. **Type** "Say hello" → reply appears in the chat **and** plays as audio
   (gTTS). Check the 🔊 player actually autoplays.
2. **Camera** — allow webcam access, hold something with readable text in frame,
   capture, then ask *"What text do you see?"*. It must read the real text, and
   the `Vision:` line in the left panel must say `on`.
3. **Mic** — click record, say *"what is the capital of France"*, stop, press
   **Send** with the text box empty. It transcribes via Groq Whisper, then
   answers.
4. **Stats panel** — `Memory facts` should tick up after you tell it to remember
   something; `p50/p95` timings appear after a few turns.

---

## 4. Live mode — always-on mic + camera

```bash
python backend/main.py
```

Confirm the boot lines appear:

```
🟢 assistant live — speak to it (Ctrl-C to stop)
🎙️  mic open @ 16000 Hz, 30ms frames
VAD backend: _SileroVAD          <- Silero, not _EnergyVAD
📷 camera 0 open
```

Then:

- **Speech → answer.** Say *"what time is it"*. You should see `🗣️ what time is
  it`, then a spoken reply, then `[turn] total=... | stt=... respond=...`.
- **Vision gate.** Wave something new in front of the camera. A `👁️` line appears
  with a readable sentence — **not** a wall of `{"scene_summary": ...` JSON. Hold
  still and it should *stop* firing (that's the scene-change gate working).
- **Visual question.** Point the camera at an object and ask *"what am I looking
  at?"* — the router classifies it VISUAL and takes a fresh look.
- **Barge-in.** Ask something that produces a long answer, then start talking
  over it. Playback must cut immediately. (`LOG_LEVEL=DEBUG` prints
  `barge-in detected — cutting playback`.)
- **Ctrl-C** → clean shutdown, `mic closed` / `assistant stopped`.

### Hardware pre-checks if live mode misbehaves

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"   # a '*' marks the default mic
python -c "import cv2; c=cv2.VideoCapture(0); print('camera:', c.isOpened()); c.release()"
spd-say -w "voice output works"                                    # you should hear this
```

---

## 5. Component-level checks

```bash
# STT on a file
python -c "
from core.config import config
from brain.llm_engine import get_groq_client
from speech.stt import STT
s=STT(config.stt, 16000, groq_client=get_groq_client(config.groq_api_key()))
print(s._backend, '->', s.transcribe_file('output.wav'))"

# Record 5s and transcribe it
python test_record.py && python -c "
from core.config import config
from brain.llm_engine import get_groq_client
from speech.stt import STT
print(STT(config.stt,16000,groq_client=get_groq_client(config.groq_api_key())).transcribe_file('output.wav'))"

# Router intent classification
python -c "
from brain.router import Router
from core.config import config
r=Router(cfg=config.llm)
for t in ['what am I looking at?','hey what time is it','yeah ok sure','remind me to leave']:
    print(f'{r.route(t).value:8} <- {t}')"
# expect: visual / answer / ambient / answer

# Provider switch (Gemini is configured on this machine)
LLM_PROVIDER=gemini python backend/main.py --selftest
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Every reply is `(…response interrupted.)` | `LLM_MODEL` no longer exists on Groq | `curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models` and pick a live one |
| Vision check fails `model_not_supported` | that VLM isn't served by a provider enabled on your HF account | `curl -H "Authorization: Bearer $HF_TOKEN" https://router.huggingface.co/v1/models` |
| First token takes 5–15s, then normal again | Groq free tier caps tokens/minute (8000 on this key) and **queues** over-cap requests instead of erroring | wait ~60s, use `openai/gpt-oss-20b`, or switch `LLM_PROVIDER=gemini` |
| No voice output | `TTS_BACKEND=groq` needs org terms acceptance for Orpheus | `TTS_BACKEND=spd` (already set in `.env`); it also auto-falls back at runtime |
| `VAD backend: _EnergyVAD` in live mode | torch/silero didn't load | `pip install torch silero-vad`; energy VAD works but triggers on noise |
| Assistant states stale room details as current | junk in long-term memory | `python tools_memory.py audit` then `prune` |
