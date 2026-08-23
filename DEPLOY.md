# Deploying the web console

The console is a FastAPI app (`api/server.py`) serving a static frontend
(`web/`). All capture happens in the browser, so **the server needs no camera,
microphone or GPU** — which is what makes it deployable anywhere.

## What gets deployed

`Dockerfile` + `requirements-web.txt` build a deliberately torch-free image:

| | With torch | This image |
|---|---|---|
| Size | ~4.5 GB | **607 MB** |
| Cold start | ~23 s | **4.2 s** |

Nothing in the request path needs a local model — the browser does mic capture
and speech synthesis, Groq does STT, and the HF router does vision. Embeddings
switch from `sentence-transformers` to `fastembed` (ONNX, same 384 dims), and
the model is baked into the image at build time so the first question doesn't
pay a download.

> **Switching embedding backends invalidates an existing store.** The vectors
> are not interchangeable. The store detects the dimension change, archives the
> old files as `.bak`, and starts fresh rather than crashing on first search.

---

## Hugging Face Spaces

### 1. Create the Space

New Space → **SDK: Docker** → *Blank*. Note the repo URL it gives you.

### 2. Push the code

```bash
git clone https://huggingface.co/spaces/<you>/<space-name> hf-space
cd hf-space

# Copy everything the image needs (see .dockerignore for what's excluded)
rsync -a --exclude venv --exclude .git --exclude data --exclude __pycache__ \
      /path/to/multimodal-ai-assistant/ .

# The Space reads its config from README.md frontmatter
cp deploy/hf-space-README.md README.md

git add -A && git commit -m "OPTIC web console" && git push
```

The Space builds the Dockerfile automatically and serves on `app_port: 7860`.

### 3. Set the secrets

Space **Settings → Variables and secrets**, add as *Secrets*:

| Name | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key — powers LLM + STT |
| `HF_TOKEN` | your HF token — powers the vision model |

Never commit `.env`; `.dockerignore` already excludes it.

### 4. Enable persistent storage

Space **Settings → Persistent storage → Small (20 GB)**.

The image already points `MEMORY_DIR` at `/data/memory`, which is where Spaces
mounts the disk, so durable memories survive restarts and rebuilds with no
further config.

> **This is a paid add-on** (~$5/month). Without it the Space still runs
> perfectly — `/data` just resets on every restart, so long-term memory is
> ephemeral. Nothing errors; the assistant simply starts each life with an
> empty store.

### Verify the deploy

```bash
curl https://<you>-<space-name>.hf.space/api/status
```

Every field should be populated and `ready` should be `true`.

---

## Anywhere else (VPS, Fly, Render, local)

```bash
docker build -t optic-console .
docker volume create optic-mem

docker run -d --name optic -p 7860:7860 \
  -v optic-mem:/data \
  -e GROQ_API_KEY=... \
  -e HF_TOKEN=... \
  optic-console
```

`PORT` is honoured if the platform injects one. Put it behind Caddy or nginx
for TLS — and note that **browsers only grant camera and microphone access over
HTTPS** (or on `localhost`), so a plain-HTTP deploy on a public IP will load the
page but silently fail to capture.

### Useful environment variables

| Variable | Default here | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | required |
| `HF_TOKEN` | — | required for vision |
| `EMBEDDING_BACKEND` | `fastembed` | `hash` needs no model at all |
| `MEMORY_DIR` | `/data/memory` | point at your mounted volume |
| `LLM_MODEL` | `openai/gpt-oss-120b` | must exist on your Groq key |
| `LLM_REASONING_EFFORT` | `low` | drives time-to-first-token |
| `VISION_MODEL` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | must be enabled for your HF account |
| `MEMORY_RELEVANCE_THRESHOLD` | `0.45` | raise if it drags in irrelevant facts |
| `PORT` | `7860` | |

---

## Local development

```bash
source venv/bin/activate
uvicorn api.server:app --reload --port 8100    # http://127.0.0.1:8100
```

`?static=1` disables all scroll motion — handy for screenshots and deep links.
