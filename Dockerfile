# ---------------------------------------------------------------------------
# OPTIC web console — built for Hugging Face Spaces (sdk: docker), and works
# unchanged for local `docker run` / any container host.
#
# Torch-free by design (see requirements-web.txt): the browser handles mic
# capture and speech synthesis, Groq handles STT, and the HF router handles
# vision — so nothing in the request path needs a local model. That is the
# difference between a ~540MB image and a ~4.5GB one.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_BACKEND=fastembed \
    HOST=0.0.0.0 \
    PORT=7860

# opencv-python-headless still needs a couple of shared libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as uid 1000. Match it so the persistent /data mount
# and the model cache are writable at runtime.
RUN useradd -m -u 1000 user && mkdir -p /data/memory && chown -R user:user /data

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    HF_HOME=/home/user/.cache/huggingface \
    MEMORY_DIR=/data/memory

WORKDIR /home/user/app

# Dependencies first so code edits don't invalidate the wheel layer.
COPY --chown=user:user requirements-web.txt .
RUN pip install --user -r requirements-web.txt

# Bake the ONNX embedding model into the image. Downloading it on first request
# would otherwise add ~15s to the very first question a visitor asks.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(); print('embedding model cached')"

COPY --chown=user:user core/    ./core/
COPY --chown=user:user brain/   ./brain/
COPY --chown=user:user memory/  ./memory/
COPY --chown=user:user speech/  ./speech/
COPY --chown=user:user vision/  ./vision/
COPY --chown=user:user agents/  ./agents/
COPY --chown=user:user api/     ./api/
COPY --chown=user:user web/     ./web/
COPY --chown=user:user tools_memory.py ./

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT','7860')}/api/status\",timeout=4).status==200 else 1)"

# `exec` so uvicorn replaces the shell and becomes PID 1 — otherwise it never
# receives SIGTERM and every stop waits out the 10s kill timeout.
CMD ["sh", "-c", "exec uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
