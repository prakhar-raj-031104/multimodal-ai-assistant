# multimodal-ai-assistant

Multimodal AI assistant with vision, speech, memory and agent capabilities.

## Audio integration (Voxtral Mini 4B Realtime)

This repository now includes a native audio transcription module at
`speech/speech_to_text.py` using:

- Model: `mistralai/Voxtral-Mini-4B-Realtime-2602`
- Runtime: `transformers` + `mistral-common[audio]`

### How audio is wired into the assistant pipeline

1. `process_user_query(...)` now accepts:
   - `audio_data` (already-transcribed text), or
   - `audio_file_path` (local audio file path).
2. If `audio_file_path` is provided without `audio_data`, the assistant auto-transcribes it with Voxtral.
3. The transcript is added into the shared context as `Detected Audio Transcript`.
4. The final LLM answer is generated from vision + audio + memory + chat context.

### Example usage

```python
from brain.assistant import process_user_query

result = process_user_query(
    user_input="Summarize what was said in the audio.",
    audio_file_path="/path/to/meeting_clip.wav",
    vision_data="A slide that says Q2 Planning",
    instruction="Give a brief summary"
)

print(result["audio_transcript"])
print(result["response"])
```

## Docker

A `Dockerfile` is included and installs the system dependencies needed for audio
processing (`ffmpeg`, `libsndfile1`) in addition to Python packages.

Build and run:

```bash
docker build -t multimodal-ai-assistant .
docker run --rm -it --env-file .env multimodal-ai-assistant
```

> Note: Voxtral model weights are downloaded at runtime on first use.
