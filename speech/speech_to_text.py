import whisper
from pathlib import Path


class LocalSpeechToText:
    """
    Local Whisper-based Speech-to-Text
    """

    def __init__(self, model_size: str = "base"):
        print("🧠 Loading Whisper model...")
        self.model = whisper.load_model(model_size)
        print("✅ Whisper loaded")

    def transcribe(self, audio_path: str) -> str:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print("🧠 Running Whisper inference...")
        result = self.model.transcribe(str(audio_path))

        return result["text"].strip()
