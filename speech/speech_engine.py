import sounddevice as sd
import numpy as np
import tempfile
from scipy.io.wavfile import write

from speech.speech_to_text import LocalSpeechToText


class SpeechEngine:

    def __init__(self):
        self.sample_rate = 16000
        self.duration = 5  # seconds
        self.stt = LocalSpeechToText()

    def record_audio(self) -> str:
        """
        Record audio from mic and save to temp file
        """
        print("🎤 Recording... Speak now")

        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, self.sample_rate, recording)

        print(f"✅ Saved audio: {temp_file.name}")
        return temp_file.name

    def transcribe(self) -> str:
        """
        Full pipeline: record → transcribe
        """
        audio_path = self.record_audio()

        print("🧠 Converting speech to text...")
        text = self.stt.transcribe(audio_path)

        print(f"📝 You said: {text}")
        return text
