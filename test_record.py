import sounddevice as sd
import scipy.io.wavfile as wav

fs = 16000
duration = 5

print("🎤 Recording...")

recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
sd.wait()

wav.write("output.wav", fs, recording)

print("✅ Saved as output.wav")
