# multimodal-ai-assistant
Multimodal AI assistant with vision, speech, memory and agent capabilities
# 🚀 Multimodal AI Assistant

A modular **multimodal AI pipeline** that integrates:

* 📸 Vision (Qwen VL model via Hugging Face)
* 🎤 Audio (Local Whisper speech-to-text)
* 🧠 Memory (RAG-based retrieval)
* 🤖 LLM (Groq API)

---

## 🧠 System Overview

The pipeline works as:

```
User Input (Text / Audio / Vision)
        ↓
Multimodal Request Wrapper
        ↓
Context Builder
        ↓
Prompt Manager
        ↓
LLM (Groq)
        ↓
Final Response
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```
git clone <your-repo-url>
cd multimodal-ai-assistant
```

---

### 2️⃣ Create Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Install Whisper (IMPORTANT)

Whisper is **not included in requirements.txt**, install manually:

```
pip install openai-whisper
```

Also install FFmpeg (required):

```
sudo apt install ffmpeg
```

---

### 5️⃣ Setup Environment Variables

Create a `.env` file in root directory:

```
HUGGINGFACE_API_KEY=your_hf_token_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Running the Project

```
python3 backend/main.py
```

You will be prompted:

* Enter text input (optional)
* Camera will capture image
* Microphone will record audio

---

## 🎤 Audio Notes

* Uses **local Whisper model**
* First run may take time (model download)
* Default model: `base` (fast + efficient)

---

## 📸 Vision Notes

* Uses **Qwen2.5-VL via Hugging Face API**
* Requires valid HF token

---

## 🧠 Memory System

* Retrieves context using internal memory module
* Used to enhance responses (RAG-based)

---

## 📁 Project Structure (Simplified)

```
backend/
    main.py

brain/
    processor.py
    context_builder.py
    prompt_manager.py

vision/
    vision_adapter.py

speech/
    speech_engine.py
    local_speech_to_text.py

memory/
    memory_manager.py
```

---

## ⚠️ Important Notes

* Ensure microphone permissions are enabled
* Ensure webcam is accessible
* Internet is required for:

  * Vision model (HF)
  * LLM (Groq)

---

## ✅ Status

✔ Vision working
✔ Audio working (local)
✔ Full pipeline integrated

---

## 🚀 Future Improvements

* Real-time streaming audio
* Multimodal fusion layer
* Better memory retrieval
* UI integration

---

## 👥 Contributors

* Team Multimodal AI Assistant

---
