from typing import Optional, Dict
import json

def build_context(
    user_input: str,
    conversation_history: Optional[str] = None,
    vision_data: Optional[Dict] = None,
    audio_data: Optional[str] = None,
    memory_data: Optional[str] = None
) -> str:
    """
    Build structured multimodal context for the AI assistant
    """

    context_parts = []

    # 🧾 Conversation history
    if conversation_history:
        context_parts.append(f"Conversation History:\n{conversation_history}")

    # 🧑 User input
    context_parts.append(f"Current User Request:\n{user_input}")

    # 🎤 Audio input
    if audio_data:
        context_parts.append(f"Audio Transcription:\n{audio_data}")

    # 👁️ Vision input (structured JSON → formatted)
    if vision_data:
        try:
            vision_pretty = json.dumps(vision_data, indent=2)
        except:
            vision_pretty = str(vision_data)

        context_parts.append(f"Visual Understanding:\n{vision_pretty}")

    # 🧠 Memory (RAG)
    if memory_data:
        context_parts.append(f"Relevant Memory:\n{memory_data}")

    # 🧩 Final structured context
    context = "\n\n---\n\n".join(context_parts)

    return context