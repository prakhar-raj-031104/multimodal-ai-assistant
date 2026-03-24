from typing import Optional


def build_context(
    user_input: str,
    conversation_history: Optional[str] = None,
    vision_data: Optional[str] = None,
    audio_data: Optional[str] = None,
    memory_data: Optional[str] = None,
) -> str:
    """
    Build structured context for the AI assistant.
    """

    context_parts = []

    # Add conversation history if available
    if conversation_history:
        context_parts.append(f"Conversation History:\n{conversation_history}")

    # Add current user input
    context_parts.append(f"Current Request:\n{user_input}")

    # Add vision data if available
    if vision_data:
        context_parts.append(f"Detected Scene:\n{vision_data}")

    # Add audio data if available
    if audio_data:
        context_parts.append(f"Detected Audio Transcript:\n{audio_data}")

    # Add memory data if available
    if memory_data:
        context_parts.append(f"Relevant Memory:\n{memory_data}")

    # Join all parts cleanly
    context = "\n\n".join(context_parts)

    return context
