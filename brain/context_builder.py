"""
Context fusion.

Assembles the single text block the LLM reasons over, merging every modality:
the current visual scene, recent episodic events, retrieved long-term memory,
conversation history, and the user's utterance. Kept as a pure function so it's
trivial to test and cheap to call on the hot path.
"""
from __future__ import annotations

import json
from typing import Optional, Dict


def build_context(
    user_input: str,
    conversation_history: Optional[str] = None,
    vision_data: Optional[Dict] = None,
    audio_data: Optional[str] = None,
    memory_context: Optional[str] = None,
    scene_summary: Optional[str] = None,
) -> str:
    parts = []

    if conversation_history:
        parts.append(f"CONVERSATION SO FAR:\n{conversation_history}")

    if scene_summary:
        parts.append(f"WHAT YOU CURRENTLY SEE:\n{scene_summary}")
    elif vision_data:
        try:
            parts.append("WHAT YOU CURRENTLY SEE:\n" + json.dumps(vision_data, indent=2))
        except Exception:
            parts.append(f"WHAT YOU CURRENTLY SEE:\n{vision_data}")

    if audio_data and audio_data != user_input:
        parts.append(f"HEARD:\n{audio_data}")

    if memory_context:
        parts.append(memory_context)

    parts.append(f"USER JUST SAID:\n{user_input}")

    return "\n\n---\n\n".join(parts)
