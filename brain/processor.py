"""
The Brain: high-level reasoning entrypoint.

Owns conversation history and turns a user utterance (+ current scene + memory)
into a streamed spoken answer. This is what the runtime calls for every turn.
"""
from __future__ import annotations

from collections import deque
from typing import Iterator, Optional

from core.config import LLMConfig
from core.logging_setup import get_logger
from brain.context_builder import build_context
from brain.prompt_manager import SYSTEM_PROMPT

log = get_logger("brain")


class Brain:
    def __init__(self, provider, cfg: LLMConfig, memory=None) -> None:
        self.provider = provider
        self.cfg = cfg
        self.memory = memory
        # Conversation history as native chat messages (user/assistant pairs).
        # Passed to the model as real turns — NOT flattened into the prompt text —
        # which keeps the current question from bleeding into old context.
        self._history: deque = deque(maxlen=cfg.max_history_turns * 2)

    def answer_stream(self, user_input: str, scene_summary: Optional[str] = None,
                      instruction: Optional[str] = None) -> Iterator[str]:
        memory_context = self.memory.format_context(user_input) if self.memory else None

        # Current turn only — history is delivered separately as chat messages.
        context = build_context(
            user_input=user_input,
            scene_summary=scene_summary,
            memory_context=memory_context,
        )
        if instruction:
            context += f"\n\n(Answer style: {instruction})"

        history = list(self._history)
        collected = []
        for chunk in self.provider.answer_stream(SYSTEM_PROMPT, history, context):
            collected.append(chunk)
            yield chunk

        reply = "".join(collected).strip()
        self._record_turn(user_input, reply)

    def _record_turn(self, user_input: str, reply: str) -> None:
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": reply})
        if self.memory:
            self.memory.observe("reply", reply)

    def _history_text(self) -> str:
        lines = []
        for m in list(self._history):
            who = "User" if m["role"] == "user" else "You"
            lines.append(f"{who}: {m['content']}")
        return "\n".join(lines)


# -- Backward-compatible functional API (used by old tests / scripts) --------
def process_user_query(user_input, conversation_history=None, vision_data=None,
                       memory_data=None, instruction=None, audio_data=None,
                       audio_file_path=None) -> dict:
    """Legacy one-shot helper. Prefer the streaming Brain in the live runtime."""
    from brain.prompt_manager import build_prompt
    from brain.llm_engine import GroqLLM
    from core.config import config

    context = build_context(
        user_input=user_input or audio_data or "",
        conversation_history=conversation_history,
        vision_data=vision_data,
        audio_data=audio_data,
        memory_context=(f"RELEVANT MEMORY:\n{memory_data}" if memory_data else None),
    )
    prompt = build_prompt(context, instruction)
    llm = GroqLLM(config.llm)
    response = "".join(llm.stream([{"role": "user", "content": prompt}]))
    return {"response": response, "context_used": context, "audio_transcript": audio_data}
