"""
Memory consolidation.

Raw perceptions and utterances are noisy and voluminous — we can't keep them all
forever. Periodically we ask the fast LLM to distil a batch of recent episodes
into a handful of durable, self-contained facts ("The user is preparing pasta in
the kitchen", "A colleague named Sam mentioned the demo moved to Friday"). Those
facts, not the raw stream, are what get embedded into long-term memory.
"""
from __future__ import annotations

import json
from typing import List

from core.config import LLMConfig
from core.logging_setup import get_logger

log = get_logger("memory.consolidation")

_PROMPT = """You maintain the LONG-TERM memory of an always-on assistant that sees and
hears the wearer's surroundings. You are given a log of raw observations.

Extract ONLY facts that are still true and still useful DAYS from now.

KEEP (durable):
- Identity & relationships: names, roles, who someone is to the user.
- Stable preferences: likes, dislikes, habits, routines, allergies.
- Commitments & plans: meetings, deadlines, appointments, promises, reminders.
- Decisions and outcomes the user would want recalled later.
- Facts the user explicitly asked you to remember.

DISCARD (transient — this is most of the log):
- Anything describing the current scene: furniture, walls, lighting, clothing,
  posture, what someone is holding, who is in frame, room contents.
- Momentary actions ("a person was walking", "someone waved").
- Small talk, greetings, filler, or the assistant's own replies.
- Anything you are not confident about, or that is not stated in the log.

Rules:
- Each fact must be a single self-contained sentence understandable with no log.
- Name the subject explicitly ("The user...", "Sam..."), never "he"/"this".
- Do NOT infer, generalise, or embellish. Copy only what the log states.
- Prefer 0 facts over a weak one. An empty list is a correct, common answer.

Return STRICT JSON: {"facts": ["...", "..."]}.

LOG:
%s
"""


def consolidate(episodes_text: str, groq_client, cfg: LLMConfig) -> List[str]:
    if not episodes_text.strip() or groq_client is None:
        return []
    try:
        resp = groq_client.chat.completions.create(
            model=cfg.fast_model,
            messages=[{"role": "user", "content": _PROMPT % episodes_text}],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        content = resp.choices[0].message.content
        facts = json.loads(content).get("facts", [])
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    except Exception as e:  # noqa
        log.debug("consolidation failed: %s", e)
        return []
