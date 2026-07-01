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

_PROMPT = """You compress a stream of an AI assistant's observations into durable memory.
From the log below, extract 0-5 concise, self-contained facts worth remembering
long-term (people, preferences, commitments, locations, notable events).
Ignore transient noise. Each fact must stand alone without the log.

Return STRICT JSON: {"facts": ["...", "..."]}. Empty list if nothing is worth keeping.

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
