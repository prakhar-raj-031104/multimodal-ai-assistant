"""
Agent tools exposed to the LLM via function calling.

Each tool is a small Python function plus a JSON schema the LLM uses to decide
when to call it. This is how the "second brain" acts: recall past memories, set
reminders, check the time, search the web. Add capabilities by writing a
function and registering it in build_default_registry().
"""
from __future__ import annotations

import datetime as _dt
import time
from typing import Callable, Dict, List


class ToolRegistry:
    def __init__(self) -> None:
        self._fns: Dict[str, Callable] = {}
        self._schemas: List[dict] = []

    def register(self, schema: dict, fn: Callable) -> None:
        self._fns[schema["function"]["name"]] = fn
        self._schemas.append(schema)

    def schemas(self) -> List[dict]:
        return self._schemas

    def call(self, name: str, args: dict) -> str:
        fn = self._fns.get(name)
        if fn is None:
            return f"error: unknown tool '{name}'"
        try:
            return str(fn(**args))
        except Exception as e:  # noqa
            return f"error calling {name}: {e}"

    def __len__(self) -> int:
        return len(self._fns)


def _schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def build_default_registry(memory=None) -> ToolRegistry:
    reg = ToolRegistry()

    def get_current_time() -> str:
        return _dt.datetime.now().strftime("%A %Y-%m-%d %H:%M:%S")

    reg.register(_schema("get_current_time", "Get the current local date and time.", {}, []),
                 get_current_time)

    def recall_memory(query: str) -> str:
        if memory is None:
            return "memory unavailable"
        facts = (memory.retrieve(query, force=True).get("facts") or [])
        if facts:
            return "\n".join(f"- {f['text']}" for f in facts)
        # Nothing matched semantically. For a meta-question ("what did I ask you
        # to remember?") the wording never matches the stored fact, so fall back
        # to recency — labelled, so the model doesn't treat it as a hit.
        recent = memory.recent_facts(limit=5)
        if not recent:
            return "no memories stored yet"
        lines = "\n".join(f"- {f['text']}" for f in recent)
        return ("no direct match; the most recently recorded memories are:\n"
                + lines)

    reg.register(
        _schema("recall_memory",
                "Search the user's long-term memory for facts about people, preferences, "
                "past events, or things seen/heard earlier.",
                {"query": {"type": "string", "description": "what to recall"}}, ["query"]),
        recall_memory)

    def save_memory(fact: str) -> str:
        if memory is None:
            return "memory unavailable"
        memory.remember_fact(fact, kind="fact", meta={"source": "explicit"})
        return "saved"

    reg.register(
        _schema("save_memory", "Explicitly store an important fact the user wants remembered.",
                {"fact": {"type": "string", "description": "the fact to remember"}}, ["fact"]),
        save_memory)

    def set_reminder(text: str, in_minutes: float = 0.0) -> str:
        due = time.time() + in_minutes * 60
        if memory is not None:
            memory.remember_fact(f"Reminder: {text}", kind="reminder",
                                 meta={"due": due, "source": "reminder"})
        when = _dt.datetime.fromtimestamp(due).strftime("%H:%M")
        return f"reminder set for {when}: {text}"

    reg.register(
        _schema("set_reminder", "Set a reminder for the user.",
                {"text": {"type": "string", "description": "reminder content"},
                 "in_minutes": {"type": "number", "description": "minutes from now"}}, ["text"]),
        set_reminder)

    def web_search(query: str) -> str:
        return _web_search(query)

    reg.register(
        _schema("web_search",
                "Search the web for current/factual information the assistant doesn't know.",
                {"query": {"type": "string", "description": "search query"}}, ["query"]),
        web_search)

    return reg


def _web_search(query: str) -> str:
    """DuckDuckGo instant-answer; no API key. Returns a short text result."""
    try:
        import requests
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=6,
        )
        data = r.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                return topic["Text"]
        return "no concise answer found"
    except Exception as e:  # noqa
        return f"web search unavailable: {e}"
