"""
Long-term memory admin — inspect, audit and prune what the assistant "knows".

Durable memory is the one part of this system that silently accumulates
garbage: a consolidation pass that promotes "a ceiling fan and curtains were
visible" to a permanent fact will later have the assistant assert it as if it
were true right now. This is the tool for looking at that and cleaning it up.

    python tools_memory.py list                 # everything, newest first
    python tools_memory.py search "my demo"     # what a query would retrieve
    python tools_memory.py audit                # flag transient/duplicate junk
    python tools_memory.py prune                # delete what audit flagged
    python tools_memory.py forget <id-prefix>   # delete one record
"""
from __future__ import annotations

import re
import sys

from core.config import config
from core.logging_setup import setup_logging
from memory.memory_manager import MemoryManager, _fmt_age
import time

# Scene-description language that should never have become a durable fact.
_TRANSIENT = re.compile(
    r"\b(wearing|shirt|headphones|sitting|standing|posture|hand on|"
    r"in the background|visible|ceiling fan|curtain|wall|walls|lighting|lit|"
    r"room contains|is holding|selfie|looking (at|towards)|appears to be|"
    r"partially|frame|camera|indoors|outdoors|the scene (is|was))\b", re.I)


def _mm() -> MemoryManager:
    setup_logging("WARNING")
    return MemoryManager(config.memory, config.llm)


def _rows(mm):
    return sorted(mm.store.all(), key=lambda r: r.ts, reverse=True)


def cmd_list(mm, _args):
    rows = _rows(mm)
    print(f"{len(rows)} durable memories in {config.memory.persist_dir}\n")
    for r in rows:
        print(f"  {r.id[:8]}  {r.kind:9} {_fmt_age(time.time()-r.ts):>8}  {r.text}")


def cmd_search(mm, args):
    q = " ".join(args) or "?"
    print(f"query: {q!r}")
    print(f"  gate: needs_recall={mm.needs_recall(q)}  "
          f"threshold={config.memory.relevance_threshold}\n")
    qv = mm.embeddings.encode([q])[0]
    for r, score in mm.store.search(qv, 8):
        keep = "SHOWN " if score >= config.memory.relevance_threshold else "filtered"
        print(f"  [{keep}] {score:.3f}  {r.text}")
    print("\n--- what the LLM would actually receive ---")
    print(mm.format_context(q) or "(nothing — no memory injected)")


def _flag(mm):
    """Return [(record, reason)] for records that pollute retrieval."""
    rows = _rows(mm)
    flagged, seen = [], []
    for r in rows:
        if _TRANSIENT.search(r.text):
            flagged.append((r, "transient scene description"))
            continue
        vec = mm.embeddings.encode([r.text])[0]
        dup = next((o for o, v in seen
                    if float(v @ vec) >= config.memory.dedupe_threshold), None)
        if dup is not None:
            flagged.append((r, f"duplicate of {dup.id[:8]}"))
            continue
        seen.append((r, vec))
    return flagged


def cmd_audit(mm, _args):
    flagged = _flag(mm)
    total = len(mm.store)
    print(f"{len(flagged)} of {total} memories look like junk:\n")
    for r, why in flagged:
        print(f"  {r.id[:8]}  [{why}]  {r.text}")
    print(f"\n{total - len(flagged)} would remain. Run `prune` to delete the above.")


def cmd_prune(mm, _args):
    flagged = _flag(mm)
    if not flagged:
        print("nothing to prune")
        return
    for r, why in flagged:
        mm.store.delete(r.id)
        print(f"  deleted {r.id[:8]}  ({why})  {r.text[:60]}")
    print(f"\npruned {len(flagged)}; {len(mm.store)} memories remain")


def cmd_forget(mm, args):
    if not args:
        print("usage: forget <id-prefix>")
        return
    for r in _rows(mm):
        if r.id.startswith(args[0]):
            mm.store.delete(r.id)
            print(f"deleted {r.id[:8]}: {r.text}")
            return
    print("no match")


_CMDS = {"list": cmd_list, "search": cmd_search, "audit": cmd_audit,
         "prune": cmd_prune, "forget": cmd_forget}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    fn = _CMDS.get(cmd)
    if fn is None:
        print(__doc__)
        sys.exit(2)
    fn(_mm(), sys.argv[2:])


if __name__ == "__main__":
    main()
