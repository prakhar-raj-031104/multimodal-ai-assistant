"""
Backward-compatible shim.

The live, always-on assistant now lives in `core.runtime.Assistant`. This module
re-exports it so older imports keep working. Prefer:

    from core.runtime import Assistant
    from core.config import config
    Assistant(config)
"""
from core.runtime import Assistant  # noqa: F401

__all__ = ["Assistant"]
