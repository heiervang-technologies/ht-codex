"""Token estimation using byte-length heuristic.

Matches the approach in codex-rs/core/src/truncate.rs:10 — ceil(utf8_bytes / 4).
"""

from __future__ import annotations

import math

from .parser import Turn, extract_text


def estimate_tokens(text: str) -> int:
    """Estimate token count from UTF-8 byte length."""
    return math.ceil(len(text.encode("utf-8")) / 4)


def turn_tokens(turn: Turn) -> int:
    """Estimate the token count of an entire turn."""
    return estimate_tokens(extract_text(turn))
