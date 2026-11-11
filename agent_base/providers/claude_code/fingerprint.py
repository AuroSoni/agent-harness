"""Fingerprints of the library's history, so a point in it can be found again in Claude Code's transcript.

A message's hash covers what the model sees: its role and its content blocks in the Anthropic wire form the
formatter produces. The fingerprint of a prefix chains the hashes of every message up to it, so it names one exact
conversation so far: the same history gives the same fingerprint live and after a save and a cold load, and a
fork, which copies its parent's history, finds its parent's points.

It depends on the formatter's wire form: a change to how the formatter renders a block changes every
fingerprint, and the ledger's points stop matching (each session then falls back once).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def message_hash(formatter: Any, message: Any) -> str:
    wire = {"role": message.role.value, "content": formatter.format_blocks_to_wire(message.content)}
    data = json.dumps(wire, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def prefix_fingerprints(formatter: Any, messages: list[Any]) -> list[str]:
    """``out[i]`` is the fingerprint of ``messages[: i + 1]``."""
    out: list[str] = []
    prev = ""
    for m in messages:
        prev = hashlib.sha256((prev + message_hash(formatter, m)).encode("ascii")).hexdigest()
        out.append(prev)
    return out


__all__ = ["message_hash", "prefix_fingerprints"]
