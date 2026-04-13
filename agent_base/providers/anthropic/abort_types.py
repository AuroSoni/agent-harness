"""Anthropic-specific types for abort/steer functionality.

StreamResult wraps the Anthropic streaming response with cancellation
metadata (completed_blocks maps to content_block_stop events).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.core.messages import Message


@dataclass
class StreamResult:
    """Result from the Anthropic streaming layer, enriched with cancellation metadata.

    When ``was_cancelled`` is True, ``completed_blocks`` indicates which
    content block indices received a ``content_block_stop`` event before
    the stream was interrupted. The sanitizer uses this to determine which
    blocks are safe to keep.
    """
    message: Message
    completed_blocks: set[int] = field(default_factory=set)
    was_cancelled: bool = False
