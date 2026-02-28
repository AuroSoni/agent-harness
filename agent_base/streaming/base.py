"""Abstract base class for stream formatters.

A ``StreamFormatter`` serializes canonical ``StreamDelta`` objects into a
wire-format and puts the result on an ``asyncio.Queue``.  This decouples
the transport layer from the agent/provider layer.

Concrete implementations live in ``agent_base.streaming.formatters``.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import StreamDelta

# Type alias for formatter names
StreamFormatterType = Literal["json"]


class StreamFormatter(ABC):
    """Abstract base class for stream formatters.

    Implementations serialize ``StreamDelta`` objects to a specific wire
    format (JSON envelopes, raw text, etc.) and put the result on a queue
    for the transport layer to consume.
    """

    @abstractmethod
    async def format_delta(
        self, delta: StreamDelta, queue: asyncio.Queue
    ) -> None:
        """Serialize a StreamDelta and put the result on the queue.

        Args:
            delta: The stream event to serialize.
            queue: The output queue for serialized data.
        """
        ...
