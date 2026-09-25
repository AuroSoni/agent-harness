"""Bounded UTF-8 tails shared by execution backends and SDK adapters."""
from __future__ import annotations

from collections import deque
from typing import Iterator

DEFAULT_CAPTURE_BYTES = 2_000_000
HELPER_CAPTURE_BYTES = 8 * 1024 * 1024


class SandboxOutputLimitExceeded(RuntimeError):
    """Machine-readable command output exceeded its capture budget."""


class Utf8Tail:
    """List-like append/iteration adapter; accounting is bytes, not codepoints."""

    def __init__(self, limit: int = DEFAULT_CAPTURE_BYTES) -> None:
        if limit < 1:
            raise ValueError("capture_limit_bytes must be positive")
        self.limit = limit
        self.total_bytes = 0
        self.size_bytes = 0
        self._parts: deque[tuple[str, int]] = deque()

    @property
    def truncated(self) -> bool:
        return self.total_bytes > self.size_bytes

    def append(self, text: str) -> None:
        raw = text.encode("utf-8", errors="replace")
        size = len(raw)
        self.total_bytes += size
        if not size:
            return
        if size >= self.limit:
            text = raw[-self.limit:].decode("utf-8", errors="ignore")
            size = len(text.encode("utf-8"))
            self._parts.clear()
            self.size_bytes = 0
        self._parts.append((text, size))
        self.size_bytes += size
        while self.size_bytes > self.limit:
            first, first_size = self._parts.popleft()
            excess = self.size_bytes - self.limit
            self.size_bytes -= first_size
            if first_size > excess:
                rest = first.encode("utf-8")[excess:].decode("utf-8", errors="ignore")
                rest_size = len(rest.encode("utf-8"))
                self._parts.appendleft((rest, rest_size))
                self.size_bytes += rest_size

    def __iter__(self) -> Iterator[str]:
        return (part for part, _ in self._parts)

    def __bool__(self) -> bool:
        return self.total_bytes > 0

    def text(self) -> str:
        return "".join(self)
