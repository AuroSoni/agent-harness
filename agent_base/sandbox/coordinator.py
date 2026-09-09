"""Consumer-owned coordination of sandbox bindings and session activity.

The harness has no database dependency here. Coordinators must serialize
authoritative binding changes, fence lost ownership, and release a caller's
shared activity before entering an exclusive operation (no lock upgrades).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, AsyncIterator, Protocol

from .sandbox_types import Sandbox


class SandboxCoordinator(Protocol):
    async def ensure_ready(self, agent: Any) -> Sandbox: ...

    async def validate_resident(self, agent: Any) -> None: ...

    def activity(self, agent: Any) -> AsyncContextManager[None]: ...

    def turn(self, agent: Any) -> AsyncContextManager[None]: ...

    def exclusive(self, agent: Any, *, reason: str) -> AsyncContextManager[None]: ...

    async def pause(self, agent: Any, *, epoch: int | None = None) -> bool: ...

    async def destroy(self, agent: Any) -> None: ...

    async def record_checkpoint(self, agent: Any, manifest: Any) -> None: ...

    async def reset(self, agent: Any, *, manifest_ref: str | None) -> Sandbox: ...

    async def finish_reset(self, agent: Any) -> None: ...


@asynccontextmanager
async def uncoordinated() -> AsyncIterator[None]:
    """Async no-op context for consumers that do not inject coordination."""
    yield
