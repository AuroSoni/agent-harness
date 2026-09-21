"""Consumer-owned coordination of sandbox bindings and session activity.

The harness has no database dependency here. Coordinators must serialize
authoritative binding changes, fence lost ownership, and release a caller's
shared activity before entering an exclusive operation (no lock upgrades).

Readiness reporting: while the runtime times a root sandbox warm (the
``sandbox_ready`` trace span, see :mod:`agent_base.core.trace_spans`) it sets
:data:`readiness_sink` to that warm's ``detail`` dict. An implementation MAY
call :func:`report_readiness` from ``ensure_ready`` to say how the warm went
(its mode, admission/setup/restore timings and the like); the facts land on
the span. The protocol is unchanged: a call to ``ensure_ready`` that did not
come through the runtime's warm (a file API preparing the sandbox, say) has
no sink, so its reports go nowhere and it leaves no span.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncContextManager, AsyncIterator, Protocol

from .sandbox_types import Sandbox

#: The ``detail`` dict of the root sandbox warm in progress in this context,
#: or ``None`` when no warm is being timed. Set and reset by the runtime around
#: the warm only; read it through :func:`report_readiness`.
readiness_sink: ContextVar[dict[str, Any] | None] = ContextVar(
    "agent_base_sandbox_readiness_sink", default=None
)


def report_readiness(**detail: Any) -> None:
    """Merge ``detail`` into the warm being timed; a no-op when there is none.

    Meant for ``SandboxCoordinator.ensure_ready`` implementations (and the
    runtime's own uncoordinated warm). Keys are shallow-merged, a later
    report of a key replacing the earlier one. Values should be JSON-shaped
    (str, number, bool, None, lists and dicts of them); the span keeps a
    JSON-safe copy.
    """
    sink = readiness_sink.get()
    if sink is not None:
        sink.update(detail)


class SandboxCoordinator(Protocol):
    async def ensure_ready(self, agent: Any) -> Sandbox:
        """Ready the agent's sandbox and return it.

        May call :func:`report_readiness` to describe the warm; that is
        optional and a no-op outside a runtime-timed warm.
        """
        ...

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
