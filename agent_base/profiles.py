"""Declarative agent profiles (DESIGN_CONTRACT §6).

A :class:`Profile` is a named, declarative capability bundle:
``Profile{name, tools, frontend_tools, system_prompt, tail}``. The active
profile is persisted in ``AgentConfig`` and auto-restored on resume; it is
switched via ``ctx.switch_profile()`` (O7 — the ONE switch path).

Consumer-specific FE payloads do NOT live here: ``Profile.ui_capabilities``
was removed (2026-06-10 amendment) in favour of the ``on_profile_changed``
observer hook — the library auto-emits a minimal ``ProfileChanged(profile)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Profile:
    """A named, declarative tool/prompt bundle.

    Fields:
        name: The profile's registry key (what ``ctx.switch_profile(name)``
            addresses).
        tools: Backend tool callables (or tool objects) active under this
            profile.
        frontend_tools: Frontend-executed tool callables/specs (relay mode).
        system_prompt: Profile-specific system prompt (``None`` = agent
            default).
        tail: Profile-specific tail instruction appended at render time
            (``None`` = agent default).
    """

    name: str
    tools: list[Callable[..., Any] | Any] = field(default_factory=list)
    frontend_tools: list[Callable[..., Any] | Any] = field(default_factory=list)
    system_prompt: str | None = None
    tail: str | None = None


__all__ = ["Profile"]
