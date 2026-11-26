"""``ClaudeCodeConfig``: how the Claude Code provider runs its CLI processes, and what it does with a request's
``AnthropicLLMConfig`` fields the CLI can't take.

The CLI takes fewer request fields than the Messages API. A field it can't honour is refused when it changes what
the agent can do, and logged once per provider when it only tunes the request:

======================================================  ==========================================================
``max_tokens``                                          ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` in the CLI's environment
``effort``                                              ``ClaudeAgentOptions.effort``, when the process starts
``server_tools``: ``web_search``, ``web_fetch``         Claude Code's ``WebSearch`` / ``WebFetch``, held to their
                                                        ``max_uses`` per step
``server_tools``: any other; ``skills``; ``container``  refused (:class:`ClaudeCodeConfigError`)
``thinking_tokens``, ``beta_headers``,                  logged once
``context_management``, ``inference_geo``, ``speed``,
``service_tier``, ``api_kwargs``; ``enable_caching``
off
======================================================  ==========================================================
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .meter import PlanMeter
    from .slots import ProcessSlots
    from .url_guard import UrlGuard

#: Name patterns (regular expressions, case-insensitive) of the environment variables blanked in the CLI's
#: environment: the CLI inherits the host's environment otherwise.
DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL", "DSN", "DATABASE_", "^AWS_", "^E2B_",
)
#: The one secret the CLI needs: the plan's long-lived login (``claude setup-token``).
KEEP_ENV: frozenset[str] = frozenset({"CLAUDE_CODE_OAUTH_TOKEN"})
#: Claude Code's own limits lifted, and its own compaction and memory off, so the library's history and tools
#: are what the model sees.
CLI_ENV: dict[str, str] = {
    "ENABLE_TOOL_SEARCH": "false",                       # every tool schema stays loaded
    "CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH": "200000",  # no truncation of the tools' descriptions
    "MAX_MCP_OUTPUT_TOKENS": "200000",                   # the library's externalizer is the only result cap
    "DISABLE_AUTO_COMPACT": "1",                         # one history: the library's (compaction is its own)
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",              # no Claude Code memory: the host's is the only one
}

#: Claude Code's own web tools, by the Anthropic server tool each stands in for.
WEB_TOOLS: dict[str, str] = {"web_search": "WebSearch", "web_fetch": "WebFetch"}

REFUSED_FIELDS = ("skills", "container_id")
LOGGED_FIELDS = ("thinking_tokens", "beta_headers", "context_management", "inference_geo", "speed",
                 "service_tier", "api_kwargs")


class ClaudeCodeConfigError(ValueError):
    """A request field changes what the agent can do, and Claude Code can't honour it."""


@dataclass(frozen=True)
class ClaudeCodeConfig:
    """How a :class:`~agent_base.providers.claude_code.ClaudeCodeProvider` runs its CLI processes.

    Args:
        workdir: The one folder every Claude Code session runs in. Passed as its full path: on Windows a short
            (8.3) path gives the CLI and the SDK different project keys, and a resume from the store misses.
        server_name: The in-process MCP server the library's tools are listed on. The model sees it in every
            tool name (``mcp__<server_name>__<tool>``). A host whose transcripts were written under another name
            passes that name, so its resumes keep matching.
        fallback: ``"direct"``: a step Claude Code can't carry moves the session to the direct client
            (``AnthropicProvider``, on the host's API key) for good. ``"off"``: it fails instead, so nothing
            ever runs on an API key.
        web_tools: ``None``: Claude Code's ``WebSearch`` / ``WebFetch`` are on when the request's
            ``server_tools`` name ``web_search`` / ``web_fetch``. ``True``: both on. ``False``: both off.
        url_guard: Decides which URLs ``WebFetch`` may open; None is the library's public-address guard.
        env_blank_patterns: Environment variables whose names match are blanked in the CLI's environment.
        max_processes: Claude Code processes the provider's slots allow at once, for the sessions' own runs.
        max_child_processes: Claude Code processes sub-agents may run at once, per level: children draw from a
            count of their own, grandchildren from the next one, so a parent waiting on its children never takes
            their room. Shared by every session's sub-agents at that level.
        slot_wait_s: How long a run's first model call waits for a free slot.
        max_buffer_bytes: The SDK's message buffer (its 1 MB default overflows on a large result or an image).
        max_turns: Claude Code's own turn limit, kept above the library's step limit so the library's holds.
        ledger_wait_s: How long the end of a run waits for its ledger points to be recorded.
        slots, meter: Shared state; None is the process-wide default.
    """

    workdir: Path | str
    server_name: str = "agent_base"
    fallback: Literal["direct", "off"] = "direct"
    web_tools: bool | None = None
    url_guard: "UrlGuard | None" = None
    env_blank_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS
    max_processes: int = 4
    max_child_processes: int = 4
    slot_wait_s: float = 15.0
    max_buffer_bytes: int = 64 << 20
    max_turns: int = 1000
    ledger_wait_s: float = 30.0
    slots: "ProcessSlots | None" = field(default=None, compare=False)
    meter: "PlanMeter | None" = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workdir", Path(self.workdir).resolve())
        if self.fallback not in ("direct", "off"):
            raise ValueError(f"fallback must be 'direct' or 'off', got {self.fallback!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.server_name or ""):
            raise ValueError("server_name must be letters, digits, '_' or '-'")
        if self.max_processes < 1:
            raise ValueError("max_processes must be at least 1")
        if self.max_child_processes < 1:
            raise ValueError("max_child_processes must be at least 1")

    @property
    def tool_prefix(self) -> str:
        return f"mcp__{self.server_name}__"


def cli_env(patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS, *, max_tokens: int | None = None) -> dict[str, str]:
    """The CLI's environment overrides: every variable of this process whose name matches ``patterns`` blanked
    (the CLI inherits the rest), no API key (the plan's login only), and Claude Code's own limits lifted."""
    secret = re.compile("|".join(f"(?:{p})" for p in patterns), re.I) if patterns else None
    blank = {k: "" for k in os.environ if secret is not None and secret.search(k) and k not in KEEP_ENV}
    env = {**blank, "ANTHROPIC_API_KEY": "", **CLI_ENV}
    if max_tokens:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
    return env


def _server_tool_name(tool: dict[str, Any]) -> str:
    name = str(tool.get("name") or "")
    if name:
        return name
    kind = str(tool.get("type") or "")
    return next((w for w in WEB_TOOLS if kind.startswith(w)), kind)


def web_tools_for(llm_config: Any, setting: bool | None) -> dict[str, int | None]:
    """Claude Code's web tools a process gets, each with its per-step ``max_uses`` (None: no limit)."""
    if setting is False:
        return {}
    limits: dict[str, int | None] = {}
    for tool in getattr(llm_config, "server_tools", None) or []:
        name = _server_tool_name(tool)
        if name in WEB_TOOLS:
            uses = tool.get("max_uses")
            limits[WEB_TOOLS[name]] = int(uses) if uses else None
    if setting is True:
        for cc in WEB_TOOLS.values():
            limits.setdefault(cc, None)
    return limits


def check_llm_config(llm_config: Any) -> list[str]:
    """Refuse a field that changes what the agent can do (:class:`ClaudeCodeConfigError`); return the names of
    set fields that only tune the request and that Claude Code ignores."""
    refused = [_server_tool_name(t) for t in getattr(llm_config, "server_tools", None) or []
               if _server_tool_name(t) not in WEB_TOOLS]
    refused += [f for f in REFUSED_FIELDS if getattr(llm_config, f, None)]
    if refused:
        raise ClaudeCodeConfigError(
            "Claude Code can't honour these settings, which change what the agent can do: "
            + ", ".join(refused) + ". Remove them, or run the agent on AnthropicProvider.")
    ignored = [f for f in LOGGED_FIELDS if getattr(llm_config, f, None) not in (None, [], {}, "")]
    if getattr(llm_config, "enable_caching", True) is False:
        ignored.append("enable_caching")
    return ignored


__all__ = [
    "CLI_ENV", "ClaudeCodeConfig", "ClaudeCodeConfigError", "DEFAULT_SECRET_PATTERNS", "KEEP_ENV",
    "WEB_TOOLS", "check_llm_config", "cli_env", "web_tools_for",
]
