"""Claude Code as a model client: ``ClaudeCodeProvider`` sends a session's model calls through Claude Code (the
Claude Agent SDK's CLI) and keeps the direct client (``AnthropicProvider``) as its fallback.

    from agent_base.providers.anthropic import AnthropicAgent
    from agent_base.providers.claude_code import ClaudeCodeConfig, ClaudeCodeProvider

    agent = AnthropicAgent(..., provider_value=ClaudeCodeProvider(config=ClaudeCodeConfig(workdir="/var/cc")))

Needs the ``claude-code`` extra (``pip install 'agent-base[claude-code]'``): constructing the provider without it
raises ``ImportError``. Importing this package does not need it.

A claude.ai plan's login may not be used in a third-party product without Anthropic's prior approval (Claude Agent
SDK overview); the host decides which login the CLI runs on.
"""

from .config import (CLI_ENV, DEFAULT_SECRET_PATTERNS, KEEP_ENV, ClaudeCodeConfig, ClaudeCodeConfigError,
                     check_llm_config, cli_env, web_tools_for)
from .fingerprint import message_hash, prefix_fingerprints
from .meter import DEFAULT_METER, LimitState, PlanMeter
from .provider import ClaudeCodeProvider, prune_prompt_files
from .slots import DEFAULT_SLOTS, ProcessSlots
from .transcripts import (MemoryTranscriptLedger, MemoryTranscriptStore, SessionKey, TranscriptLedger,
                          TranscriptStore)
from .url_guard import UrlGuard, make_public_url_guard, public_url_guard

__all__ = [
    "CLI_ENV",
    "ClaudeCodeConfig",
    "ClaudeCodeConfigError",
    "ClaudeCodeProvider",
    "DEFAULT_METER",
    "DEFAULT_SECRET_PATTERNS",
    "DEFAULT_SLOTS",
    "KEEP_ENV",
    "LimitState",
    "MemoryTranscriptLedger",
    "MemoryTranscriptStore",
    "PlanMeter",
    "ProcessSlots",
    "SessionKey",
    "TranscriptLedger",
    "TranscriptStore",
    "UrlGuard",
    "check_llm_config",
    "cli_env",
    "make_public_url_guard",
    "message_hash",
    "prefix_fingerprints",
    "prune_prompt_files",
    "public_url_guard",
    "web_tools_for",
]
