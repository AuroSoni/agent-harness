"""Externalize oversized prompt and tool-result payloads to sandbox files.

Interface contract: a near-verbatim port of
``agent_base/providers/litellm/context_externalizer.py`` — same config
fields/defaults, same public methods, same ``.context/`` layout and
reference-text convention.  Like litellm (and unlike Anthropic) there is NO
``ServerToolResultContent`` skip-guard: this provider has no server tools,
so the branch simply does not exist.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from agent_base.core.messages import Message
from agent_base.core.types import ContentBlock
from agent_base.tools.tool_types import ToolResultEnvelope

if TYPE_CHECKING:
    from agent_base.sandbox.sandbox_types import Sandbox

    from .token_estimation import AnyLLMTokenEstimator


CONTEXT_DIR = ".context"


@dataclass
class ExternalizationConfig:
    """Per-agent externalization thresholds (persisted)."""

    max_prompt_tokens: int = 80_000
    max_tool_result_tokens: int = 25_000
    max_combined_tool_result_tokens: int = 100_000

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExternalizationConfig":
        if not data:
            return cls()
        valid_fields = {field.name for field in dataclasses.fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_fields}
        return cls(**filtered)


class ContextExternalizer:
    """Swap oversized payloads for sandbox-file references, constructed by
    the shared runtime with this provider's token estimator."""

    def __init__(
        self,
        config: ExternalizationConfig,
        sandbox: "Sandbox",
        token_estimator: "AnyLLMTokenEstimator",
    ) -> None:
        self.config = config
        self.sandbox = sandbox
        self.token_estimator = token_estimator

    async def externalize_prompt(self, message: Message) -> Message:
        """If the (deep-copied) prompt exceeds ``max_prompt_tokens``, write
        its rendered content to ``.context/prompt_<message_id>.txt`` in the
        sandbox and replace the content with a single reference
        ``TextContent``; otherwise return the copy unchanged."""
        raise NotImplementedError

    async def externalize_tool_results(
        self,
        envelopes: list[ToolResultEnvelope],
    ) -> tuple[Message, Message]:
        """Build the full-fidelity user message from ``envelopes`` (one
        ``ToolResultContent`` per envelope, ``for_context_window()``
        payloads), then produce the context-window variant with oversized
        result blocks swapped for ``.context/tool_result_<tool_id>.txt``
        references.  Returns ``(original_message, context_message)``."""
        raise NotImplementedError

    async def externalize_relay_results(
        self,
        completed_results: list[Message],
        relay_results: list[ContentBlock],
    ) -> tuple[Message, Message]:
        """Same as :meth:`externalize_tool_results`, but for a relay resume:
        merge the deep-copied blocks of ``completed_results`` with
        ``relay_results`` into one user message before externalizing.
        Returns ``(original_message, context_message)``."""
        raise NotImplementedError
