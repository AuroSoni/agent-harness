"""Externalize oversized prompt and tool-result payloads to sandbox files."""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from agent_base.core.messages import Message
from agent_base.core.types import ContentBlock, TextContent, ToolResultBase, ToolResultContent
from agent_base.tools.tool_types import ToolResultEnvelope

if TYPE_CHECKING:
    from agent_base.providers.litellm.token_estimation import LiteLLMTokenEstimator
    from agent_base.sandbox.sandbox_types import Sandbox


CONTEXT_DIR = ".context"


def _strip_binary_data(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_strip_binary_data(item) for item in obj]
    if isinstance(obj, dict):
        source = obj.get("source")
        if (
            isinstance(source, dict)
            and source.get("type") == "base64"
            and "data" in source
        ):
            b64_len = len(source["data"]) if isinstance(source["data"], str) else 0
            byte_size = b64_len * 3 / 4
            if byte_size >= 1024 * 1024:
                size_label = f"{byte_size / (1024 * 1024):.1f} MB"
            else:
                size_label = f"{byte_size / 1024:.1f} KB"
            new_source = {k: v for k, v in source.items() if k != "data"}
            new_source["data"] = f"[base64, {size_label}]"
            return {**obj, "source": new_source}
        return {k: _strip_binary_data(v) for k, v in obj.items()}
    return obj


@dataclass
class ExternalizationConfig:
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
    def __init__(
        self,
        config: ExternalizationConfig,
        sandbox: "Sandbox",
        token_estimator: "LiteLLMTokenEstimator",
    ) -> None:
        self.config = config
        self.sandbox = sandbox
        self.token_estimator = token_estimator

    async def externalize_prompt(self, message: Message) -> Message:
        prompt_copy: Message = copy.deepcopy(message)
        if self.token_estimator.estimate_message(prompt_copy) <= self.config.max_prompt_tokens:
            return prompt_copy

        path = self._prompt_path(prompt_copy.id)
        await self.sandbox.write_file(path, self._render_message_content(prompt_copy.content))
        prompt_copy.content = [TextContent(text=self._prompt_reference_text(path))]
        return prompt_copy

    async def externalize_tool_results(
        self,
        envelopes: list[ToolResultEnvelope],
    ) -> tuple[Message, Message]:
        original_blocks: list[ContentBlock] = []
        for envelope in envelopes:
            original_blocks.append(
                ToolResultContent(
                    tool_name=envelope.tool_name,
                    tool_id=envelope.tool_id,
                    tool_result=envelope.for_context_window(),
                    is_error=envelope.is_error,
                )
            )

        original_message = Message.user(original_blocks)
        context_message = await self._externalize_tool_result_message(original_message)
        return original_message, context_message

    async def externalize_relay_results(
        self,
        completed_results: list[Message],
        relay_results: list[ContentBlock],
    ) -> tuple[Message, Message]:
        all_result_blocks: list[ContentBlock] = []
        for message in completed_results:
            all_result_blocks.extend(copy.deepcopy(message.content))
        all_result_blocks.extend(copy.deepcopy(relay_results))

        original_message = Message.user(all_result_blocks)
        context_message = await self._externalize_tool_result_message(original_message)
        return original_message, context_message

    async def _externalize_tool_result_message(self, message: Message) -> Message:
        context_message: Message = copy.deepcopy(message)
        original_blocks: dict[int, ToolResultBase] = {}
        inline_candidates: list[int] = []

        for index, block in enumerate(context_message.content):
            if not isinstance(block, ToolResultBase):
                continue

            original_block = copy.deepcopy(block)
            original_blocks[index] = original_block
            block_tokens = self._estimate_tool_result_block_tokens(original_block)

            if block_tokens > self.config.max_tool_result_tokens:
                context_message.content[index] = await self._externalize_tool_result_block(
                    original_block
                )
            else:
                inline_candidates.append(index)

        total_tokens = self.token_estimator.estimate_message(context_message)
        if total_tokens > self.config.max_combined_tool_result_tokens:
            for index in inline_candidates:
                context_message.content[index] = await self._externalize_tool_result_block(
                    original_blocks[index]
                )

        return context_message

    def _estimate_tool_result_block_tokens(self, block: ToolResultBase) -> int:
        return self.token_estimator.estimate_message(Message.user([block]))

    async def _externalize_tool_result_block(
        self,
        block: ToolResultBase,
    ) -> ToolResultBase:
        path = self._tool_result_path(block.tool_id)
        await self.sandbox.write_file(path, self._render_tool_result_payload(block))

        return dataclasses.replace(
            block,
            tool_result=[TextContent(text=self._tool_result_reference_text(path))],
        )

    def _render_message_content(self, blocks: list[ContentBlock]) -> str:
        if len(blocks) == 1 and isinstance(blocks[0], TextContent):
            return blocks[0].text
        return self._render_content_blocks(blocks)

    def _render_tool_result_payload(self, block: ToolResultBase) -> str:
        if isinstance(block.tool_result, str):
            return block.tool_result
        if isinstance(block.tool_result, dict):
            return json.dumps(
                _strip_binary_data(block.tool_result),
                ensure_ascii=False,
                indent=2,
            )
        return self._render_content_blocks(block.tool_result)

    def _render_content_blocks(self, blocks: list[ContentBlock]) -> str:
        rendered_parts: list[str] = []
        for block in blocks:
            if isinstance(block, str):
                rendered_parts.append(block)
                continue
            if isinstance(block, TextContent):
                rendered_parts.append(block.text)
                continue
            if isinstance(block, ToolResultBase):
                rendered_parts.append(self._render_tool_result_payload(block))
                continue
            if isinstance(block, dict):
                rendered_parts.append(
                    json.dumps(_strip_binary_data(block), ensure_ascii=False, indent=2)
                )
                continue
            rendered_parts.append(
                json.dumps(
                    _strip_binary_data(block.to_dict() if hasattr(block, "to_dict") else block),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return "\n\n".join(rendered_parts)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        return sanitized or "payload"

    def _prompt_path(self, message_id: str) -> str:
        safe_id = self._sanitize_name(message_id)
        return f"{CONTEXT_DIR}/prompt_{safe_id}.txt"

    def _tool_result_path(self, tool_id: str) -> str:
        safe_tool_id = self._sanitize_name(tool_id)
        return f"{CONTEXT_DIR}/tool_result_{safe_tool_id}.txt"

    @staticmethod
    def _prompt_reference_text(path: str) -> str:
        return f"Prompt content externalized to {path}. Read that file before continuing."

    @staticmethod
    def _tool_result_reference_text(path: str) -> str:
        return f"Tool result content externalized to {path}. Read that file before continuing."
