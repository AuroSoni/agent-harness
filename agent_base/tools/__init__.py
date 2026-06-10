"""Tool infrastructure — registry, decorator, envelope pattern, and configurable base."""

from .tool_types import ToolResultEnvelope, GenericErrorEnvelope, GenericTextEnvelope, ToolSchema
from .decorators import tool, ExecutorType
from .base import ConfigurableToolBase
from .bundle import ToolBundle
from .registry import ToolRegistry, ToolCallInfo, ToolCallClassification, RegisteredTool
from .schema_utils import (
    generate_tool_schema,
    TypeHintParsingException,
    DocstringParsingException,
)
from .context import (
    CTX_PARAM_NAME,
    DEFAULT_EMIT_MAX_BYTES,
    DEFAULT_EMIT_MAX_CHARS,
    TOOL_RESULTS_DIR,
    OnceStore,
    ToolContext,
    stable_hash,
)

# NOTE: ``agent_base.tools.media_helpers`` is intentionally NOT imported here —
# it depends on ``agent_base.media_backend.projection`` (Pillow-backed) and
# stays a leaf module so the core tool surface imports clean without media.

__all__ = [
    # Schema
    "ToolSchema",
    # Envelope types
    "ToolResultEnvelope",
    "GenericErrorEnvelope",
    "GenericTextEnvelope",
    # Decorator
    "tool",
    "ExecutorType",
    # Configurable base + bundles
    "ConfigurableToolBase",
    "ToolBundle",
    # Registry
    "ToolRegistry",
    "ToolCallInfo",
    "ToolCallClassification",
    "RegisteredTool",
    # Schema utilities
    "generate_tool_schema",
    "TypeHintParsingException",
    "DocstringParsingException",
    # Tool context (dependency injection)
    "ToolContext",
    "OnceStore",
    "stable_hash",
    "CTX_PARAM_NAME",
    "DEFAULT_EMIT_MAX_CHARS",
    "DEFAULT_EMIT_MAX_BYTES",
    "TOOL_RESULTS_DIR",
]
