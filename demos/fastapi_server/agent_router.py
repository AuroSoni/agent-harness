"""Agent endpoint for streaming responses via SSE."""
import asyncio
import base64
import json
import logging
import mimetypes
import os
import traceback
from typing import Any, AsyncGenerator, Literal, Optional
from urllib.parse import urlparse

import anthropic
import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, model_validator

from agent_base.providers.anthropic import AnthropicAgent, AnthropicLLMConfig
from agent_base.common_tools import (
    ReadFileTool, GlobFileSearchTool, GrepSearchTool,
    ListDirTreeTool, ApplyPatchTool, CodeExecutionTool,
)
from agent_base.tools import tool
from agent_base.core.types import ToolResultContent, TextContent
from storage import config_adapter, conversation_adapter, run_adapter

logger = logging.getLogger(__name__)


########################################################
# FILE UPLOAD CONSTANTS
########################################################

SUPPORTED_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
SUPPORTED_DOCUMENT_TYPES = frozenset({"application/pdf", "text/plain"})
SUPPORTED_MIME_TYPES = SUPPORTED_IMAGE_TYPES | SUPPORTED_DOCUMENT_TYPES

MAX_IMAGE_SIZE = 20 * 1024 * 1024    # 20 MB
MAX_PDF_SIZE = 32 * 1024 * 1024      # 32 MB
MAX_TEXT_SIZE = 5 * 1024 * 1024      # 5 MB


########################################################
# SAMPLE CALCULATOR TOOLS
########################################################

@tool
def add(a: float, b: float) -> str:
    """Add two numbers together and return the sum.

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        String representation of the sum
    """
    return str(a + b)


@tool
def subtract(a: float, b: float) -> str:
    """Subtract the second number from the first number.

    Args:
        a: The number to subtract from (minuend)
        b: The number to subtract (subtrahend)

    Returns:
        String representation of the difference
    """
    return str(a - b)


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together and return the product.

    Args:
        a: The first number to multiply (multiplicand)
        b: The second number to multiply (multiplier)

    Returns:
        String representation of the product
    """
    return str(a * b)


@tool
def divide(a: float, b: float) -> str:
    """Divide the first number by the second number.

    Args:
        a: The dividend (number to be divided)
        b: The divisor (number to divide by)

    Returns:
        String representation of the quotient, or error message for division by zero
    """
    if b == 0:
        return "Error: Division by zero"
    return str(a / b)


SAMPLE_TOOL_FUNCTIONS = [add, subtract, multiply, divide]


########################################################
# FRONTEND TOOLS (executed by browser, schema only on server)
########################################################

@tool(executor="frontend")
def user_confirm(message: str) -> str:
    """Ask the user for yes/no confirmation before proceeding with an action.

    Use this tool when you need explicit user approval before taking an action
    that could have significant consequences, such as:
    - Deleting or modifying data
    - Making purchases or transactions
    - Sending communications
    - Any irreversible operation

    Args:
        message: The confirmation message to display to the user, explaining
                what action requires their approval and any relevant details.

    Returns:
        "yes" if the user confirms, "no" if the user declines
    """
    pass  # Never executed server-side - runs in browser

@tool
def read_image_raw(image_path: str, description: str = "") -> str:
    """Read an image file and return it with a description.

    Args:
        image_path: Path to the image file to read
        description: Optional description to include with the image

    Returns:
        JSON content blocks with image and text
    """
    import base64
    from pathlib import Path

    path = Path(image_path)

    if not path.exists():
        return f"Error: Image not found at {image_path}"

    # Determine media type from extension
    ext = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/png")

    # Read and encode image
    image_bytes = path.read_bytes()
    b64_data = base64.b64encode(image_bytes).decode("utf-8")

    label = description if description else f"Image from {path.name}"
    return json.dumps([
        {"type": "text", "text": label},
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
    ])

# List of all frontend tools for easy registration
FRONTEND_TOOL_FUNCTIONS = [user_confirm]

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentConfig(BaseModel):
    """Configuration for an agent instance.

    Fields map to AnthropicAgent constructor parameters.
    Provider-specific LLM settings (thinking_tokens, max_tokens, server_tools,
    skills, beta_headers) are bundled into AnthropicLLMConfig at construction time.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_prompt: str
    model: str
    thinking_tokens: int = 0
    max_tokens: int = 64000
    tools: list = []
    frontend_tools: list = []  # Frontend-executed tools (schema only on server)
    server_tools: list = []
    skills: list = []
    context_management: dict | None = None
    beta_headers: list[str] = []
    config_adapter: Any = None
    conversation_adapter: Any = None
    run_adapter: Any = None
    subagents: dict | None = None  # dict[str, AnthropicAgent] for subagent delegation
    stream_meta_history_and_tool_results: bool = False


# Agent type literal for request validation
AgentType = Literal[
    "agent_no_tools",
    "agent_client_tools",
    "agent_frontend_tools",
    "agent_all_json",
    "agent_skills",
    "agent_subagents",
]


class AgentRunRequest(BaseModel):
    """Request to run an agent with a user prompt."""
    agent_uuid: Optional[str] = None
    agent_type: Optional[AgentType] = None
    user_prompt: str | list[dict] | dict

    @model_validator(mode='after')
    def validate_agent_source(self) -> "AgentRunRequest":
        # Allow both agent_uuid and agent_type (agent_type provides config)
        # Default to agent_frontend_tools if neither provided
        if not self.agent_uuid and not self.agent_type:
            self.agent_type = "agent_frontend_tools"
        return self


class FileMetadata(BaseModel):
    """Metadata for an uploaded file from Anthropic Files API."""
    id: str
    filename: str
    mime_type: str
    size_bytes: int
    created_at: str
    downloadable: bool


class UploadResponse(BaseModel):
    """Response containing metadata for all uploaded files."""
    files: list[FileMetadata]


class FrontendToolResult(BaseModel):
    """A single frontend tool result."""
    tool_use_id: str
    content: str
    is_error: bool = False


class ToolResultsRequest(BaseModel):
    """Request to submit frontend tool results and resume agent execution."""
    agent_uuid: str
    tool_results: list[FrontendToolResult]
    agent_type: Optional[AgentType] = None  # Original agent type for correct config (defaults to agent_frontend_tools)


class ConversationItem(BaseModel):
    """A single conversation turn in the history."""
    conversation_id: str
    run_id: str
    sequence_number: int
    user_message: str
    final_response: str | None
    started_at: str | None
    completed_at: str | None
    stop_reason: str | None
    total_steps: int | None
    usage: dict | None
    generated_files: list[dict] | None
    messages: list[dict]


class ConversationListResponse(BaseModel):
    """Response for paginated conversation history."""
    conversations: list[ConversationItem]
    has_more: bool
    title: str | None = None  # Session title if available


class AgentSessionItem(BaseModel):
    """A single agent session in the list."""
    agent_uuid: str
    title: str | None
    created_at: str | None
    updated_at: str | None
    total_runs: int


class AgentSessionListResponse(BaseModel):
    """Response for paginated agent sessions list."""
    sessions: list[AgentSessionItem]
    total: int


########################################################
# HELPER: Build AnthropicAgent from AgentConfig
########################################################

def _create_agent(cfg: AgentConfig, agent_uuid: str | None = None) -> AnthropicAgent:
    """Construct an AnthropicAgent from an AgentConfig pydantic model."""
    llm_config = AnthropicLLMConfig(
        thinking_tokens=cfg.thinking_tokens or None,
        max_tokens=cfg.max_tokens or None,
        server_tools=cfg.server_tools or None,
        skills=cfg.skills or None,
        beta_headers=cfg.beta_headers or None,
    )

    return AnthropicAgent(
        system_prompt=cfg.system_prompt,
        model=cfg.model,
        config=llm_config,
        tools=cfg.tools or None,
        frontend_tools=cfg.frontend_tools or None,
        subagents=cfg.subagents or None,
        stream_meta_history_and_tool_results=cfg.stream_meta_history_and_tool_results,
        config_adapter=cfg.config_adapter,
        conversation_adapter=cfg.conversation_adapter,
        run_adapter=cfg.run_adapter,
        agent_uuid=agent_uuid,
    )


########################################################
# AGENT CONFIGURATIONS
########################################################

# Shared config
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "./workspace")

# Common tools (sandbox-based I/O — sandbox is attached by AnthropicAgent.initialize())
COMMON_TOOL_FUNCTIONS = [
    ReadFileTool().get_tool(),
    GlobFileSearchTool().get_tool(),
    GrepSearchTool().get_tool(),
    ListDirTreeTool().get_tool(),
    ApplyPatchTool().get_tool(),
    CodeExecutionTool().get_tool(),
]

# --- 1. Plain chat (no tools) ---
agent_no_tools = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    stream_meta_history_and_tool_results=True,
)

# --- 2. Calculator tools only ---
agent_client_tools = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=SAMPLE_TOOL_FUNCTIONS,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    stream_meta_history_and_tool_results=True,
)

# --- 3. Frontend tools (browser-executed, pause/resume) ---
agent_frontend_tools = AgentConfig(
    system_prompt="""You are a helpful assistant that should help the user with their questions.
When performing calculations that result in significant values (over 50), or when taking
any action that could have consequences, ask for user confirmation using the user_confirm tool.""",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=SAMPLE_TOOL_FUNCTIONS + [read_image_raw],
    frontend_tools=FRONTEND_TOOL_FUNCTIONS,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    stream_meta_history_and_tool_results=True,
)

# --- 4. All common tools + server tools ---
agent_all_json = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=COMMON_TOOL_FUNCTIONS + [read_image_raw],
    server_tools=[
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 50},
        {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 50},
    ],
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
                "clear_at_least": {"type": "input_tokens", "value": 5000},
                "exclude_tools": ["web_search"],
            }
        ]
    },
    beta_headers=[
        "web-fetch-2025-09-10",
        "files-api-2025-04-14",
        "context-management-2025-06-27",
    ],
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    stream_meta_history_and_tool_results=True,
)

# --- 5. Subagent orchestration ---
_calculator_agent = AnthropicAgent(
    system_prompt="You are a calculator specialist. Use the provided math tools to perform calculations accurately.",
    description="Performs arithmetic calculations using add, subtract, multiply, and divide tools",
    model="claude-sonnet-4-5",
    tools=SAMPLE_TOOL_FUNCTIONS,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
)

_researcher_agent = AnthropicAgent(
    system_prompt="You are a research specialist. Search the web for information and provide comprehensive answers.",
    description="Searches the web for information and provides summarised results",
    model="claude-sonnet-4-5",
    config=AnthropicLLMConfig(
        server_tools=[
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 50},
            {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 50},
        ],
        beta_headers=["web-fetch-2025-09-10"],
    ),
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
)

agent_subagents = AgentConfig(
    system_prompt=(
        "You are a coordinator agent. You MUST delegate tasks to your specialist subagents:\n"
        "- Use the 'calculator' subagent for any arithmetic or math questions.\n"
        "- Use the 'researcher' subagent for any questions requiring web search or factual lookup.\n"
        "After receiving results from subagents, synthesise and present them to the user."
    ),
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    subagents={
        "calculator": _calculator_agent,
        "researcher": _researcher_agent,
    },
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    stream_meta_history_and_tool_results=True,
)

# --- 6. Anthropic Agent Skills (code execution + document generation) ---
agent_skills = AgentConfig(
    system_prompt="You are a helpful assistant that can generate documents and run code using Anthropic Skills.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    server_tools=[
        {"type": "code_execution_20250825", "name": "code_execution"},
    ],
    skills=[
        {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
        {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
        {"type": "anthropic", "skill_id": "docx", "version": "latest"},
        {"type": "anthropic", "skill_id": "pdf", "version": "latest"},
    ],
    beta_headers=[
        "code-execution-2025-08-25",
        "skills-2025-10-02",
        "files-api-2025-04-14",
    ],
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    stream_meta_history_and_tool_results=True,
)

# Agent config registry
AGENT_CONFIGS: dict[AgentType, AgentConfig] = {
    "agent_no_tools": agent_no_tools,
    "agent_client_tools": agent_client_tools,
    "agent_frontend_tools": agent_frontend_tools,
    "agent_all_json": agent_all_json,
    "agent_skills": agent_skills,
    "agent_subagents": agent_subagents,
}

async def stream_agent_response(
    user_prompt: str | list[dict] | dict,
    agent_uuid: Optional[str] = None,
    agent_type: Optional[AgentType] = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted stream of agent responses.

    Args:
        user_prompt: The user's question or task (str, list, or dict)
        agent_uuid: Optional UUID to resume existing agent
        agent_type: Optional agent type to use for configuration

    Yields:
        SSE-formatted strings containing raw agent output chunks
    """
    try:
        # Get config from registry (default to agent_all_json)
        config = AGENT_CONFIGS[agent_type or "agent_all_json"]

        # Create the agent with config
        agent = _create_agent(config, agent_uuid=agent_uuid)

        # Create queue for streaming
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Run the agent in background (this will populate the queue)
        async def run_agent_and_signal():
            try:
                result = await agent.run_stream(user_prompt, queue)
                return result
            finally:
                await queue.put(None)  # Always signal completion

        agent_task = asyncio.create_task(run_agent_and_signal())

        # Yield raw chunks as they arrive from the queue
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
                queue.task_done()
        except asyncio.CancelledError:
            agent_task.cancel()
            raise

        # Wait for agent to complete (re-raises if agent.run() failed)
        await agent_task

        # Send final SSE marker to close stream
        yield "data: [DONE]\n\n"

    except Exception as e:
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        logger.error(f"Error in agent stream: {e}\n{''.join(tb)}")
        error_detail = {
            "error": type(e).__name__,
            "message": str(e),
            "traceback": "".join(tb),
        }
        yield f"data: {json.dumps(error_detail)}\n\n"


async def stream_tool_results_response(
    request: ToolResultsRequest,
) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted stream after frontend tools complete.

    Re-hydrates the agent from the database using agent_uuid, submits
    the frontend tool results, and continues streaming the response.

    Args:
        request: Tool results request containing agent_uuid and results

    Yields:
        SSE-formatted strings containing agent output chunks
    """
    try:
        # Re-hydrate agent from DB (state is loaded automatically via agent_uuid)
        # Use the specified agent_type config, defaulting to agent_frontend_tools
        agent_type = request.agent_type or "agent_frontend_tools"
        config = AGENT_CONFIGS.get(agent_type, AGENT_CONFIGS["agent_frontend_tools"])

        agent = _create_agent(config, agent_uuid=request.agent_uuid)

        # Create queue for streaming
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Build relay result content blocks from frontend tool results
        relay_blocks = []
        for r in request.tool_results:
            relay_blocks.append(ToolResultContent(
                tool_name="",  # Name not required for results
                tool_id=r.tool_use_id,
                tool_result=[TextContent(text=r.content)],
                is_error=r.is_error,
            ))

        # Continue the agent with frontend tool results
        async def run_continuation():
            try:
                result = await agent.resume_with_relay_results(
                    relay_results=relay_blocks,
                    queue=queue,
                )
                return result
            finally:
                await queue.put(None)  # Always signal completion

        agent_task = asyncio.create_task(run_continuation())

        # Yield chunks as they arrive
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
                queue.task_done()
        except asyncio.CancelledError:
            agent_task.cancel()
            raise

        # Wait for agent to complete (re-raises if continuation failed)
        await agent_task

        # Send final SSE marker
        yield "data: [DONE]\n\n"

    except Exception as e:
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        logger.error(f"Error in tool results stream: {e}\n{''.join(tb)}")
        error_detail = {
            "error": type(e).__name__,
            "message": str(e),
            "traceback": "".join(tb),
        }
        yield f"data: {json.dumps(error_detail)}\n\n"


async def build_content_blocks_from_files(
    files: list[UploadFile],
    user_prompt: str,
) -> list[dict]:
    """Convert uploaded files + text prompt into Anthropic API content blocks.

    Reads each file, validates MIME type and size, then constructs the
    appropriate content block (image or document) with base64 encoding.
    Text files use source.type="text" (raw text, no base64).
    File blocks come first; the user's text prompt is appended last.

    Args:
        files: List of FastAPI UploadFile objects.
        user_prompt: The user's text message.

    Returns:
        List of content block dicts ready for Anthropic API.

    Raises:
        HTTPException: On unsupported MIME type or file too large.
    """
    content_blocks: list[dict] = []

    for upload_file in files:
        file_bytes = await upload_file.read()
        file_size = len(file_bytes)

        # Determine MIME type (fall back to guessing from filename)
        mime_type = upload_file.content_type
        if not mime_type or mime_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(upload_file.filename or "")
            mime_type = guessed or "application/octet-stream"

        # Validate MIME type
        if mime_type not in SUPPORTED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{mime_type}' for file "
                    f"'{upload_file.filename}'. Supported types: "
                    f"{', '.join(sorted(SUPPORTED_MIME_TYPES))}"
                ),
            )

        # Validate file size
        size_limits = {
            "application/pdf": ("PDF", MAX_PDF_SIZE),
            "text/plain": ("Text file", MAX_TEXT_SIZE),
        }
        if mime_type in SUPPORTED_IMAGE_TYPES:
            limit_label, limit = "Image", MAX_IMAGE_SIZE
        else:
            limit_label, limit = size_limits[mime_type]

        if file_size > limit:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{limit_label} '{upload_file.filename}' is "
                    f"{file_size / (1024 * 1024):.1f} MB, exceeding the "
                    f"{limit / (1024 * 1024):.0f} MB limit."
                ),
            )

        # Build content block
        if mime_type in SUPPORTED_IMAGE_TYPES:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64.standard_b64encode(file_bytes).decode("utf-8"),
                },
            })
        elif mime_type == "application/pdf":
            content_blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.standard_b64encode(file_bytes).decode("utf-8"),
                },
            })
        elif mime_type == "text/plain":
            try:
                text_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text_content = file_bytes.decode("utf-8", errors="replace")
            content_blocks.append({
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": text_content,
                },
            })

    # User's text prompt is always the final block
    content_blocks.append({"type": "text", "text": user_prompt})
    return content_blocks


@router.post("/run")
async def run_agent(request: AgentRunRequest) -> StreamingResponse:
    """Run an agent with the given prompt and stream responses via SSE.

    This endpoint creates a new agent (or resumes an existing one) and streams
    its responses in real-time using Server-Sent Events (SSE).

    Only one of agent_type or agent_uuid can be provided. If neither is provided,
    defaults to agent_all_json.

    Args:
        request: Agent run request containing user_prompt and optional agent_uuid/agent_type

    Returns:
        StreamingResponse with text/event-stream content type

    Example:
        ```
        POST /agent/run
        {
            "user_prompt": "Calculate (15 + 27) * 3 - 8",
            "agent_type": "agent_all_json"
        }
        ```
    """
    return StreamingResponse(
        stream_agent_response(
            user_prompt=request.user_prompt,
            agent_uuid=request.agent_uuid,
            agent_type=request.agent_type,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


@router.post("/run/multipart")
async def run_agent_multipart(
    user_prompt: str = Form(...),
    agent_uuid: Optional[str] = Form(default=None),
    agent_type: Optional[AgentType] = Form(default=None),
    files: list[UploadFile] = File(default=[]),
) -> StreamingResponse:
    """Run an agent with text prompt and file attachments via multipart/form-data.

    Files are base64-encoded server-side and sent as Anthropic content blocks.
    Supported file types:
    - Images: image/png, image/jpeg, image/gif, image/webp (max 20 MB each)
    - PDF documents: application/pdf (max 32 MB each)
    - Text documents: text/plain (max 5 MB each)

    The text prompt is always included as the final content block.
    If no files are attached, only the text prompt is sent.

    Args:
        user_prompt: The user's text message (required)
        agent_uuid: Optional UUID to resume an existing agent session
        agent_type: Optional agent type for configuration
        files: Optional list of file attachments

    Returns:
        StreamingResponse with text/event-stream content type

    Example:
        ```
        POST /agent/run/multipart
        Content-Type: multipart/form-data

        user_prompt: "What's in this image?"
        agent_type: "agent_all_json"
        files: <image.png>
        ```
    """
    # Apply default agent_type (same logic as AgentRunRequest.validate_agent_source)
    if not agent_uuid and not agent_type:
        agent_type = "agent_frontend_tools"

    # Build content blocks from files + prompt
    if files:
        prompt: str | list[dict] = await build_content_blocks_from_files(files, user_prompt)
    else:
        prompt = user_prompt

    return StreamingResponse(
        stream_agent_response(
            user_prompt=prompt,
            agent_uuid=agent_uuid,
            agent_type=agent_type,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tool_results")
async def submit_tool_results(request: ToolResultsRequest) -> StreamingResponse:
    """Resume agent execution after frontend tools have been executed.

    This endpoint is called by the browser after it executes frontend tools
    (e.g., user_confirm). The agent state is re-hydrated from the database
    using the agent_uuid, and execution continues with the tool results.

    Args:
        request: Tool results request containing agent_uuid and tool results

    Returns:
        StreamingResponse with text/event-stream content type

    Example:
        ```
        POST /agent/tool_results
        {
            "agent_uuid": "abc123...",
            "tool_results": [
                {"tool_use_id": "tool_001", "content": "yes"},
                {"tool_use_id": "tool_002", "content": "confirmed"}
            ]
        }
        ```
    """
    return StreamingResponse(
        stream_tool_results_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


@router.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(default=[]),
    urls: list[str] = Form(default=[]),
) -> UploadResponse:
    """Upload files to Anthropic Files API.

    Accepts files via form data (direct uploads or URLs). For URLs, the server
    downloads the file first and then uploads to Anthropic.

    Args:
        files: List of files to upload directly
        urls: List of URLs to download and upload

    Returns:
        UploadResponse containing metadata for all uploaded files

    Example:
        ```
        POST /agent/upload
        Content-Type: multipart/form-data

        files: <file1>, <file2>
        urls: https://example.com/image.png
        ```
    """
    if not files and not urls:
        raise HTTPException(
            status_code=400,
            detail="At least one file or URL must be provided",
        )

    client = anthropic.Anthropic()
    uploaded_files: list[FileMetadata] = []

    # Upload direct files
    for upload_file in files:
        if upload_file.filename:
            try:
                content = await upload_file.read()
                mime_type = upload_file.content_type or "application/octet-stream"

                result = client.beta.files.upload(
                    file=(upload_file.filename, content, mime_type),
                )

                uploaded_files.append(FileMetadata(
                    id=result.id,
                    filename=result.filename,
                    mime_type=result.mime_type,
                    size_bytes=result.size_bytes,
                    created_at=result.created_at.isoformat(),
                    downloadable=result.downloadable,
                ))
            except Exception as e:
                logger.error(f"Failed to upload file {upload_file.filename}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload file {upload_file.filename}: {str(e)}",
                )

    # Download and upload from URLs
    async with httpx.AsyncClient() as http_client:
        for url in urls:
            try:
                # Download file from URL
                response = await http_client.get(url, follow_redirects=True)
                response.raise_for_status()

                # Extract filename from URL or Content-Disposition header
                content_disposition = response.headers.get("content-disposition")
                if content_disposition and "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[-1].strip('"\'')
                else:
                    parsed_url = urlparse(url)
                    filename = os.path.basename(parsed_url.path) or "downloaded_file"

                # Determine MIME type
                content_type = response.headers.get("content-type", "").split(";")[0]
                if not content_type:
                    content_type, _ = mimetypes.guess_type(filename)
                    content_type = content_type or "application/octet-stream"

                content = response.content

                result = client.beta.files.upload(
                    file=(filename, content, content_type),
                )

                uploaded_files.append(FileMetadata(
                    id=result.id,
                    filename=result.filename,
                    mime_type=result.mime_type,
                    size_bytes=result.size_bytes,
                    created_at=result.created_at.isoformat(),
                    downloadable=result.downloadable,
                ))
            except httpx.HTTPError as e:
                logger.error(f"Failed to download from URL {url}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download from URL {url}: {str(e)}",
                )
            except Exception as e:
                logger.error(f"Failed to upload file from URL {url}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload file from URL {url}: {str(e)}",
                )

    return UploadResponse(files=uploaded_files)


@router.get("/{agent_uuid}/images/{image_id}")
async def get_tool_image(
    agent_uuid: str,
    image_id: str,
) -> StreamingResponse:
    """Fetch an image generated by a tool result.

    This endpoint serves images stored by the local media backend.

    Args:
        agent_uuid: The agent session UUID
        image_id: The image identifier (e.g., "img_abc123")

    Returns:
        StreamingResponse with the image content

    Raises:
        HTTPException 404: If image not found
    """
    from pathlib import Path

    # Get file backend configuration
    # For now, use default local path - in production this should match agent config
    base_path = Path("./agent-files")
    agent_dir = base_path / agent_uuid

    if not agent_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Agent directory not found: {agent_uuid}",
        )

    # Find image file matching the image_id pattern
    # Files are stored as: {image_id}_{filename}.{ext}
    matching_files = list(agent_dir.glob(f"{image_id}_*"))

    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {image_id}",
        )

    image_path = matching_files[0]

    # Determine media type from extension
    ext = image_path.suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")

    # Read and return image
    try:
        content = image_path.read_bytes()
        return StreamingResponse(
            iter([content]),
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{image_path.name}"',
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
            },
        )
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read image: {str(e)}",
        )


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(default=50, le=100, description="Maximum number of sessions to return"),
    offset: int = Query(default=0, ge=0, description="Number of sessions to skip"),
) -> AgentSessionListResponse:
    """List all agent sessions with metadata.

    Returns sessions sorted by updated_at descending (newest first).
    Supports offset-based pagination.

    Args:
        limit: Maximum number of sessions to return (max 100)
        offset: Number of sessions to skip

    Returns:
        AgentSessionListResponse with list of sessions and total count
    """
    # Use shared adapter directly
    sessions, total = await config_adapter.list_sessions(limit=limit, offset=offset)

    return AgentSessionListResponse(
        sessions=[
            AgentSessionItem(
                agent_uuid=s["agent_uuid"],
                title=s.get("title"),
                created_at=s.get("created_at"),
                updated_at=s.get("updated_at"),
                total_runs=s.get("total_runs", 0),
            )
            for s in sessions
        ],
        total=total,
    )


@router.get("/{agent_uuid}/conversations")
async def get_conversations(
    agent_uuid: str,
    before: int | None = Query(default=None, description="Load conversations with sequence_number < before"),
    limit: int = Query(default=20, le=100, description="Maximum conversations to return"),
) -> ConversationListResponse:
    """Get paginated conversation history for an agent (newest first).

    Uses cursor-based pagination for efficient infinite scroll. On initial load,
    omit the `before` parameter to get the newest conversations. For subsequent
    pages, pass the smallest `sequence_number` from the previous response as `before`.

    Args:
        agent_uuid: The agent's UUID
        before: Load conversations with sequence_number < this value (None = latest)
        limit: Maximum number of conversations to return (max 100)

    Returns:
        ConversationListResponse with conversations and has_more flag

    Example:
        ```
        # Initial load (newest conversations)
        GET /agent/{uuid}/conversations?limit=20

        # Load older conversations (scroll up)
        GET /agent/{uuid}/conversations?before=31&limit=20
        ```
    """
    # Use conversation adapter - returns list of Conversation dataclasses
    conversations, has_more = await conversation_adapter.load_cursor(
        agent_uuid=agent_uuid,
        before=before,
        limit=limit,
    )

    # Convert Conversation dataclasses to response models
    items = [
        ConversationItem(
            conversation_id=conv.conversation_id,
            run_id=conv.run_id,
            sequence_number=conv.sequence_number or 0,
            user_message=conv.user_message,
            final_response=conv.final_response,
            started_at=conv.started_at,
            completed_at=conv.completed_at,
            stop_reason=conv.stop_reason,
            total_steps=conv.total_steps,
            usage=conv.usage,
            generated_files=conv.generated_files,
            messages=conv.messages,
        )
        for conv in conversations
    ]

    # Get session title from config adapter
    agent_config = await config_adapter.load(agent_uuid)
    session_title = agent_config.title if agent_config else None

    return ConversationListResponse(
        conversations=items,
        has_more=has_more,
        title=session_title,
    )
