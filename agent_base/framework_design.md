# agent_base — Framework Design

> Provider-agnostic, async-first framework for building production AI agents.
> Refactored from `anthropic_agent` to decouple the core orchestration loop from any single LLM provider.

---

## 1. Design Principles

1. **Separate identity from location from representation.** A media file's ID is stable; where its bytes live is a backend concern; how those bytes are projected (base64, URL, reference) depends on the consumer.

2. **Tools never touch the OS directly.** All file I/O and command execution flows through a `Sandbox` interface. Isolation ships incrementally — swap `LocalSandbox` for `DockerSandbox` by changing config, not code.

3. **The tool owns its result shape.** Each tool defines a `ToolResultEnvelope` subclass with two projections: one for the LLM context window (minimal), one for the conversation log / UI (rich). The agent loop is polymorphic over envelope types.

4. **Provider-agnostic core, provider-specific edges.** The canonical model (`ContentBlock`, `Message`, `AgentConfig`) lives in `core/`. Provider-specific translation (wire format, API calls, streaming parsing) lives in `providers/<name>/`.

5. **Composition over inheritance.** The agent is assembled from independently swappable components: `Provider`, `Sandbox`, `MediaBackend`, `Compactor`, `MemoryStore`, storage adapters. All wired at construction time.

6. **Lazy resolution at the boundary.** Content blocks store abstract references (`media_id`, `source_type=FILE_ID`) internally. Resolution to base64/URL/bytes happens at the consumer boundary (LLM call, frontend response, persistence), never at creation time.

---

## 2. Directory Layout

```
agent_base/
├── core/                          # Provider-agnostic domain model & contracts
│   ├── types.py                   # ContentBlock hierarchy, enums, serialization
│   ├── messages.py                # Message, Usage, MessageFormatter ABC
│   ├── config.py                  # AgentConfig, Conversation dataclasses
│   ├── agent.py                   # Agent ABC (run, resume_run_with_tool_results)
│   ├── provider.py                # Provider ABC (generate, generate_stream)
│   ├── result.py                  # AgentResult dataclass
│   └── streaming.py               # StreamDelta types for framework-level SSE
│
├── tools/                         # Tool infrastructure (registry, decorator, envelope)
│   ├── types.py                   # ToolResultEnvelope ABC, GenericErrorEnvelope
│   ├── registry.py                # ToolRegistry (register, execute, schema export)
│   ├── decorators.py              # @tool decorator, schema generation
│   └── base.py                    # ConfigurableToolBase ABC (templated docstrings)
│
├── common_tools/                  # Built-in tool implementations
│   ├── code_execution.py          # CodeExecutionTool + CodeExecEnvelope
│   ├── file_read.py               # ReadFile tool
│   ├── file_write.py              # WriteFile / ApplyPatch tools
│   ├── glob_search.py             # GlobFilesSearch tool
│   ├── grep_search.py             # GrepSearch tool
│   ├── list_dir.py                # ListDir tool
│   ├── subagent.py                # SubAgentTool + SubAgentEnvelope
│   └── ...
│
├── sandbox/                       # Sandbox infrastructure (isolation layer)
│   ├── types.py                   # Sandbox ABC, ExecResult dataclass
│   ├── local.py                   # LocalSandbox (path restriction, ~1ms setup)
│   ├── docker.py                  # DockerSandbox (container isolation, 2-5s setup)
│   └── e2b.py                     # E2BSandbox (cloud VM isolation, 5-15s setup)
│
├── media_backend/                 # Media storage + resolution (combined)
│   ├── types.py                   # MediaMetadata dataclass, MediaBackend ABC
│   ├── local.py                   # LocalMediaBackend (filesystem)
│   ├── s3.py                      # S3MediaBackend (AWS S3)
│   ├── memory.py                  # MemoryMediaBackend (in-process, for tests)
│   └── cache.py                   # LocalFileCache (read-through + write-back)
│
├── compaction/                    # Context window management
│   ├── base.py                    # Compactor ABC
│   ├── sliding_window.py          # SlidingWindowCompactor (default)
│   ├── summarizing.py             # SummarizingCompactor (LLM-based)
│   ├── tool_result_removal.py     # ToolResultRemovalCompactor
│   └── noop.py                    # NoOpCompactor
│
├── memory/                        # Cross-session memory stores
│   ├── base.py                    # MemoryStore ABC (retrieve, update)
│   └── noop.py                    # NoOpMemoryStore
│
├── storage/                       # Persistence adapters (three-adapter pattern)
│   ├── base.py                    # ABCs: AgentConfigAdapter, ConversationAdapter, AgentRunAdapter
│   ├── adapters/
│   │   ├── memory.py              # In-process dict storage (default)
│   │   ├── filesystem.py          # JSON files on disk
│   │   └── postgres.py            # PostgreSQL backend
│   └── factory.py                 # create_adapters(backend_type, **kwargs)
│
├── logging/                       # Structured logging (structlog)
│
├── providers/                     # Concrete provider implementations
│   ├── anthropic/
│   │   ├── agent.py               # AnthropicAgent (Agent subclass)
│   │   ├── provider.py            # AnthropicProvider (Provider subclass)
│   │   ├── formatter.py           # AnthropicMessageFormatter (MessageFormatter subclass)
│   │   └── config.py              # AnthropicLLMConfig dataclass
│   ├── openai/                    # (future)
│   └── litellm/                   # (future)
│
└── developer-diaries/             # Design discussion records (not shipped)
```

---

## 3. Core Module (`core/`)

The core module defines the **canonical domain model**. Everything here is provider-agnostic. No provider SDK imports. No provider-specific logic.

### 3.1 Content Block Hierarchy (`core/types.py`)

The polymorphic content model — "best of all" philosophy: neither the LCD of all LLMs nor the aggregate. It represents the richest semantically meaningful content types that the framework understands, with `kwargs` as the escape hatch for provider extensions.

```
ContentBlock (ABC, dataclass)
├── TextContent                          # text
├── ThinkingContent                      # thinking, signature
├── MediaContent                         # media_type, media_id (required)
│   ├── ImageContent                     # source_type, data (transient), filename
│   ├── DocumentContent                  # source_type, data (transient), filename
│   └── AttachmentContent                # filename, source_type, data (transient)
├── ToolUseBase                          # tool_name, tool_id (required), tool_input
│   ├── ToolUseContent                   # client-side tool invocation
│   ├── ServerToolUseContent             # server-side (provider-managed) tool
│   └── MCPToolUseContent                # MCP protocol tool (+ mcp_server_name)
├── ToolResultBase                       # tool_name, tool_id (required), tool_result, is_error
│   ├── ToolResultContent                # client-side tool result
│   ├── ServerToolResultContent          # server-side tool result
│   └── MCPToolResultContent             # MCP tool result (+ mcp_server_name)
├── CitationBase                         # cited_text
│   ├── CharCitation                     # document_index, start/end_char_index
│   ├── PageCitation                     # document_index, start/end_page_number
│   ├── ContentBlockCitation             # document_index, start/end_block_index
│   ├── SearchResultCitation             # search_result_index, source, start/end_block_index
│   └── WebSearchResultCitation          # url, title
└── ErrorContent                         # error_message, error_type, error_code
```

**Serialization contract:**
- `to_dict()` emits JSON-native primitives only. Enums emit `.value` strings.
- `raw` field is transient — **never** serialized. In-memory reference to original provider object.
- `kwargs` field is **always** persisted. Provider-specific extensions (user must ensure JSON-safe values).
- `data` on media blocks is transient. The actual bytes/base64 are resolved on demand via `media_id` from the media backend, not stored inline.
- `from_dict()` is a match-based polymorphic dispatcher. Restores the correct subclass from serialized dicts.

**Key invariants:**
- `MediaContent` subclasses require a non-empty `media_id` (enforced in `__post_init__`).
- `ToolUseBase` and `ToolResultBase` subclasses require a non-empty `tool_id` for correlation.
- `ToolResultBase.tool_result` supports both `str` and `List[ContentBlock]` for nested content round-tripping.

### 3.2 Message Layer (`core/messages.py`)

**`Usage`** — Token counts for a single API call. Purely numeric. Source identity (provider, model) and billing context live on `Message`, not here.

**`Message`** — A single message in a conversation. Fields:
- `id` (uuid4), `role` (Role enum), `content` (list of ContentBlocks)
- `stop_reason`, `usage`, `provider`, `model`
- `usage_kwargs` — provider-specific billing metadata (e.g. `service_tier`)
- Convenience constructors: `Message.system(text)`, `Message.user(...)`, `Message.assistant(...)`
- Full `to_dict()` / `from_dict()` round-tripping.

**`MessageFormatter` (ABC)** — Translates between canonical `Message` objects and provider wire format:
- `format_messages(messages, params) -> dict` — canonical messages to provider wire format (LLM input)
- `parse_response(raw_response) -> Message` — provider raw response to canonical Message

Each provider implements its own `MessageFormatter`. This is where provider-specific content type mappings live (e.g., Anthropic's `source.type = "base64"` for images, OpenAI's `image_url` format).

### 3.3 Agent Configuration (`core/config.py`)

**`AgentConfig`** — The persistent agent session state, resumable across runs:
- Identity: `agent_uuid`, `description`, `provider`, `model`
- State: `messages` (compacted context), `conversation_history` (unabridged per-run), `current_step`
- Tools: `tool_schemas`, `tool_names`, `server_tools`, `subagent_schemas`
- Provider config: `llm_config: dict[str, Any]` (provider-specific params serialized)
- Components: `formatter`, `compactor_type`, `memory_store_type`
- Media: `file_registry` (to be migrated to media_backend reference)
- Frontend relay: `pending_frontend_tools`, `pending_backend_results`, `awaiting_frontend_tools`
- Hierarchy: `parent_agent_uuid`
- Token tracking: `last_known_input_tokens`, `last_known_output_tokens`
- Extension: `extras: dict[str, Any]`

**`Conversation`** — A single run record for UI display and pagination:
- Identity: `agent_uuid`, `run_id`, `sequence_number`
- Content: `user_message`, `final_response`, `messages` (full conversation for this run)
- Outcome: `stop_reason`, `total_steps`, `usage`, `cost`
- Files: `generated_files`
- Extension: `extras: dict[str, Any]`

### 3.4 Agent ABC (`core/agent.py`)

Minimal abstract base defining the agent lifecycle:

```python
class Agent(ABC):
    @abstractmethod
    async def run(self, prompt: str | Message) -> AgentResult: ...

    @abstractmethod
    async def resume_run_with_tool_results(self, results: ...) -> AgentResult: ...
```

All orchestration logic (the agentic loop, tool dispatch, compaction, streaming) lives in concrete subclasses under `providers/`. The ABC only defines the external contract.

### 3.5 Provider ABC (`core/provider.py`)

**Status:** To be defined.

The `Provider` interface abstracts the raw LLM API call. Each provider (Anthropic, OpenAI, LiteLLM) implements this to handle authentication, request formation, retry/backoff, and response parsing.

```python
class Provider(ABC):
    @abstractmethod
    async def generate(
        self,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[dict],
        llm_config: dict[str, Any],
        model: str,
        max_retries: int,
        base_delay: float,
        **kwargs,
    ) -> Message: ...

    @abstractmethod
    async def generate_stream(
        self,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[dict],
        llm_config: dict[str, Any],
        model: str,
        max_retries: int,
        base_delay: float,
        queue: asyncio.Queue,
        stream_formatter: StreamFormatter,
        **kwargs,
    ) -> Message: ...
```

**Separation from Agent:** The `Agent` subclass owns the orchestration loop (step counting, tool dispatch, compaction, memory). The `Provider` owns the LLM API call (authentication, request building via `MessageFormatter`, retry/backoff, response parsing, stream accumulation). The Agent calls `self.provider.generate(...)` — it never imports a provider SDK directly.

**Separation from MessageFormatter:** The `Provider` holds a reference to its `MessageFormatter`. Inside `generate()`, it calls `formatter.format_messages(messages, params)` to build the wire request, and `formatter.parse_response(raw)` to convert the response back to canonical `Message`. The formatter is a pure translator; the provider handles the transport (HTTP call, streaming, retries).

### 3.6 Streaming (`core/streaming.py`)

**Status:** To be defined.

Framework-level streaming types for client-facing SSE output. These are **not** provider-specific — they represent the canonical stream events the framework emits to any consumer (FastAPI endpoint, WebSocket, CLI).

```python
@dataclass
class StreamDelta:
    """Base for all stream events."""
    type: str                   # "text", "thinking", "tool_call", "tool_result", "meta", "error", ...
    agent_uuid: str
    is_final: bool = False      # Last delta in this logical chunk

@dataclass
class TextDelta(StreamDelta):
    text: str = ""

@dataclass
class ThinkingDelta(StreamDelta):
    thinking: str = ""

@dataclass
class ToolCallDelta(StreamDelta):
    tool_name: str = ""
    tool_id: str = ""
    arguments_json: str = ""    # Accumulated JSON chunk

@dataclass
class ToolResultDelta(StreamDelta):
    tool_name: str = ""
    tool_id: str = ""
    envelope_log: dict = ...    # from ToolResultEnvelope.for_conversation_log()

@dataclass
class MetaDelta(StreamDelta):
    """Framework-level meta events (init, final, files, awaiting_relay, etc.)."""
    payload: dict = ...
```

**How streaming works end-to-end:**
1. The `Provider` streams raw LLM events and writes `StreamDelta` objects to the `asyncio.Queue`.
2. The agent loop writes `ToolResultDelta` and `MetaDelta` events for tool results and lifecycle signals.
3. A `StreamFormatter` (e.g., `JsonStreamFormatter`) reads from the queue and serializes deltas into the transport format (SSE JSON envelopes, chunked with UTF-8-safe splitting).
4. The transport layer (FastAPI SSE endpoint) consumes the formatted output.

**Provider boundary:** Each provider translates its native streaming events (Anthropic's `content_block_delta`, OpenAI's `chat.completion.chunk`) into framework `StreamDelta` types. This translation lives inside the provider's `generate_stream()` method.

### 3.7 AgentResult (`core/result.py`)

**Status:** To be defined.

```python
@dataclass
class AgentResult:
    final_message: Message          # Last assistant message
    final_answer: str               # Extracted text answer
    conversation_history: list[Message]
    stop_reason: str
    model: str
    provider: str
    usage: Usage                    # Last-turn usage
    cumulative_usage: dict          # Summed across all turns
    total_steps: int
    agent_logs: list[dict] | None
    generated_files: list[dict] | None
    cost: dict | None
```

---

## 4. Tool System (`tools/` + `common_tools/`)

### 4.1 Architecture Split

The tool system is split into two modules:

- **`tools/`** — Infrastructure: the registry, the `@tool` decorator, schema generation, the `ToolResultEnvelope` ABC, and `ConfigurableToolBase`. This is framework code.
- **`common_tools/`** — Concrete tool implementations: `CodeExecutionTool`, `ReadFile`, `GlobSearch`, `GrepSearch`, `SubAgentTool`, etc. These are application code that depends on `tools/` infrastructure and `sandbox/`.

### 4.2 ToolResultEnvelope (`tools/types.py`)

The dual-projection pattern for tool results. A `ToolResultEnvelope` is **not** a `ContentBlock` — it is a pre-projection object that **produces** `ContentBlock`s for different consumers.

```python
@dataclass
class ToolResultEnvelope(ABC):
    tool_name: str
    tool_id: str
    is_error: bool
    error_message: str | None
    duration_ms: float | None

    @abstractmethod
    def for_context_window(self) -> list[ContentBlock]:
        """Minimal projection for the LLM's next turn."""
        ...

    @abstractmethod
    def for_conversation_log(self) -> dict[str, Any]:
        """Rich projection for UI display / conversation history."""
        ...
```

**Why not a fatter ToolResultBase?** Because tool result shapes vary by tool type (code exec returns stdout+files, sub-agent returns a nested conversation, web search returns citations). This is a fundamentally different axis of variation from content blocks, which vary by LLM provider. The envelope isolates the tool-type axis; the projection methods isolate the consumer-type axis.

**Auto-wrapping for legacy tools:** `ToolRegistry.execute()` always returns a `ToolResultEnvelope`. Tools that return plain `str` or legacy `ToolResult` objects are automatically wrapped in `GenericTextEnvelope` / `LegacyMediaEnvelope`. No existing tool breaks.

### 4.3 ToolRegistry (`tools/registry.py`)

- `register(name, func, schema)` — explicit single-tool registration
- `register_tools(tools: list[Callable])` — batch registration from `@tool`-decorated functions
- `execute(tool_name, tool_input, ...) -> ToolResultEnvelope` — async dispatch with auto-wrapping
- `execute_tools(tool_calls, max_parallel, max_result_tokens) -> list[ToolResultEnvelope]` — bounded parallel execution via `asyncio.Semaphore`, preserves result ordering
- `get_schemas(format="anthropic"|"openai")` — export schemas in provider-specific format
- `attach_sandbox(sandbox)` — inject the sandbox into all registered tools
- `check_for_relay(tool_calls) -> (bool, list)` — identify frontend/relay tool calls

### 4.4 How Tools Use the Sandbox

After the one-time migration, tools delegate all I/O through the `Sandbox` interface:

```python
# Before (raw OS access):
async with aiofiles.open(path, "r") as f:
    content = await f.read()

# After (sandbox-mediated):
content = await self.sandbox.read_file(path)
```

Tools never import `os`, `subprocess`, or `aiofiles`. They receive the sandbox via `ToolRegistry.attach_sandbox()` during agent initialization. The litmus test: `grep -r "import os\|import subprocess" common_tools/` returns zero results.

### 4.5 Wiring Envelopes Into the Agent Loop

The agent loop is polymorphic over envelope types:

```python
# In the agent's tool handling:
envelope: ToolResultEnvelope = await self.tool_registry.execute(tool_name, tool_input)

# Projection 1: LLM context window
context_blocks = envelope.for_context_window()
result_block = ToolResultContent(
    tool_id=envelope.tool_id,
    tool_name=envelope.tool_name,
    tool_result=context_blocks,
    is_error=envelope.is_error,
)
self.agent_config.messages.append(Message.user([result_block]))

# Projection 2: conversation log / UI
log_entry = envelope.for_conversation_log()
self.conversation.append_tool_result(log_entry)
```

No `if/elif` chains. Adding a new tool type means: define one envelope subclass in the tool's own file. Zero changes outside the tool's code.

---

## 5. Sandbox System (`sandbox/`)

### 5.1 Sandbox ABC (`sandbox/types.py`)

The sandbox abstracts all tool interactions with "the outside world" into four verbs: read a file, write a file, list a directory, run a command.

```python
class Sandbox(ABC):
    # Lifecycle
    async def setup(self) -> None: ...
    async def teardown(self) -> None: ...
    async def __aenter__(self) -> "Sandbox": ...
    async def __aexit__(self, *exc) -> None: ...

    # Filesystem (all paths are relative)
    async def read_file(self, path: str) -> str: ...
    async def write_file(self, path: str, content: str) -> None: ...
    async def read_file_bytes(self, path: str) -> bytes: ...
    async def write_file_bytes(self, path: str, content: bytes) -> None: ...
    async def list_dir(self, path: str = ".") -> list[str]: ...
    async def file_exists(self, path: str) -> bool: ...
    async def delete(self, path: str) -> None: ...

    # Execution
    async def exec(self, command: str, timeout: int = ..., cwd: str = ..., env: dict = ...) -> ExecResult: ...
    async def exec_stream(self, command: str, ...) -> AsyncIterator[str]: ...

    # File Coordination (import/export across the storage boundary)
    async def import_file(self, file_id: str, filename: str, content: bytes) -> str: ...
        # Idempotent: skips write if already imported. Returns sandbox-relative path.
    async def get_exported_files(self) -> list[tuple[str, bytes]]: ...
        # Recursively scans exports area. Returns (relative_filename, content) pairs.
```

**Key design decisions:**
- **Relative paths only.** The LLM generates paths like `src/main.py`. The sandbox resolves them internally. No consumer ever sees the real filesystem location. This is what makes sandbox levels swappable.
- **`command: str` not `list[str]` for exec.** The LLM generates shell commands as strings. Passing strings to a shell preserves pipes, redirection, and globbing.
- **Async context manager lifecycle.** `async with sandbox:` ensures `setup()` on enter, `teardown()` on exit. Resource leaks are impossible.

### 5.2 Isolation Levels

| Level | Implementation | Setup Time | Filesystem | Execution | Security Boundary |
|-------|---------------|------------|------------|-----------|-------------------|
| 1 | `LocalSandbox` | ~1ms | Path-restricted directory on host | `subprocess` on host | Organizational (path containment only) |
| 2 | `DockerSandbox` | 2-5s | Bind-mounted host directory | `docker exec` in container | Process + filesystem isolation |
| 3 | `E2BSandbox` | 5-15s | Remote VM filesystem (API calls) | Remote command execution | Full VM isolation |

**The bind-mount asymmetry (Level 2):** File operations go through a bind-mount (local disk speed), but command execution goes through `docker exec` (isolated). Best of both: fast file I/O and isolated execution.

**Path containment (`_resolve()`):** Levels 1 and 2 resolve relative paths and check containment to block `../../etc/passwd` traversal. Level 3 doesn't need it — the VM itself is the containment boundary.

### 5.3 File Coordination (Sandbox + MediaBackend)

Files cross between two worlds: the sandbox (ephemeral working files) and the media backend (persistent storage). Instead of a separate bridge class, the responsibilities are split between the two natural owners:

**Sandbox owns zone layout.** The sandbox creates three zone directories at `setup()`: `.imported/` (materialized files from storage), `workspace/` (default cwd for tools), `.exports/` (tool-produced artifacts for the user). It exposes two semantic methods — `import_file()` and `get_exported_files()` — so that callers never reason about zone paths.

**MediaBackend owns orchestration.** The `MediaBackend` ABC provides three concrete (non-abstract) methods that compose abstract storage operations with sandbox file coordination:

```python
# On MediaBackend ABC (concrete methods, not abstract):
def attach_sandbox(self, sandbox: Sandbox) -> None: ...
async def materialize(self, media_id: str, agent_uuid: str) -> str: ...
    # retrieve metadata + bytes from storage, call sandbox.import_file()
async def flush_exports(self, agent_uuid: str) -> list[MediaMetadata]: ...
    # call sandbox.get_exported_files(), store each file in backend
```

**Key invariant:** MediaBackend never knows about `.imported/`, `.exports/`, or any zone path. It calls `sandbox.import_file(media_id, filename, content)` and gets back a sandbox-relative path. It calls `sandbox.get_exported_files()` and gets back `(relative_filename, content)` pairs. The zone layout is entirely the sandbox's internal concern — different sandbox implementations can organize files however they want.

### 5.4 Sandbox Attachment

During `Agent.initialize()`, the sandbox is set up and attached to both the tool registry and the media backend:

```python
await self.sandbox.setup()              # Creates root + zone dirs (.imported/, workspace/, .exports/)
self.tool_registry.attach_sandbox(self.sandbox)   # Tools read/write through sandbox
self.media_backend.attach_sandbox(self.sandbox)    # Enables materialize() and flush_exports()
```

This connects the sandbox to both I/O paths: tools read/write files through it, and the media backend uses it for file import/export across the storage boundary.

---

## 6. Media Backend (`media_backend/`)

### 6.1 Design

The media backend is the **combined** storage + resolution system. It replaces the old `file_backends/` module. It handles both storing bytes and projecting `media_id` references into the format each consumer needs (base64 for LLM, URL for frontend, metadata reference for conversation log).

This combines what the diaries described as `FileStorageBackend` + `MediaResolver` into a single `MediaBackend` ABC.

### 6.2 MediaMetadata (`media_backend/types.py`)

```python
@dataclass
class MediaMetadata:
    media_id: str                    # Stable identity (e.g., "media_{uuid4().hex[:16]}")
    media_mime_type: str             # MIME type (e.g., "image/png", "application/pdf")
    media_filename: str              # Original filename
    media_extension: str             # File extension (e.g., "png", "pdf")
    media_size: int                  # Size in bytes
    storage_type: str                # Backend type (e.g., "local", "s3", "memory")
    storage_location: str            # Physical location (path, S3 key, etc.)
    extras: dict[str, Any]           # Backend-specific data (e.g., S3 bucket/key)
```

### 6.3 MediaBackend ABC (`media_backend/types.py`)

**Status:** Implemented (`LocalMediaBackend` complete with full test coverage).

```python
class MediaBackend(ABC):
    # Lifecycle
    async def connect(self) -> None: ...
    async def close(self) -> None: ...

    # Storage operations (abstract — each backend implements these)
    async def store(self, content: bytes, filename: str, mime_type: str,
                    agent_uuid: str) -> MediaMetadata: ...
    async def retrieve(self, media_id: str, agent_uuid: str) -> bytes | None: ...
    async def delete(self, media_id: str, agent_uuid: str) -> bool: ...
    async def exists(self, media_id: str, agent_uuid: str) -> bool: ...
    async def get_metadata(self, media_id: str, agent_uuid: str) -> MediaMetadata | None: ...

    # Resolution — abstract (projections for different consumers)
    async def to_base64(self, media_id: str, agent_uuid: str) -> dict: ...
        # Returns {"data": "<base64>", "media_type": "image/png"} for LLM provider adapters
    async def to_url(self, media_id: str, agent_uuid: str) -> str: ...
        # Returns URL for frontend rendering (presigned S3 URL, local API path, etc.)
    async def to_reference(self, media_id: str, agent_uuid: str) -> dict: ...
        # Returns lightweight metadata dict for conversation log

    # Sandbox integration — concrete (compose abstract storage + sandbox)
    _sandbox: Sandbox | None = None
    def attach_sandbox(self, sandbox: Sandbox) -> None: ...

    async def materialize(self, media_id: str, agent_uuid: str) -> str: ...
        # get_metadata() + retrieve() + sandbox.import_file()
        # Idempotent: import_file() skips write if already imported.
        # Raises RuntimeError if no sandbox, FileNotFoundError if media_id missing.

    async def flush_exports(self, agent_uuid: str) -> list[MediaMetadata]: ...
        # sandbox.get_exported_files() + store() each file
        # Uses os.path.basename() for storage filename (no slashes).
        # Preserves full relative path in metadata.extras["export_path"] for nested files.
```

**Key decisions:**
- `to_url()` lives on the backend (not a separate URL service) because URL shape is tied to storage type: local returns `/api/files/{id}`, S3 returns presigned URL.
- All methods take `agent_uuid` — files are namespaced to agent sessions for cleanup, security, and storage layout.
- `materialize()` and `flush_exports()` are **concrete** on the ABC — they compose abstract storage methods with sandbox file coordination. Concrete backends (LocalMediaBackend, S3MediaBackend) inherit them for free.
- MediaBackend imports `Sandbox` under `TYPE_CHECKING` only — no runtime circular dependency.

### 6.4 LocalFileCache (`media_backend/cache.py`)

**Status:** To be defined.

A read-through + write-back cache sitting between tools and the media backend. Not a storage backend itself — a separate performance layer.

```python
class LocalFileCache:
    async def get_path(self, media_id: str, agent_uuid: str) -> str:
        """Fast path: return cached local path. Slow path: download from backend."""

    def mark_dirty(self, media_id: str) -> None:
        """Called by write tools when they modify a cached file."""

    async def flush(self, agent_uuid: str) -> None:
        """Stream dirty files back to media backend."""

    async def cleanup(self, agent_uuid: str) -> None:
        """Flush then delete the entire agent cache directory."""
```

**Double-checked locking** prevents duplicate downloads when concurrent tools request the same file. **LRU eviction** reclaims space by evicting least-recently-accessed non-dirty files.

### 6.5 Lazy Resolution in Content Blocks

Media content blocks store `media_id` references, not inline data:

```python
ImageContent(
    media_type="image/png",
    media_id="media_abc123",
    source_type=SourceType.FILE_ID,  # Abstract reference
    # data field is transient -- not serialized
)
```

Resolution happens at the consumer boundary:
- **LLM call**: `MessageFormatter.format_messages()` encounters `source_type=FILE_ID`, calls `media_backend.to_base64(media_id)` to inject inline data.
- **Frontend SSE**: Streaming formatter calls `media_backend.to_url(media_id)` for a renderable URL.
- **Persistence**: `to_dict()` serializes the `media_id` reference, not the bytes.

---

## 7. Compaction System (`compaction/`)

Context window management. Operates on canonical `Message` objects — completely provider-agnostic.

### 7.1 Compactor ABC

```python
class Compactor(ABC):
    @abstractmethod
    async def apply_compaction(self, agent_config: AgentConfig) -> tuple[bool, list[Message]]:
        """Returns (did_compact, compacted_messages)."""
        ...
```

### 7.2 Strategies

| Compactor | Strategy |
|-----------|----------|
| `NoOpCompactor` | Pass-through, no compaction |
| `SlidingWindowCompactor` | **Default.** 4-phase progressive: (1) remove thinking blocks, (2) truncate long tool results, (3) replace old tool results with placeholders, (4) remove old turn pairs |
| `ToolResultRemovalCompactor` | 2-phase: (1) replace old tool results, (2) remove old turn pairs |
| `SummarizingCompactor` | LLM-based summarization of older history using summary tags |

Compactors are stateless — they receive the full `AgentConfig` and return compacted messages. They use `model` from the config to look up model-specific token limits.

### 7.3 Separation from Memory

Compactors and memory stores serve different purposes:
- **Compactors** manage **within-session** context window pressure. They operate every step of the agent loop.
- **Memory stores** manage **across-session** knowledge. They operate once at run start (retrieve) and once at run end (update).

They are completely independent and do not interact.

---

## 8. Memory System (`memory/`)

Cross-session knowledge stores. Operate at run boundaries only.

```python
class MemoryStore(ABC):
    @abstractmethod
    async def retrieve(self, user_message: Message, messages: list[Message], **kwargs) -> list[ContentBlock]:
        """Called at run start. Returns content blocks to inject into the prompt."""
        ...

    @abstractmethod
    async def update(self, messages: list[Message], conversation_history: list[Message], **kwargs) -> dict:
        """Called at run end. Extracts and stores learnings from this run."""
        ...
```

---

## 9. Storage System (`storage/`)

Three-adapter pattern, carried over from `anthropic_agent` mostly as-is. Each adapter is independently swappable.

| Adapter | Entity | Purpose |
|---------|--------|---------|
| `AgentConfigAdapter` | `AgentConfig` | Agent session state (messages, model, file registry, relay state) |
| `ConversationAdapter` | `Conversation` | Per-run conversation records for UI history/pagination |
| `AgentRunAdapter` | `AgentRunLog` | Step-by-step execution logs for debugging |

Backends: `memory` (default), `filesystem`, `postgres`.

Factory: `create_adapters(backend_type, **kwargs)` returns `(config_adapter, conversation_adapter, run_adapter)`.

---

## 10. Provider Implementations (`providers/`)

### 10.1 Responsibilities of a Provider Implementation

Each provider directory (`providers/anthropic/`, `providers/openai/`, etc.) contains:

1. **Provider subclass** — Implements `Provider.generate()` and `Provider.generate_stream()`. Handles: SDK client creation, authentication, request building (via formatter), retry with backoff, streaming event accumulation, error classification (retryable vs. non-retryable).

2. **MessageFormatter subclass** — Implements `format_messages()` and `parse_response()`. Translates canonical `ContentBlock`s to/from provider wire format. This is where provider-specific content type mappings live (e.g., Anthropic's `source.type = "base64"` image format vs OpenAI's `image_url` format). The formatter also handles media resolution — calling `media_backend.to_base64()` for `FILE_ID` source types during format.

3. **Agent subclass** — Implements `Agent.run()` and `Agent.resume_run_with_tool_results()`. Contains the orchestration loop: step counting, compaction, LLM calls (via Provider), tool dispatch (via ToolRegistry), streaming, memory, persistence. Provider-specific agent behavior (e.g., Anthropic cache control, container management) lives here.

4. **LLM config dataclass** — Provider-specific configuration (e.g., `AnthropicLLMConfig` with `thinking_tokens`, `server_tools`, `beta_headers`). Serialized into `AgentConfig.llm_config` for persistence.

### 10.2 Anthropic Provider (Concrete Example)

```
providers/anthropic/
├── agent.py          # AnthropicAgent
│                     #   - Orchestration loop with Anthropic-specific behavior
│                     #   - Cache control (4-priority slot allocation)
│                     #   - Container management
│                     #   - Subagent hierarchy wiring
├── provider.py       # AnthropicProvider
│                     #   - anthropic.AsyncAnthropic() client
│                     #   - Retry with exponential backoff + jitter
│                     #   - Streaming event accumulation into canonical Message
├── formatter.py      # AnthropicMessageFormatter
│                     #   - Canonical ContentBlock -> Anthropic API format
│                     #   - Media resolution (FILE_ID -> base64) during formatting
│                     #   - Response parsing (BetaMessage -> canonical Message)
└── config.py         # AnthropicLLMConfig
                      #   - thinking_tokens, max_tokens, server_tools, skills,
                      #     beta_headers, container_id
```

### 10.3 What Lives Where — Provider vs. Framework

| Concern | Owner | Why |
|---------|-------|-----|
| Agentic loop (step counting, tool dispatch, compaction calls) | Provider's Agent subclass | Loop structure may differ per provider (e.g., Anthropic's `pause_turn`, tool relay) |
| LLM API call (auth, request, retry, streaming) | Provider's Provider subclass | SDK and wire format are provider-specific |
| Wire format translation | Provider's MessageFormatter | Content block mappings differ per provider |
| Media resolution at format time | Provider's MessageFormatter | Resolution format depends on provider expectations |
| Tool execution and enveloping | Framework (`tools/`) | Provider-agnostic — tools return canonical envelopes |
| Sandbox lifecycle | Framework (`sandbox/`) | Provider-agnostic — tools delegate to sandbox |
| Compaction | Framework (`compaction/`) | Operates on canonical Messages |
| Memory stores | Framework (`memory/`) | Operates on canonical Messages |
| Storage adapters | Framework (`storage/`) | Operates on canonical dataclasses |
| Streaming delta types | Framework (`core/streaming.py`) | Canonical event types, provider writes them |
| Stream formatting for transport | Framework (shared module) | SSE envelope format is not provider-specific |

---

## 11. Composition and Wiring

### 11.1 Agent Construction

```python
agent = AnthropicAgent(
    system_prompt="You are a helpful assistant.",
    model="claude-sonnet-4-5",
    config=AnthropicLLMConfig(thinking_tokens=10000),

    tools=[my_tool_a, my_tool_b],
    subagents={"research": research_agent},

    sandbox=DockerSandbox(image="python:3.12-slim"),
    media_backend=S3MediaBackend(bucket="my-bucket"),
    compactor=SlidingWindowCompactor(),
    memory_store=NoOpMemoryStore(),

    config_adapter=PostgresAgentConfigAdapter(db_url),
    conversation_adapter=PostgresConversationAdapter(db_url),
    run_adapter=PostgresAgentRunAdapter(db_url),
)
```

### 11.2 Initialization Flow

```
agent.__init__()           # Synchronous. Stores params. Creates Provider + MessageFormatter.
                           # Defaults: MemoryAdapters, LocalMediaBackend, SlidingWindowCompactor,
                           #           LocalSandbox.

await agent.initialize()   # Async. Loads state from storage OR creates fresh state.
                           # 1. Generate agent_uuid if new.
                           # 2. Create AgentConfig + Conversation.
                           # 3. Setup sandbox (async with sandbox).
                           # 4. Attach sandbox to tool_registry and media_backend.
                           # 5. Set _initialized = True.
```

### 11.3 Run Loop Flow

```
await agent.run(prompt)
  |
  +-- initialize() if needed
  +-- memory_store.retrieve() -> inject context into prompt
  +-- append prompt to agent_config.messages + conversation
  |
  +-- _resume_loop():
       while step < max_steps:
       |
       +-- compactor.apply_compaction(agent_config)
       |
       +-- provider.generate[_stream](                    # Provider handles:
       |    messages=agent_config.messages,               #   formatter.format_messages()
       |    tool_schemas=..., llm_config=..., model=..., #   SDK API call + retry
       |    queue=..., stream_formatter=...               #   formatter.parse_response()
       |  ) -> Message                                   #   StreamDelta emission (if streaming)
       |
       +-- append assistant message to agent_config + conversation
       |
       +-- match stop_reason:
            |
            +-- "pause_turn" -> continue
            |
            +-- "tool_use":
            |    tool_calls = extract_tool_calls(response)
            |    need_relay, relay_calls = tool_registry.check_for_relay(tool_calls)
            |    if need_relay:
            |      execute backend tools -> save partial results -> emit relay event -> return
            |    else:
            |      envelopes = tool_registry.execute_tools(tool_calls)
            |      for each envelope:
            |        context_blocks = envelope.for_context_window()  -> agent_config.messages
            |        log_entry = envelope.for_conversation_log()     -> conversation
            |      continue
            |
            +-- "end_turn":
                 if final_answer_check: validate, retry on failure
                 memory_store.update()
                 build AgentResult -> return
```

### 11.4 Dependency Graph

```
                    +-------------------+
                    |    Agent (ABC)    |
                    |  (orchestration)  |
                    +--------+----------+
                             | uses
          +------------------+------------------+
          |                  |                  |
    +-----v-----+    +------v------+    +------v------+
    |  Provider  |    | ToolRegistry|    |  Compactor  |
    | (LLM API)  |    | (dispatch)  |    | (ctx mgmt)  |
    +-----+-----+    +------+------+    +-------------+
          |                 |
    +-----v------+   +------v------+
    | Formatter  |   |   Sandbox   |
    | (wire fmt) |   | (isolation) |
    +-----+------+   +------+------+
          |                 ^
          |                 | attach_sandbox()
          +--------->  MediaBackend
                     (store+resolve+
                      materialize+
                      flush_exports)
                     +------+------+
                            |
                     +------v------+
                     |  FileCache  |
                     (read-through)
                     +-------------+

    +-------------+  +-------------+  +-------------+
    |ConfigAdapter|  |ConvAdapter  |  | RunAdapter  |
    (session state)  (run records)   (exec logs)
    +-------------+  +-------------+  +-------------+

    +-------------+
    | MemoryStore |
    (cross-session)
    +-------------+
```

---

## 12. Axes of Change

Each axis of change is isolated so that modifications along one axis require changes in one place only.

| Axis | What Changes | Where It Lives | Adding a New One Touches |
|------|-------------|----------------|------------------------|
| LLM Provider | Wire format, API call, streaming events | `providers/<name>/` | 1 directory (provider + formatter + agent + config) |
| Tool Type | Result shape, projections | `common_tools/<tool>.py` (envelope subclass) | 1 file |
| Sandbox Level | Isolation mechanism | `sandbox/<level>.py` | 1 file |
| Storage Backend | Persistence mechanism | `storage/adapters/<backend>.py` | 1 file |
| Media Backend | Where bytes live + how they're resolved | `media_backend/<backend>.py` | 1 file |
| Compaction Strategy | How context is trimmed | `compaction/<strategy>.py` | 1 file |
| Stream Transport | SSE format, chunking | Stream formatter module | 1 file |
| Content Block Type | New canonical content type | `core/types.py` + formatters | types.py + each provider's formatter |

---

## 13. Migration Notes (anthropic_agent -> agent_base)

### Carried Over As-Is
- Storage three-adapter pattern (`storage/`)
- Compaction strategies (refactored into own module)
- Memory store interface
- Structured logging
- Token counting heuristics

### Refactored
- `AnthropicAgent` split into `Agent` ABC + `AnthropicAgent` subclass + `AnthropicProvider` + `AnthropicMessageFormatter`
- `file_backends/` merged into `media_backend/` (storage + resolution combined, streaming support, sandbox integration)
- `ToolResult` return type replaced by `ToolResultEnvelope` dual-projection pattern
- Raw OS I/O in tools replaced by sandboxed I/O through `Sandbox` interface
- Provider-specific streaming replaced by framework `StreamDelta` types + per-provider translation

### New
- `Sandbox` system with three isolation levels + file coordination (`import_file`, `get_exported_files`)
- `LocalFileCache` for tool I/O performance
- `Provider` ABC separating LLM API calls from orchestration
- Framework-level `StreamDelta` types
- `MediaBackend` with combined storage + resolution + sandbox orchestration (`materialize`, `flush_exports`)
- Three-zone sandbox layout (`.imported/`, `workspace/`, `.exports/`) managed by sandbox `setup()`

---

## 14. File Coordination

Files cross between two worlds: the sandbox (ephemeral working files) and the media backend (durable persistent storage). This section describes how the two modules collaborate.

### 14.1 Three-Zone Sandbox Layout

The sandbox creates three zone directories at `setup()`:

```
sandbox_root/
├── .imported/              # Materialized files from MediaBackend
│   └── {media_id}/         #   Lazily populated via import_file()
│       └── filename.ext    #   Idempotent: skips write if already present
│
├── workspace/              # LLM's working directory (default cwd for exec)
│   └── src/...             #   Tools read/write freely
│                           #   Session-scoped — dies with sandbox
│
└── .exports/               # Tool-produced artifacts for the user
    └── report.csv          #   Convention: write here if the user should keep it
                            #   Flushed to MediaBackend via flush_exports()
```

**Design rationale:**
- `.` prefix on system-managed zones keeps them out of the LLM's natural working space
- `workspace/` is the default `cwd` for tool execution
- `.exports/` is the only zone tools explicitly write to for user-facing output
- Zone layout is the sandbox's internal concern — callers use `import_file()` and `get_exported_files()`, never zone paths

### 14.2 File Flow Patterns

**Inbound (MediaBackend → Sandbox):**
```
User uploads file → API layer stores via media_backend.store()
  → MediaMetadata created with media_id
  → File NOT yet in sandbox
  → ...later, tool needs the file...
  → media_backend.materialize(media_id, agent_uuid)
    → retrieves metadata + bytes from storage
    → calls sandbox.import_file(media_id, filename, content)
    → sandbox writes to .imported/{media_id}/{filename} (idempotent)
    → returns sandbox-relative path
  → Tool reads via sandbox.read_file(path)
```

**Outbound (Sandbox → MediaBackend):**
```
Tool generates user-facing output
  → Tool writes to sandbox: sandbox.write_file(".exports/report.csv", data)
  → At end of run (or periodic sync):
    → media_backend.flush_exports(agent_uuid)
      → calls sandbox.get_exported_files()
      → for each (relative_path, content):
        → store(content, basename, mime_type, agent_uuid)
        → if nested: metadata.extras["export_path"] = relative_path
      → returns list[MediaMetadata]
```

**Session workspace (sandbox-only):**
```
Tool writes to workspace/src/main.py
  → Lives and dies with the sandbox
  → Not tracked. Not persisted. Destroyed on teardown.
```

### 14.3 Agent Loop Integration

```python
# Initialization
await sandbox.setup()                           # Creates .imported/, workspace/, .exports/
media_backend.attach_sandbox(sandbox)
tool_registry.attach_sandbox(sandbox)

# Before tool execution (if tool needs a user-uploaded file)
path = await media_backend.materialize(media_id, agent_uuid)
# → tool can now read via sandbox.read_file(path)

# After tool execution (end of run or periodic sync)
exported = await media_backend.flush_exports(agent_uuid)
# → each exported file is now stored durably in MediaBackend

# Teardown (triggered by backend, not necessarily at end of agent loop)
await sandbox.teardown()
```

### 14.4 Open Questions

- **Sandbox lifetime policy:** When should sandboxes be torn down? Options: session end, idle timeout, resource pressure, admin action. Teardown is independent of the agent loop — the backend can trigger it externally.
- **Flush frequency:** End-of-run flush is the baseline. Periodic flush after each tool step that writes to `.exports/` improves crash recovery at minimal cost.
- **Provider server files:** Files that live on the provider's servers (e.g., Anthropic container files) are represented as metadata-only references in MediaBackend. They are never materialized into the sandbox. Resolution (`to_base64`, `to_url`) calls the provider API directly.

5. **Provider file resolution strategy:** How should REFERENCE files be resolved when the provider API is unavailable?
   - Options: fail fast, return cached version (if previously resolved), return placeholder.
   - This is a MediaBackend implementation detail, not a framework-level decision.

6. **Cross-sandbox file sharing (subagents):** When a subagent runs in its own sandbox, how do files flow between parent and child?
   - Option A: Shared MediaBackend — subagent writes to `.exports/`, parent reads via `media_id`.
   - Option B: Explicit file transfer via the `SubAgentEnvelope` result.
   - Likely: Option A (MediaBackend is already shared), with the envelope carrying `media_id` references.
