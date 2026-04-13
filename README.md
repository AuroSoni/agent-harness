<p align="center">
  <h1 align="center">agent-base</h1>
  <p align="center">
    The production-ready foundation for building your own AI agents.
    <br />
    <em>Stop gluing together SDKs. Start shipping agents.</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha" />
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License" />
    <img src="https://img.shields.io/badge/async-first-purple" alt="Async First" />
  </p>
</p>

---

**agent-base** is a Python framework that gives you everything you need to build, run, and scale AI agents — not just the LLM call, but the entire infrastructure around it.

Most agent libraries give you a thin wrapper around API calls and leave you to figure out storage, sandboxing, media handling, streaming, cost tracking, and tool execution on your own. **agent-base** ships all of that out of the box, with every component swappable via clean adapter interfaces.

If you're an AI startup building an agentic product, this is your day-one codebase.

## Why agent-base?

| What you get | Instead of building it yourself |
|---|---|
| **Multi-provider agents** | Anthropic (native) + OpenAI, Gemini, etc. via LiteLLM |
| **Tool execution engine** | `@tool` decorator, auto-schema generation, sync & async support |
| **Sandboxed execution** | Local, Docker, or E2B cloud sandboxes — swap with one line |
| **Persistent storage** | Memory, filesystem, or PostgreSQL — three-adapter pattern |
| **Media backend** | Local filesystem, S3, or in-memory — with caching layer |
| **Streaming** | XML, JSON, or raw formatters piped to `asyncio.Queue` |
| **Context management** | Sliding window, LLM summarization, or tool-result removal compaction |
| **Cost tracking** | Per-run token counting and cost calculation |
| **Session resumption** | Reload full agent state from storage by UUID |
| **Subagent orchestration** | Compose agents that delegate to specialized child agents |
| **Frontend tool relay** | Pause execution, relay tool calls to a UI, resume on response |
| **Built-in common tools** | Code execution, file ops, grep, glob, patch, todos, planning |
| **Structured logging** | Production-grade JSON logging via structlog |

## Quick Start

### Install

```bash
# Clone the repo
git clone https://github.com/anthropics/agent-base.git
cd agent-base

# Install with uv (recommended)
uv sync
```

### Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: for multi-provider support via LiteLLM
export OPENAI_API_KEY="sk-..."
```

### Your first agent (5 lines)

```python
import asyncio
from agent_base.providers.anthropic import AnthropicAgent

async def main():
    agent = AnthropicAgent(system_prompt="You are a helpful assistant.")
    result = await agent.run("What's the capital of France?")
    print(result.final_answer)

asyncio.run(main())
```

### Add tools

```python
from agent_base.tools import tool

@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return str(a * b)

@tool
def search_database(query: str) -> str:
    """Search the product database."""
    # Your real implementation here
    return f"Found 3 results for '{query}'"

agent = AnthropicAgent(
    system_prompt="Use tools to help the user.",
    tools=[multiply, search_database],
)
result = await agent.run("What is 42 * 17?")
```

### Persist everything

```python
from agent_base.storage import create_adapters

# Swap "filesystem" for "postgres" or "memory" — same interface
config_adapter, conv_adapter, run_adapter = create_adapters(
    "filesystem", base_path="./data"
)

agent = AnthropicAgent(
    system_prompt="You are a helpful assistant.",
    tools=[multiply],
    config_adapter=config_adapter,
    conversation_adapter=conv_adapter,
    run_adapter=run_adapter,
)

# Run it
result = await agent.run("Calculate 15 * 28")

# Later — resume the exact same session
resumed_agent = AnthropicAgent(
    agent_uuid=agent.agent_uuid,
    tools=[multiply],
    config_adapter=config_adapter,
    conversation_adapter=conv_adapter,
    run_adapter=run_adapter,
)
result = await resumed_agent.run("What did we calculate before?")
```

### Stream responses

```python
import asyncio

queue = asyncio.Queue()

async def print_stream():
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        print(chunk, end="", flush=True)

printer = asyncio.create_task(print_stream())
result = await agent.run("Explain quantum computing", queue)
await queue.put(None)
await printer
```

### Use any LLM provider

```python
from agent_base.providers.litellm import LiteLLMAgent

# OpenAI
agent = LiteLLMAgent(model="openai/gpt-4o", tools=[multiply])

# Gemini
agent = LiteLLMAgent(model="gemini/gemini-2.5-flash", tools=[multiply])

# Any LiteLLM-supported provider
agent = LiteLLMAgent(model="anthropic/claude-sonnet-4-5", tools=[multiply])
```

## Architecture

```
agent_base/
├── core/                  # Provider-agnostic domain model & contracts
│   ├── agent.py           # Agent ABC — the run loop
│   ├── provider.py        # Provider ABC — LLM interface
│   ├── messages.py        # Canonical Message & Usage types
│   ├── config.py          # AgentConfig, Conversation dataclasses
│   └── types.py           # ContentBlock hierarchy
│
├── providers/             # LLM provider implementations
│   ├── anthropic/         # Native Anthropic SDK integration
│   └── litellm/           # LiteLLM (OpenAI, Gemini, Mistral, etc.)
│
├── tools/                 # Tool infrastructure
│   ├── decorators.py      # @tool decorator & schema generation
│   ├── registry.py        # ToolRegistry — register, execute, export
│   └── base.py            # ConfigurableToolBase ABC
│
├── common_tools/          # Ready-to-use tool implementations
│   ├── code_execution     # Sandboxed Python/shell execution
│   ├── read_file          # File reading with line ranges
│   ├── apply_patch        # Unified diff patching
│   ├── glob_file_search   # Fast file pattern matching
│   ├── grep_search        # Content search with regex
│   ├── todo_write         # Task management
│   └── sub_agent_tool     # Subagent delegation
│
├── sandbox/               # Execution isolation
│   ├── local.py           # LocalSandbox — path-restricted, ~1ms setup
│   ├── docker.py          # DockerSandbox — container isolation
│   └── e2b.py             # E2BSandbox — cloud VM isolation
│
├── media_backend/         # Media storage & resolution
│   ├── local.py           # Local filesystem
│   ├── s3.py              # AWS S3
│   └── memory.py          # In-memory (testing)
│
├── storage/               # Persistence (three-adapter pattern)
│   └── adapters/          # Memory, Filesystem, PostgreSQL
│
├── streaming/             # Stream formatters (XML, JSON, raw)
├── pricing/               # Token cost calculation
├── memory/                # Cross-session memory stores
└── logging/               # Structured logging via structlog
```

### Design Principles

1. **Provider-agnostic core** — The canonical model (`ContentBlock`, `Message`, `AgentConfig`) lives in `core/`. Provider-specific translation lives in `providers/<name>/`.
2. **Composition over inheritance** — The agent is assembled from swappable components: `Provider`, `Sandbox`, `MediaBackend`, `Compactor`, storage adapters. All wired at construction time.
3. **Tools never touch the OS directly** — All file I/O and command execution flows through a `Sandbox`. Swap `LocalSandbox` for `DockerSandbox` by changing config, not code.
4. **Adapter pattern everywhere** — Storage, media, sandboxing, compaction, memory — every subsystem follows the same pattern: ABC defines the interface, concrete classes implement per backend.

## Interactive Notebooks

The best way to explore agent-base hands-on is through the included Jupyter notebooks:

| Notebook | What you'll learn |
|---|---|
| [`providers/anthropic/anthropic_agent_test.ipynb`](agent_base/providers/anthropic/anthropic_agent_test.ipynb) | Full Anthropic agent walkthrough — tools, streaming, resumption, subagents |
| [`providers/litellm/litellm_agent_test.ipynb`](agent_base/providers/litellm/litellm_agent_test.ipynb) | Multi-provider agents with LiteLLM — OpenAI, multimodal, frontend relay, abort/steer |
| [`sandbox/sandbox_test.ipynb`](agent_base/sandbox/sandbox_test.ipynb) | Sandbox system — file isolation, shell execution, import/export |
| [`tools/registry_test.ipynb`](agent_base/tools/registry_test.ipynb) | Tool system deep-dive — `@tool` decorator, schemas, registry |

```bash
# Start Jupyter and explore
uv run jupyter notebook
```

## FastAPI Demo

A production-ready FastAPI server is included in `demos/fastapi_server/`, featuring SSE streaming, file uploads, and session management.

```bash
# Start the server
uv run --directory demos/fastapi_server uvicorn main:app --reload --port 8000

# Try it out
curl -N -X POST "http://127.0.0.1:8000/agent/run" \
  -H "content-type: application/json" \
  -d '{"user_prompt": "Hello! What can you do?"}'
```

**Endpoints:**
- `POST /agent/run` — Run agent with SSE streaming
- `POST /agent/tool_results` — Resume after frontend tool execution
- `GET /agent/sessions` — List sessions with pagination
- `GET /agent/{uuid}/conversations` — Conversation history
- `POST /agent/upload` — Upload files to Anthropic Files API

See [`demos/fastapi_server/README.md`](demos/fastapi_server/README.md) for full API documentation.

## Extending agent-base

Every component is designed to be swapped. Here's how:

### Custom storage backend

```python
from agent_base.storage import AgentConfigAdapter

class RedisConfigAdapter(AgentConfigAdapter):
    async def save(self, config): ...
    async def load(self, uuid): ...
    async def delete(self, uuid): ...
    # Implement the remaining abstract methods
```

### Custom sandbox

```python
from agent_base.sandbox.types import Sandbox

class KubernetesSandbox(Sandbox):
    async def setup(self): ...
    async def exec(self, command): ...
    async def teardown(self): ...
```

### Custom media backend

```python
from agent_base.media_backend.types import MediaBackend

class GCSMediaBackend(MediaBackend):
    async def store(self, media_id, data, metadata): ...
    async def retrieve(self, media_id): ...
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (for Anthropic provider) | Anthropic API key |
| `OPENAI_API_KEY` | For LiteLLM with OpenAI | OpenAI API key |
| `DATABASE_URL` | For PostgreSQL storage | PostgreSQL connection string |
| `S3_BUCKET` | For S3 media backend | S3 bucket name |
| `AWS_ACCESS_KEY_ID` | For S3 media backend | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | For S3 media backend | AWS secret key |

## Requirements

- Python **3.10+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An API key for your chosen LLM provider

## Contributing

We welcome contributions! This project is in active development (alpha), so there's plenty of opportunity to shape the direction.

```bash
# Install dev dependencies
uv sync

# Run tests
pytest -v --tb=short

# Run integration tests (requires API keys)
pytest -v -m integration
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for builders. Ship your agent, not your infrastructure.
</p>
