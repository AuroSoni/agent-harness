# CLAUDE.md — anthropic-agent

## Project Overview

Python library for building production-ready AI agents on Anthropic's Claude API. Async-first design with streaming, tool execution, state persistence, and multimodal support.

## Quick Reference

```bash
# Install dependencies
uv sync

# Run tests
pytest
pytest -v --tb=short

# Run FastAPI demo server
uv run --directory demos/fastapi_server uvicorn main:app --reload --port 8000

# Run simple demo
uv run python main.py
```

## Architecture

```
anthropic_agent/          # Main library package
├── core/                 # AnthropicAgent, AgentResult, retry logic, compaction
├── tools/                # Tool registry, @tool decorator, schema generation
├── streaming/            # Stream rendering (XML/raw formatters), async queue
├── storage/              # Adapter-based persistence (filesystem, postgres, memory)
│   └── adapters/         # Concrete implementations per backend
├── file_backends/        # File storage (local filesystem, S3)
├── logging/              # Structured logging via structlog
├── memory/               # Cross-session memory stores (retrieve/update at run boundaries)
├── common_tools/         # Built-in tools (code exec, glob, grep, read, patch)
└── python_executors/     # Python code execution (AST evaluator, local executor)

demos/
├── fastapi_server/       # Production FastAPI server with SSE streaming
└── vite_app/             # React frontend (optional)

tests/                    # pytest test suite
data/                     # Runtime persistence (agent_config, conversations, runs)
```

## Key Patterns & Conventions

- **Python 3.10+**, async/await throughout
- **snake_case** for functions/variables, **PascalCase** for classes
- Full **type hints** on all public functions; use `TYPE_CHECKING` block for import-only types
- **Adapter pattern** for storage: ABC defines interface, concrete classes per backend
- **Factory functions** for instantiation (e.g., `create_adapters()`)
- **Dataclasses** for data structures (`AgentResult`, `AgentConfig`, `Conversation`, `AgentRunLog`)
- **Protocol** classes for interface definitions (e.g., `SQLBackend`)
- Tools defined with `@tool` decorator — docstrings become schema descriptions
- Imports organized: stdlib > third-party > relative
- Private members prefixed with `_`

## Storage System (Three-Adapter Pattern)

```python
# Each adapter is independent and swappable
config_adapter: AgentConfigAdapter       # Agent session state
conversation_adapter: ConversationAdapter # Per-run conversation records
run_adapter: AgentRunAdapter             # Step-by-step execution logs

# Backends: "memory" (default), "filesystem", "postgres"
from anthropic_agent.storage import create_adapters
config, conversation, run = create_adapters("filesystem", base_path="./data")
```

## Environment Variables

- `ANTHROPIC_API_KEY` — required
- `DATABASE_URL` — PostgreSQL connection string (optional, for postgres backend)
- `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` — optional, for S3 file backend

## Important Notes

- The `database/` module is **deprecated** — use `storage/` adapters instead
- Default model is `claude-sonnet-4-5`
- `uv` is the package manager (not pip); workspace includes `demos/fastapi_server`
- Build backend is **hatchling**
- Retry logic uses exponential backoff for rate limits and transient API errors
- Streaming formatters: `"xml"` (structured, default) and `"raw"` (unformatted)
- Compaction strategies manage context window: `SlidingWindowCompactor` (default, ~160k tokens), `SummarizingCompactor` (LLM-based summarization), `ToolResultRemovalCompactor`


# gstack

Use the `/browse` skill from gstack for **all** web browsing. **Never** use `mcp__claude-in-chrome__*` tools.

Available gstack skills:

- `/office-hours`
- `/plan-ceo-review`
- `/plan-eng-review`
- `/plan-design-review`
- `/design-consultation`
- `/design-shotgun`
- `/design-html`
- `/review`
- `/ship`
- `/land-and-deploy`
- `/canary`
- `/benchmark`
- `/browse`
- `/connect-chrome`
- `/qa`
- `/qa-only`
- `/design-review`
- `/setup-browser-cookies`
- `/setup-deploy`
- `/setup-gbrain`
- `/retro`
- `/investigate`
- `/document-release`
- `/document-generate`
- `/codex`
- `/cso`
- `/autoplan`
- `/plan-devex-review`
- `/devex-review`
- `/careful`
- `/freeze`
- `/guard`
- `/unfreeze`
- `/gstack-upgrade`
- `/learn`
