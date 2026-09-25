# any-llm Provider — Interface Design

Status: **interface only** — every behavioural method body is
`NotImplementedError`; dataclasses, constants, attribute wiring and the two
test notebooks are final. Implementation fills the stubs without changing
any signature.

## 1. What this provider is

A third `Provider` value (after `anthropic`, `litellm`) wrapping Mozilla
AI's [any-llm](https://docs.mozilla.ai/) (`pip install any-llm-sdk`, import
module `any_llm`). One provider routes to 50+ model providers through the
OpenAI-compatible **Completions** surface (`any_llm.acompletion`); the
Responses API is not used (only a handful of any-llm providers — 7 of ~50 at
the time of writing — support it; Anthropic does not).

Zero changes to `agent_base/core/` — the package mirrors
`agent_base/providers/litellm/` file-for-file:

| File | Contents |
|---|---|
| `provider.py` | `AnyLLMProvider(Provider)`, `DEFAULT_MODEL = "openai:gpt-4o-mini"`, `_resolve_provider_and_model` |
| `any_llm_config.py` | `AnyLLMConfig(LLMConfig)` — the native config + escape hatches |
| `formatters.py` | `AnyLLMMessageFormatter(MessageFormatter)` — block ↔ OpenAI wire |
| `token_estimation.py` | `AnyLLMTokenEstimator` — heuristic (`estimate_message`/`estimate_messages`) |
| `compaction.py` | `CompactionConfig` + `CompactionController` (litellm port) |
| `context_externalizer.py` | `ExternalizationConfig` + `ContextExternalizer` (litellm port) |
| `any_llm_agent.py` | `AnyLLMAgent(AnthropicAgent)` — Style-3 factory |
| `any-llm-test.ipynb` | provider round-trip notebook |
| `any-llm-agent-test.ipynb` | agent capability notebook |

## 2. The escape-hatch pattern (the defining design decision)

This provider deliberately exposes **no provider-specific functionality by
default** — no server tools, citations, prompt caching, skills, containers,
beta headers, or API-key rotation (the entire `AnthropicLLMConfig` feature
surface is absent on purpose). Instead, `AnyLLMConfig` carries exactly two
passthroughs, mapping 1:1 onto any-llm's own passthrough mechanisms:

- **`api_kwargs: dict`** → merged verbatim into the `acompletion(**...)`
  call, **applied last** (wins over provider-built params). Any-llm forwards
  unknown kwargs to the underlying SDK — e.g. `temperature`,
  `response_format`, `tool_choice`, `parallel_tool_calls`, Mistral's
  `safe_prompt`, an Anthropic `thinking` dict.
- **`client_args: dict`** → forwarded to the any-llm provider **client
  constructor** (not the request) — e.g. `{"timeout": 30}`. any-llm has no
  top-level `timeout` param.

Promotion rule: a knob becomes a first-class config field only when any-llm
itself harmonises it across providers. That is why `reasoning_effort`
(`'none'|'minimal'|'low'|'medium'|'high'|'xhigh'|'max'|'auto'`) and
`max_tokens` are first-class while everything else is not.

Consequences the formatter enforces: unknown/unsupported content blocks
raise `ValueError` (loud failure, no silent drops); the provider emits only
`TextDelta` / `ThinkingDelta` / `ToolCallDelta`; only generic canonical
blocks (text/thinking/tool_use) are produced.

## 3. Provider/model resolution

any-llm identifies a model as `(provider, model)`. The docs *recommend* a
separate `provider=` parameter (covered by `AnyLLMConfig.provider`); the
combined **colon** form `"openai:gpt-4o-mini"` is the supported
single-string alternative, which this provider adopts as its model-string
convention because the runtime carries one model string. (The slash form
`provider/model` is deprecated upstream — do not use it.) Precedence (see
`_resolve_provider_and_model`):

1. `AnyLLMConfig.provider` set → passed as `provider=`, runtime `model`
   string used verbatim as the bare model id.
2. Otherwise → the `model` string must be colon-form and is passed through
   for any-llm to parse.

Credentials default to any-llm's per-provider env vars (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, …); `AnyLLMConfig.api_key` /
`api_base` are per-call overrides.

## 4. Contract points implementers must honour

- **Retries**: any-llm exposes **no retry mechanism** (none documented, no
  retry params on `acompletion`). `generate`/`generate_stream` wrap the call
  in exponential backoff driven by `self.retry_policy` (O12c), retrying only
  errors `classify_error` marks retriable.
- **Errors**: match any-llm's unified hierarchy first (`AnyLLMError`,
  `RateLimitError`, `AuthenticationError`, `ModelNotFoundError`,
  `InvalidRequestError`, `ProviderError` — alias the last one; it clashes
  with `agent_base.core.provider.ProviderError`), then fall back to
  status-code sniffing on raw SDK exceptions (unified exceptions are opt-in
  via `ANY_LLM_UNIFIED_EXCEPTIONS`). Context-window failures must map to
  `ErrorCode.CONTEXT_OVERFLOW` so the runtime can compact-and-retry.
- **Streaming**: request `stream_options={"include_usage": True}` — an
  OpenAI-convention passthrough, not an any-llm guarantee; usage-in-stream
  is provider-dependent and its absence must not break the parse. Text and
  reasoning (`delta.reasoning`) stream incrementally; tool calls buffer by
  OpenAI `index` and emit one terminal `ToolCallDelta(is_final=True)` each;
  `stream_bookkeeping` is `list[ChainToolCall]` (litellm shape) consumed by
  `plan_stream_abort`; final message is rebuilt from collected chunks
  through the same parse path as `generate`.
- **Reasoning asymmetry**: `message.reasoning` (object with `.content`) is
  parsed IN as `ThinkingContent`; `ThinkingContent` is dropped on the way
  OUT (OpenAI-style APIs reject assistant reasoning as input).
- **Tool schemas**: always send dicts built by `format_tool_schemas`. Never
  hand any-llm raw Python callables — agent-base owns schemas
  (`ToolRegistry`) and execution.
- **Return shape**: current contract — `ProviderTurn` from both generate
  paths (`.message` is the canonical assistant `Message`,
  `provider="any_llm"`, normalised `stop_reason`, `Usage` with
  `thinking_tokens` from `completion_tokens_details.reasoning_tokens`).
- **Token estimation**: heuristic (chars/4 + media constants) — any-llm has
  no unified token counter.
- **Compaction / externalizer**: port the litellm modules; the only
  provider-specific line is `_build_summary_config()` →
  `AnyLLMConfig(reasoning_effort="none")`.

## 5. Notebook conventions

Both notebooks live in this package dir, import `agent_base` absolutely (no
`sys.path` games), load env via `load_dotenv("../../../.env")`, and use the
**current** runtime API (`ProviderTurn`, `end_turn_hook`, `await_input` +
`submit(ToolReply(cid, results))`, `result.settlement`).

Env vars:

| Var | Used by | Default |
|---|---|---|
| `ANY_LLM_TEST_MODEL_PRIMARY` | `any-llm-test.ipynb` | required |
| `ANY_LLM_TEST_MODEL_SECONDARY` | `any-llm-test.ipynb` | optional, skip |
| `ANY_LLM_TEST_MODEL_REASONING` | `any-llm-test.ipynb` | falls back to primary |
| `ANY_LLM_ENABLE_REASONING_TEST` | both | reasoning cells skip unless `1` |
| `ANY_LLM_AGENT_TEST_MODEL_PRIMARY` | `any-llm-agent-test.ipynb` | `openai:gpt-4o-mini` |
| `ANY_LLM_AGENT_TEST_MODEL_REASONING` | `any-llm-agent-test.ipynb` | optional |

Model values are colon-form any-llm ids (e.g. `openai:gpt-4o-mini`,
`anthropic:claude-haiku-4-5`).

Execution-dependency rule (by construction): in both notebooks the Setup
section is a global prerequisite; top-level sections are mutually
independent (each builds its own provider inputs / agent + storage
adapters); cells *within* a section run top-to-bottom.

Deliberately absent from the agent notebook (not supported by design):
server tools, Anthropic Skills, citations, provider-hosted file generation,
API-key rotation.
