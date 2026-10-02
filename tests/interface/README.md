# Interface suite (the living spec)

Behavioural specs for the public surface of `agent_base`. This suite is the
contract in executable form: what it pins is what a consumer may rely on. See
`mental_model/infrastructure/packaging-and-release.md`.

The suite was written test-first, against the design in `interface_plan/`.
`mental_model/` has since replaced that folder. Comments here still cite its
sections and ledger ids (`GF-P8G3`, `O12`, `relay-await §2.4`, …); the files
are in git history (`git log -- interface_plan`).

## Running

The default `pytest` run does **not** collect this tree (`testpaths` in
`pyproject.toml` covers `tests/unit` and `tests/integration` only), so target
it:

```bash
# The whole suite
uv run --all-extras --all-packages pytest tests/interface -q

# One subsystem
uv run --all-extras --all-packages pytest tests/interface/relay_await -q
```

The `mcp/` tests need the `mcp` extra, and the SSE transport tests need
`fastapi` or `starlette`, which the demo workspace member brings. Without them
those files fail at import.

## Layout

One package per subsystem. "Told in" is the file under `mental_model/` that
describes the behaviour the package pins.

| Package | Told in | Owns (deep-tests) |
|---|---|---|
| `tenancy_principal/` | `subsystems/identity.md` | `SessionPrincipal`, `PrincipalPolicy`, `StrictScopePolicy` |
| `storage/` | `subsystems/storage.md` | `ColumnSpec` registry, adapters, `for_principal`, `runs_matching` |
| `agent_loop_hooks/` | `subsystems/hooks-and-profiles.md` | the hook catalog, `HookOutcome`, observer hooks |
| `streaming_and_meta/` | `subsystems/streaming.md` | `MetaEnvelope`/`MetaBody`, `StreamDelta`, wire + decoder |
| `relay_await/` | `features/pause-and-resume.md` | `AwaitTable`, `await_external`, `ResumeOutcome` |
| `session_control/` | `subsystems/session-actor.md` | `SessionManager`, `OpenAwait`, submit/eviction |
| `tools/` | `subsystems/tools.md` | `@tool`, registry, `ToolContext`, `ExecutorPolicy` |
| `mcp/` | `subsystems/mcp.md` | `McpServerSpec`, connect and compile, auth, the result budget |
| `sandbox/` | `subsystems/sandbox.md` | sandbox FS surface, namespacing |
| `media_backend/` | `subsystems/blob-store-and-media.md` | `BlobStore`, `BlobRef`, flush strategies |
| `fork_reset/` | `features/fork-and-reset.md` | `fork_session`, `reset_session`, `CheckpointAdapter`, the checkpoint codec, sandbox snapshots |
| `finalization/` | `features/answer-finalization.md` | `early_answer_completion`, the finalization journal |
| `memory/` | `CLAUDE.md` (characters without a file) | memory stores, `MemoryContribution` |
| `providers/` | `subsystems/providers.md` | `Provider` value, `RetryPolicy`, `make_llm_config` |
| `core/` | `subsystems/turn-loop.md`, `features/run.md` | `AgentRuntime`, `record_turn`, `ErrorCode` |
| `pricing_cost/` | `features/billing-a-run.md` | `TurnSettlement`, `SettlementAggregator`, `cost_for_turn` |
| `python_executors/` | `CLAUDE.md` (characters without a file) | executor interfaces |
| `logging/` | `infrastructure/external-services.md` | structlog config, never-log-claims invariant |

Shared types are deep-tested only in their owning package; elsewhere they are
used as collaborators.

## Conventions

- An interface change updates the matching package here and the mental model
  in the same PR.
- Imports target the canonical homes (e.g. `agent_base.core.identity`,
  `agent_base.streaming.meta`).
- No `xfail`, no `skip`, no `importorskip`, no `try/except ImportError`.
- Async tests are plain `async def` (asyncio_mode=auto).
- File names are unique suite-wide, usually `test_<subsystem>_<topic>.py`.
- A symbol removed from the public surface is never imported here.
