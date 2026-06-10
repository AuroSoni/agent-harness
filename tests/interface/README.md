# Interface red suite (TDD)

Behavioral test specs for the redesigned `agent_base` interfaces defined in
`interface_plan/` (15 subsystem docs + `DESIGN_CONTRACT.md`, amended per
`AMENDMENTS.md` — the canonical decision ledger).

These tests are written **before** the implementation exists. They are expected
to fail (mostly with `ImportError`/`AttributeError`) until each interface is
built. That is the point: implementation proceeds test-first against this suite.

## Running

The default `pytest` run does **not** collect this tree (`testpaths` in
`pyproject.toml` covers `tests/unit` and `tests/integration` only), so the
existing suite stays usable while this one is red.

```bash
# Run the red suite (collection errors are expected pre-implementation)
uv run pytest tests/interface --continue-on-collection-errors -q

# Run one subsystem
uv run pytest tests/interface/relay_await --continue-on-collection-errors -q
```

## Layout

One package per subsystem, mirroring `interface_plan/subsystems/*.md`:

| Package | Interface doc | Owns (deep-tests) |
|---|---|---|
| `tenancy_principal/` | `tenancy-principal.md` | `SessionPrincipal`, `PrincipalPolicy`, `StrictScopePolicy` |
| `storage/` | `storage.md` | `ColumnSpec` registry, adapters, `for_principal`, `runs_matching` |
| `agent_loop_hooks/` | `agent-loop-hooks.md` | 12-hook catalog, `HookOutcome`, observer hooks |
| `streaming_and_meta/` | `streaming-and-meta.md` | `MetaEnvelope`/`MetaBody`, `StreamDelta`, wire + decoder |
| `relay_await/` | `relay-await.md` | `AwaitTable`, `await_external`, `ResumeOutcome` |
| `session_control/` | `session-control.md` | `SessionManager`, `OpenAwait`, submit/eviction |
| `tools/` | `tools.md` | `@tool`, registry, `ToolContext`, `ExecutorPolicy` |
| `sandbox/` | `sandbox.md` | sandbox FS surface, namespacing |
| `media_backend/` | `media-backend.md` | `BlobStore`, `BlobRef`, flush strategies |
| `memory/` | `memory.md` | memory stores, `MemoryContribution` |
| `providers/` | `providers.md` | `Provider` value, `RetryPolicy`, `make_llm_config` |
| `core/` | `core.md` | `AgentRuntime`, `record_turn`, `ErrorCode` |
| `pricing_cost/` | `pricing-cost.md` | `TurnSettlement`, `SettlementAggregator`, `cost_for_turn` |
| `python_executors/` | `python-executors.md` | executor interfaces |
| `logging/` | `logging.md` | structlog config, never-log-claims invariant |

Shared types are deep-tested only in their owning package; elsewhere they are
used as collaborators.

## Conventions

- Imports target the **future** canonical homes (e.g.
  `agent_base.core.identity`, `agent_base.streaming.meta`) per the
  "Canonical homes for new symbols" table in `interface_plan/AMENDMENTS.md`.
- No `xfail`, no `skip`, no `importorskip`, no `try/except ImportError` —
  red means red.
- Async tests are plain `async def` (asyncio_mode=auto).
- File names are unique suite-wide: `test_<subsystem>_<topic>.py`.
- Deleted symbols (see AMENDMENTS deletions) must never be imported here.
