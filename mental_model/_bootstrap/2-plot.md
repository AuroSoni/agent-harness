# Checkpoint 2: the plot

For Auro. Checkpoint 1 settled the characters. This one settles why they are the way they are. It has three parts:

- **Part A: 40 reasons found written down in the repo.** Each was checked against today's code and reworded as the why I would write. Strike the wrong ones, or reword them; every row you leave alone counts as confirmed.
- **Part B: 13 questions**, guess first, about choices that look unnecessary, competing patterns, and history.
- **Part C:** what I dropped, and facts I will state without a reason.

**How to answer:** by number, in chat or on the PR. For Part A, "S4 wrong: …" or "strike S12" is enough. For Part B, "confirm", "correct: …" or "leave it out".

**What unblocks the pilot:** Part A1 and questions Q1 to Q4. The pilot is `subsystems/session-actor.md` and `features/pause-and-resume.md`.

Names follow Checkpoint 1: a **run** is the agent's work on one prompt; a **step** is one provider call; a **relay** is a frontend-tool pause; an **await** is its `AwaitTable` record.

---

## Part A: reasons to strike or confirm

"From" is where the reason is written. Ledger ids refer to `interface_plan/AMENDMENTS.md`.

### A1. Session actor, and pause and resume (needed for the pilot)

| # | The why, as I would write it | From |
|---|---|---|
| S1 | **One writer per session.** Before the redesign, abort, steer, tool results and new messages arrived as racing method calls on shared state with no per-session lock, and the live agent was rebuilt from storage on every request. | `NEW_CONSOLIDATEED_ARCHITECTURE.md:28-30` |
| S2 | **The redesign's seams come from a real consumer.** Each one traces to a place where the Nova backend had to rebuild or reach inside the library. | `interface_plan/README.md:3` |
| S3 | **`submit` never waits for the run, and `Ack` carries no completion future.** `wait_idle()` is the one way to wait for a run to finish: one pattern, not two. | Ledger GF-P6G4 |
| S4 | **`MISDIRECTED` is in `Disposition` today with nothing producing it,** because adding a member to a public enum later would be a breaking change. | `interface_plan/subsystems/session-control.md:197-199`; ledger O4 |
| S5 | **The session-attach check always runs.** A missing claimant is treated as anonymous, because skipping the check when no principal is given is an auth bypass by omission. | Ledger M5 |
| S6 | **The runtime resolves a `ToolReply` as the owner,** presenting its own principal. Without that, a runtime with a named principal was an anonymous claimant against its own record: every reply was rejected and the pause parked forever. | `agent_base/core/runtime.py:1572-1582`; ledger GF-P8G3 |
| S7 | **The generation decides whether a reply still counts.** Once an abort or steer retires a generation, a late reply for it returns `IGNORED_STALE` and never wakes the run. | `agent_base/await_table/table.py:9-11` |
| S8 | **Scripted pauses queue behind one lock per runtime,** because the client holds one pending relay slot per agent and the HTTP transport stops streaming at the first `await_input`. | `agent_base/core/runtime.py:357-361` (WT-3) |
| S9 | **A relay resume takes no fork/reset checkpoint.** Each one re-encoded the transcript, snapshotted the sandbox and wrote the checkpoint row, only for the end of the run to rewrite it. | Ledger RP-1; commit `a4f9d12` |
| S10 | **`current_step` only ever grows within a run.** Resume, re-arm and steer never reset it, because consumers key idempotent billing on `(run_id, agent_id, step_count)`; a reset makes two legs indistinguishable and silently drops charges. | `agent_base/core/config.py:254-262` |
| S11 | **A forceful steer marks the stream `Custom('steered')`, not `Custom('aborted')`.** Consumers close their stream on `aborted`, and the steered run's frames were then dropped. | Ledger NV-4 |
| S12 | **`invalidate_idle` discards a resident session without aborting, running end hooks, checkpointing or pausing the sandbox,** because those writes could overwrite newer state owned by another backend process. | `agent_base/session/manager.py:599-604` |

### A2. The rest

**Turn loop and providers**

| # | The why, as I would write it | From |
|---|---|---|
| S13 | **A provider is a value injected into the runtime, not a base class.** A shared mixin would have fixed the duplication but not the shape of the problem, provider-as-subclass. | `agent_base/core/provider.py:3-8`; `interface_plan/subsystems/providers.md:528` |
| S14 | **Moving the loop into `AgentRuntime` was sequenced last,** so that everything else could be written against "the runtime" and the move is a relocation, not a rewrite. | `agent_base/core/runtime.py:2107-2119` |
| S15 | **Consecutive user messages are not merged for Anthropic.** The API combines them itself, and a merged message is a history edit that invalidates the prompt cache and every later thinking block. | `agent_base/providers/anthropic/provider.py:434-436` |
| S16 | **`before_tool` hooks work on a deep copy of the tool input.** The model's `tool_use` block in the history keeps exactly what it sent, because editing history breaks the cache and preserved thinking. | `agent_base/core/runtime.py:1435-1439`; commit `85aedbb` |
| S17 | **The thinking paradigm is picked by which config field the caller set, never by model name,** which keeps the provider model-agnostic. | Ledger AT-1 |
| S18 | **Breaking changes are allowed and compatibility shims are deleted,** because the library is preview and unreleased. | Ledger G0; `CLAUDE.md` |

**Streaming**

| # | The why, as I would write it | From |
|---|---|---|
| S19 | **The encoder and the decoder ship from one module,** so the protocol cannot drift between the two ends. | `agent_base/streaming/wire.py:3-6` |
| S20 | **`Custom` meta bodies are open by construction:** the library knows nothing of consumer events, and they are still correlated like every other envelope. | `interface_plan/subsystems/streaming-and-meta.md:714` |
| S21 | **`Rollback` is a meta body, not a content delta,** so the content channel stays exactly "the model's own output". | `interface_plan/subsystems/streaming-and-meta.md:708` |
| S22 | **The keepalive is a `data:` frame, not an SSE comment.** Comments never fire the client's `onmessage`, so app-level idle watchdogs would still abort. | Ledger SSE-1a |

**Storage and checkpoints**

| # | The why, as I would write it | From |
|---|---|---|
| S23 | **Columns that are replayed to the model are `JSON`, not `JSONB`.** JSONB re-sorts keys and rewrites some numbers, so a reloaded session would send different bytes: a prompt-cache miss, and invalidated thinking blocks. | `agent_base/storage/pg/row_mappers.py:139-142`; commit `4d6a374` |
| S24 | **Postgres SQL is composed from declared columns,** so every scoped column lands in every WHERE and a missed tenant predicate is structurally impossible. | `agent_base/storage/pg/__init__.py:6-9` |
| S25 | **Adding a column to a library table must bump `LIBRARY_SCHEMA_VERSION` and ship a migration.** `CREATE TABLE IF NOT EXISTS` does nothing on a database already stamped, and a column once went silently missing on a staging database. | Ledger GF-SCHEMA4 |
| S26 | **A checkpoint stores the transcript as content-addressed segments.** A full serialized config per run is O(n²); with segments, an unchanged prefix costs nothing and a fork is a pointer copy. | Ledger FR-1 |
| S27 | **A checkpoint captures the config and the sandbox together,** because the sandbox filesystem is the only runtime state that cannot be rebuilt from `AgentConfig`. | Ledger FR-2 |
| S28 | **Checkpoint blob keys are scoped by tenant.** A bare content hash would let two tenants share a blob. | `agent_base/storage/checkpoint_codec.py:23-25` |
| S29 | **Transcript segments keep each message's own key order; log segments are sorted.** The transcript is replayed to the model, so re-sorting it misses the prompt cache; the log is never replayed. | `agent_base/storage/checkpoint_codec.py:15-21` |
| S30 | **Reset archives the tail instead of deleting it,** so undoing a reset is a re-point, not a recovery. | `agent_base/storage/base.py:363-366` |
| S31 | **A checkpoint's `consumer_payload` is never overwritten by a re-save.** The library saves the same checkpoint on many paths, and a mutable upsert kept clobbering what the consumer had reconciled into it. | Ledger FR-8 |
| S32 | **Sandbox snapshots are a content manifest, not a git-like repo per session.** A repo adds a binary dependency and leaks `.git` into the agent's own workspace. | `fork_reset_design/SPEC.md:59` |

**Cost and finalization**

| # | The why, as I would write it | From |
|---|---|---|
| S33 | **An errored run is persisted and never billed.** The settlement watermark passes its steps. Only the root agent's own spend is written off; a sub-agent that finished before the error stays billed. | Ledger TR-6; commit `8b7607b` |
| S34 | **Answer finalization journals the answer and its already-priced settlement in the config before it emits `answer_completed`,** so a crash between the two adapter writes cannot lose the answer or re-price or re-run the model. | `agent_base/providers/anthropic/finalization.py:1-6` |

**Sandbox**

| # | The why, as I would write it | From |
|---|---|---|
| S35 | **`SandboxCoordinator` is a Protocol the host implements.** The library has no database there; the host owns binding changes, fencing and locks. | `agent_base/sandbox/coordinator.py:1-5` |
| S36 | **Each E2B command runs under `setsid` with a process tag.** E2B starts every command inside its own daemon's process group, so the pid it reports cannot be group-killed. | `agent_base/sandbox/e2b.py:471-476` |

**Tools, sub-agents, MCP**

| # | The why, as I would write it | From |
|---|---|---|
| S37 | **Relay is an execution mode, not a hook family.** A tool with `executor="frontend"` goes through the same `before_tool`, `after_tool` and `on_tool_error` as a backend tool. | `interface_plan/subsystems/agent-loop-hooks.md:541-542` |
| S38 | **`SubAgentSpec` copies data fields and keeps runtime resources by reference.** Deep-copying a pool-backed object raises, and a connection pool is a singleton, not a value. | `agent_base/common_tools/sub_agent_tool.py:28-33`; ledger GF-P8G1 |
| S39 | **MCP servers get their own `mcp_servers=` argument, not a tool bundle.** Bundles expand synchronously at registration and MCP discovery is async. | Ledger MC-D3 |
| S40 | **There is no in-place `unregister_tools()`.** It would put mutation semantics (unknown names, in-flight calls, partial failure) into the most-consumed subsystem's contract for good. | `interface_plan/subsystems/mcp.md:647` (MC-D12) |

---

## Part B: questions

Highest impact first. Q1 to Q4 touch the pilot.

**Q1. The Rung 2 plumbing that nothing uses today.**
These exist and have no caller or no effect: `CommandMeta` (`command_id`, `client_seq`), `Disposition.MISDIRECTED`, `Target` (one member), `Abort.grace_ms`, `ToolContext.idempotency_key` / `attempt` / `replay_reason` / `once()`, the `from_seq` parameter the stream reserves, `set_await_table`, and the `checkpoint()` seam. A developer or agent tidying up will delete them.
Guess: they are kept on purpose. The command and cid protocol is final at Rung 1 so that a cross-process tier (shared session store, stream replay) changes no consumer call. And Rungs 2 to 4 are "gated, not scheduled": climbed only when a metric forces it.
Evidence: "idempotency + ordering fields are plumbed in Rung 1, enforced at Rung 2" and "Rungs 2–4 are gated, not scheduled — climb only when a metric forces it" (`NEW_CONSOLIDATEED_ARCHITECTURE.md:1127`, `:198`).
→ Confirm / correct. Is "gated, not scheduled" still your position? Is anything on that list plain leftover?

**Q2. One live stream reader, and frames with no reader are dropped.**
`attach_stream` takes the stream from the previous reader, hands over only the undelivered tail, and never replays. After `detach_stream`, frames are dropped. The only written reason is a label ("lossy by policy").
Guess: at Rung 1 the run is the record: it carries on to its end whether or not anyone is reading, and the conversation log is persisted. The stream is a live view of it, and replay waits for Rung 2.
→ Confirm / give the reason / leave the why out.

**Q3. `PrincipalPolicy` has one implementation, and the injected one does not reach the await table.**
`SessionManager` takes `principal_policy` and uses it for the attach check. `AwaitTable.resolve` accepts a policy that no caller passes, and `cancel` constructs `StrictScopePolicy()` itself. The ledger (I1) says both checks consult the one injected policy.
Guess: the seam is deliberate, for hosts that need a looser rule than "same tenant and subject". The missing forwarding is a bug, and goes to the PR as a follow-up.
→ Confirm / "leftover, don't document".

**Q4. Four pairs where two mechanisms do one job. Which is canon?**

| Job | Mechanism A | Mechanism B | Guess |
|---|---|---|---|
| React at the end of a run | `on_turn_end` in the hook catalog | `end_turn_hook=` constructor argument, with its own `EndTurnContext` class | A is canon. B is pre-redesign and survives because the consumer still passes it. |
| React to tool results | `after_tool` hook | `_on_tool_results` subclass override | A is canon; B is leftover. |
| Abort or steer a run | `submit(Abort())`, `submit(Steer(...))` | `agent.abort()`, `agent.steer()` | A is canon: one front door. B is leftover. |
| Give a tool its sandbox, run context, parent or cancellation | `ToolContext` | Setters found by name on the tool instance (`set_sandbox`, `set_run_context`, `set_parent_context`, `set_cancellation_event`) | A is canon. B is what `SubAgentTool` and class-style tools still use. |

→ Confirm each row / correct. For the drift rows: document the canon only and list the other as a follow-up?

**Q5. Two ways a large tool result is moved out of the context.**
A tool can cap its own output with `ctx.emit_capped` or `ctx.spill`, which writes the overflow to `.tool_results/` (default 25,000 characters). Separately, after the hooks, `ContextExternalizer` replaces any result block over 25,000 tokens with a reference to a file in `.context/`.
Guess: both are canon and they are two layers. The first is the tool author's budget; the second is the automatic backstop for tools that set none.
→ Confirm / one of them is drift.

**Q6. Two costs for the same run.**
`Conversation.cost` is accumulated per step by calling `calculate_step_cost` directly. The settlement is priced through `pricing_policy`. A custom pricing policy therefore changes what is billed and not what the row shows.
Guess: the settlement is the billing truth. `Conversation.cost` is a record of spend for display and analytics, and its bypass of the policy is drift.
→ Confirm / correct.

**Q7. LiteLLM.**
`LiteLLMAgent` subclasses `AnthropicAgent`. Its provider stores `retry_policy` and never reads it. Its constructor has no `principal`, `mcp_servers`, `checkpoint_adapter`, `sandbox_coordinator` or `pricing_policy`. The pricing CSV has only `claude-*` rows, so an unknown model costs zero.
Guess: Anthropic is the only production path. LiteLLM is partial, and parity waits on the loop lift. The model says that in a short paragraph and lists the gaps.
→ Confirm / correct. Should the model call it "supported, partial" or "experimental"?

**Q8. What exactly is the `agent-base` package contract?**
The product map says: the public Python surface, plus the library-owned tables. The backend also reads about twenty private names (`agent._sandbox`, `_phase`, `_run_id`, `sandbox._transport`, …).
Guess: the contract is what `tests/interface/` pins, plus the four tables and `LIBRARY_SCHEMA_VERSION`. A private name a consumer reads is not contract; it marks a gap in the public surface. This is the same guess the backend run put to you from its side.
→ Confirm / correct.

**Q9. Knowledge of Nova inside the library.**
Three places: the checkpoint codec strips a `_nova_lifecycle` key from `sandbox_config`; the E2B secret guard lists a `STYTCH_` prefix; a snapshot comment names `/home/nova`.
Guess: drift. The library should know nothing of its consumer. Not in the model; listed as follow-ups.
→ Confirm / correct.

**Q10. Running model code in the library's own process.**
`CodeExecutionTool` uses `python_executors` (an AST interpreter) inside the host process and changes the process's working directory. In the backend run you called Nova's code execution tools dead.
Guess: legacy in the library too. The cast line says model code belongs in the sandbox, and the tool is listed as a removal follow-up.
→ Confirm / "kept for hosts without a sandbox".

**Q11. `ZoneLayout` and the common tools.**
The library's sandbox has a zone layout (`workspace/`, `.exports/`, `.plans/`, `.context/`, `.tool_results/`), and its built-in tools (read, grep, glob, list, apply-patch, todos) work inside it. In September the sandbox gained absolute in-VM trees (`open_roots`, capture roots). Nova has moved to an absolute layout with its own tools, and you called its zone-layout rosters dead.
Guess: in the library, the zone layout is still the default and the common tools are a generic starter set; absolute trees are an addition for hosts that need them. Both are documented.
→ Confirm / "zone layout and common tools are legacy here too".

**Q12. Which history earns a place?**
The rule is history only where it explains the present.
Guess: one plot point is kept, the June 2026 redesign (S1, S2), told in the session actor's file. The February move from an XML stream to typed JSON frames and the move from `anthropic_agent` to `agent_base` are left to git.
→ Confirm / add one.

**Q13. Three names.**
- The loop inside a run: `CLAUDE.md` calls it "the turn loop". With "run" as the unit, guess: keep **turn loop** for the loop, since it is the established term.
- The library class `Conversation` is one run's record, while your "conversation" is everything said in a session. Guess: the cast says so in one line, and the model writes `Conversation` (in code font) only for the class.
- The table `agent_runs` holds log entries, not runs. Guess: one cast line, same treatment.
→ Confirm / correct.

---

## Part C: dropped, and stated without a reason

**Reasons found and not carried forward**

| Candidate | Why it is dropped |
|---|---|
| Why the memory subsystem is small (`interface_plan/subsystems/memory.md:35-36`) | Memory gets a cast line only. |
| Why the executor interface is synchronous (`python-executors.md:325`) | Python executors get a cast line only; see Q10. |
| Why `SettlementAggregator` was deleted (ledger I9) | It explains something that no longer exists. |
| Why there is no executor registry (`python-executors.md:228`) | Same as above. |

**Facts I will state without a why, unless you give one**

- An aborted run's stream ends with `Custom('aborted')`; a completed or errored one ends with `RunCompleted`.
- A checkpoint is taken only at a run boundary, never while a relay is pending.
- A late or duplicate `ToolReply` is a harmless no-op (`IGNORED_STALE`, `IGNORED_DUP`).
- A sub-agent shares its parent's stream, sandbox, storage and cancellation.
- A dead MCP server fails the tool call, never the run.
- Limits: mailbox capacity 32; 128 resident sessions; idle time-to-live 900 s; frames chunked at 2,048 bytes; keepalive after 15 s; default `max_tokens` 16,384.

**Found on the way, for the PR's follow-up list (not questions)**

- Several docstrings and `CLAUDE.md` say the loop is in `AgentRuntime`.
- `AnthropicLLMConfig.enable_caching` is never read; caching is always on.
- LiteLLM's `retry_policy` is never read.
- `interface_plan/subsystems/mcp.md` says MCP images go to the media backend; the code builds inline base64.
- `README.md` describes XML formatters, a Docker sandbox and a file tree that no longer exist.
