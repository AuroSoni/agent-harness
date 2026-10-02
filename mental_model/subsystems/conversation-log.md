# Conversation log

The record of a run as a person would read it: every message, every tool result with its rich detail, every retry, and timing for the things that are slow. It is what a client renders as history and what the host analyses afterwards.

It is separate from the **context** (`agent_config.context_messages`), which is what the model is sent. The two start from the same messages and then diverge: the context is compacted, externalized and repaired; the log keeps the originals.

| | Context | Conversation log |
|---|---|---|
| Reader | The model | People, clients, analytics |
| Scope | The whole session, carried across runs | One run |
| Edited by | Compaction, externalization, chain repair | Nothing; append-only |
| Tool results | The blocks the model sees | A projection with summary, details and timing |
| Stored in | `agent_config.context_messages` | `conversation_history.conversation_log`, one row per run |

The log's shape is part of the wire protocol, whose contract is marked in [Streaming](streaming.md): the same JSON travels in `run_completed.conversation_log` and is what the host serves as history.

## One run's record

The class `Conversation` (`core/config.py`) is one run's row, not a whole conversation. A session's history is its rows in order.

| Field | Meaning | Set |
|---|---|---|
| `agent_uuid`, `run_id` | The session and the run | Start |
| `sequence_number` | The run's position in the session; assigned by storage | First save |
| `started_at`, `user_message` | When, and the prompt as written | Start |
| `conversation_log` | The log below | Throughout |
| `final_response` | The last assistant message | End. Absent on an aborted run |
| `stop_reason` | `end_turn`, `max_tokens`, `max_steps`, `refusal`, `context_window_exceeded`, `aborted`, `error` | End |
| `total_steps` | The agent's own steps in the run | End |
| `usage`, `cost` | Run totals, sub-agents included | End |
| `generated_files` | Files the run produced | End |
| `completed_at` | Close time | End |
| `archived` | Hidden by a [reset](../features/fork-and-reset.md) | Reset |
| `extras` | `error`, `persist_errors`, `answer_lifecycle`, `forked_from_run_id` | As they occur |

## The log

```json
{
  "agents": { "<agent uuid>": { "agent_uuid": "…", "parent_agent_uuid": null, "name": "…", "description": "…", "model": "…", "provider": "…", "completed": true } },
  "entries": [ ],
  "spans": [ ],
  "_v": 1
}
```

`agents` describes every agent that wrote to the log. `spans` is present only when there are some.

**Entries**, in order of occurrence. Each has `entry_type`, `agent_uuid`, `timestamp` and `_v`.

| `entry_type` | Other keys | Written when |
|---|---|---|
| `message` | `role`, `content`, `attachments`, `contributions`, `stop_reason`, `usage`, `provider`, `model`; on a model response also `timing`, `cost_usd`, `step` | A user message is added, or a provider call completes |
| `tool_result` | `tool` (below) | A backend tool call finishes, in a step that does not pause |
| `rollback` | `message`, `code`, `details`, `targets_previous_assistant_message` | The previous answer was rejected |
| `stream_event` | `stream_type`, `payload` | The older [`end_turn_hook`](hooks-and-profiles.md#an-older-mechanism) marked one of its events to be kept |

In a step that [pauses](../features/pause-and-resume.md), the step's backend results and the client's reply are logged together as one user `message` entry whose content is tool-result blocks. They get no `tool_result` entries.

A response's metadata:

- `usage`: `input_tokens`, `output_tokens`, `cache_write_tokens`, `cache_read_tokens`, `thinking_tokens`, `raw_usage`.
- `timing`: `started_at`, `ended_at`, `flight_ms` of the provider call.
- `cost_usd`: omitted when the model has no price row.
- `step`: 1-based.

**A tool result** in the log (`ToolLogProjection`) is richer than what the model saw:

```json
{
  "tool_name": "…", "tool_id": "toolu_…", "is_error": false,
  "summary": "…",
  "content_blocks": [ ],
  "details": { },
  "duration_ms": 12.0, "started_at": "…", "ended_at": "…", "queued_ms": 0.0,
  "executor": "backend",
  "nested_conversation": null
}
```

- A [tool](tools.md) chooses what goes to the model and what goes to the log separately.
- For a [sub-agent](sub-agents.md) call, `nested_conversation` is the child's whole log and `details` names the child and its outcome.

**Content blocks** carry a `content_block_type`: `text`, `thinking`, `image`, `document`, `tool_use`, `server_tool_use`, `mcp_tool_use`, `tool_result`, `server_tool_result`, `mcp_tool_result`, `attachment`, `citation`, `error`.

## Trace spans

Timing for what a successful step does not already time. Each span has `kind`, `v` and `agent_uuid`.

| `kind` | Records |
|---|---|
| `sandbox_ready` | One sandbox warm: `trigger`, window, `ok`, and detail reported by the [sandbox coordinator](sandbox.md) |
| `relay` | One [pause](../features/pause-and-resume.md): `cid`, `reason`, the calls, `paused_at`, `await_emitted_at`, `resumed_at`, `spliced_at`, `outcome` |
| `model_call_failed`, `model_call_cancelled` | A provider call that did not produce a step |
| `turn_error` | The error that ended a run |

- A successful provider call has no span; its timing is on the response's entry. A tool call's timing is on its projection.
- Spans recorded before a run exists (a warm on session load) are buffered and adopted by the next run. Stale ones are dropped.
- Recording a span never fails a run: an error while tracing is logged once and ignored.

## Where it travels

| Place | Form |
|---|---|
| `run_completed.conversation_log` | The run's log, spans included, binary data replaced by a size label. Only with `stream_meta_history_and_tool_results` |
| `conversation_history.conversation_log` | The run's log, as stored |
| `agent_checkpoints` | Entries stored as content-addressed segments. See [Fork and reset](../features/fork-and-reset.md) |

## Versioning

- The log and every entry carry `_v` (currently 1). Readers do not branch on it yet.
- Rows written before typed entries existed hold a bare list of messages. They load as `message` entries, with the original dict kept in `legacy_message` so nothing is lost on the next save.

## Contracts

- The JSON shapes above: the log, its four entry types, the tool projection, the span kinds.
- `Conversation`, `ConversationLog`, `ToolLogProjection`, and the content block classes in `core/types.py`.
