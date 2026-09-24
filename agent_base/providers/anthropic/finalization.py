"""Opt-in durable answer boundary and actor-owned, resumable finalization.

The config journal is written before emitting answer_completed. It contains the
answer row as well as the already-priced settlement, so a crash between the two
adapter writes cannot lose the answer or re-price/re-run the model. No task is
spawned here: the session actor owns both normal work and recovery.
"""
from __future__ import annotations

import copy
import inspect
import posixpath
import re
from datetime import datetime, timezone
from typing import Any

from agent_base.core.config import Conversation
from agent_base.core.cost import TurnSettlement
from agent_base.streaming.meta import AnswerCompleted, FinalizationUpdated, FilesUpdated, RunCompleted

JOURNAL = "pending_finalization"
LIFECYCLE = "answer_lifecycle"


class FinalizationFailed(RuntimeError):
    retriable = True


class WorkspaceStateLost(FinalizationFailed):
    retriable = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pending_paths(agent: Any, response: Any) -> list[str]:
    root = getattr(agent._sandbox, "exports_dir", "")
    if not root or not root.startswith("/"):
        return []
    text = agent._extract_text(response)
    # Markdown destinations and present_files resources preserve spaces/nesting.
    paths = re.findall(r"\]\((?:sandbox:)?([^)]*)\)", text)
    log_text = str(agent.conversation.conversation_log.to_dict())
    paths.extend(re.findall(r"<file_path>(.*?)</file_path>", log_text))
    known = {m.extras.get("export_path") for m in agent.agent_config.media_registry.values()}
    return sorted({p for p in paths if p.startswith(root.rstrip("/") + "/")
                   and posixpath.normpath(p).startswith(root.rstrip("/") + "/")
                   and p[len(root.rstrip("/"))+1:] not in known})



async def _save(agent: Any, journal: dict) -> None:
    """One durable journal first, then the query projection; both precede events."""
    agent.agent_config.updated_at = _now()
    if agent._sandbox is not None:
        agent.agent_config.sandbox_config = agent._sandbox.config
    journal["conversation"] = agent.conversation.to_dict()
    agent.agent_config.extras[JOURNAL] = journal
    await agent.config_adapter.save(agent.agent_config)
    await agent.conversation_adapter.save(agent.conversation)


def _progress(agent: Any, *, status: str = "pending", stage: str, error: str | None = None) -> None:
    state = agent.conversation.extras[LIFECYCLE]
    state.update(status=status, stage=stage, error=error)


def _emit_progress(agent: Any) -> None:
    agent._hook_emit(FinalizationUpdated(finalization=dict(agent.conversation.extras[LIFECYCLE])))


async def finalize_answer(agent: Any, response: Any, stop_reason: str) -> Any:
    conversation = agent.conversation
    answer_at = _now()
    conversation.final_response = response
    conversation.stop_reason = stop_reason
    conversation.total_steps = agent.agent_config.current_step
    conversation.usage = agent._run_cumulative_usage
    conversation.cost = agent._compute_cost()
    conversation.conversation_log.mark_agent_completed(agent.agent_uuid)
    agent.agent_config.conversation_log.mark_agent_completed(agent.agent_uuid)
    conversation.extras[LIFECYCLE] = {
        "answer_completed_at": answer_at, "status": "pending", "stage": "exports",
        "files_ready": False, "pending_paths": _pending_paths(agent, response), "error": None,
    }
    journal = {"run_id": conversation.run_id, "settlement": agent._price_unbilled_fact().to_dict(),
               "sandbox_id": getattr(agent._sandbox, "e2b_sandbox_id", None)}
    if agent.agent_config.title is None and conversation.user_message:
        agent.agent_config.title = agent._derive_title(conversation.user_message)
    # Durable step records are part of the small answer write. They are not
    # retried after the boundary (the legacy run adapter is append-only).
    if agent._run_logs:
        await agent.run_adapter.save_logs(agent.agent_uuid, agent._run_id, agent._run_logs)
    await _save(agent, journal)
    agent._hook_emit(AnswerCompleted(answer_completed_at=answer_at, finalization=dict(conversation.extras[LIFECYCLE])))
    return await _finish(agent, journal)


async def recover_finalization(agent: Any) -> Any:
    journal = agent.agent_config.extras[JOURNAL]
    row = Conversation.from_dict(journal["conversation"])
    stored = await agent.conversation_adapter.load_by_run_id(agent.agent_uuid, row.run_id)
    if stored is not None:
        # Sequence allocation belongs to the row adapter. Journal progress wins.
        row.sequence_number = stored.sequence_number
    agent.conversation = row
    agent._run_id = row.run_id
    agent._run_cumulative_usage = row.usage
    agent._cumulative_cost = row.cost or agent._cumulative_cost
    agent._run_logs = []
    agent._reset_cancellation_state(None)
    agent._hook_emit(AnswerCompleted(answer_completed_at=row.extras[LIFECYCLE]["answer_completed_at"], finalization=dict(row.extras[LIFECYCLE])))
    return await _finish(agent, journal, recovering=True)


async def _finish(agent: Any, journal: dict, *, recovering: bool = False) -> Any:
    conversation = agent.conversation
    try:
        # The durable answer consumed the model usage even if publication or
        # workspace recovery later fails permanently. Settle it exactly once
        # before those steps, using the journal's already-priced identity.
        _progress(agent, stage="usage")
        settlement = TurnSettlement.from_dict(journal["settlement"])
        if not journal.get("usage_done"):
            # Retry the SAME priced identity after an uncertain callback. The
            # consumer's durable dedupe key makes this at-least-once safe.
            for callback in list(agent._usage_report_callbacks):
                outcome = callback(settlement)
                if inspect.isawaitable(outcome):
                    await outcome
            journal["usage_done"] = True
            await _save(agent, journal)
        agent._settled_upto = len(agent._turn_steps)
        agent._restored_settlement = None
        from agent_base.streaming.meta import UsageReport
        agent._hook_emit(UsageReport.of(settlement))

        _progress(agent, stage="exports")
        if recovering:
            # Reuse the last live VM, or restore its durable checkpoint. Never
            # run another model turn while publication/checkpointing is pending.
            agent._sandbox_preparation_pending = True
            await agent.prepare_sandbox(trigger="finalization_recovery")
            if (journal.get("sandbox_id") and not journal.get("checkpoint_done")
                    and journal["sandbox_id"] != getattr(agent._sandbox, "e2b_sandbox_id", None)):
                raise WorkspaceStateLost("Your answer is saved, but its workspace was lost before it could be backed up. Start a new chat to continue; retry cannot reconstruct unpublished files.")
        if not journal.get("exports_done"):
            from agent_base.media_backend.flush import IncrementalBlake3Flush
            from .finalization_media import ConfigExportRegistry
            async def pending_exports(exports):
                root = getattr(agent._sandbox, "exports_dir", "")
                conversation.extras[LIFECYCLE]["pending_paths"] = [
                    posixpath.join(root, item.path) for item in exports
                ]
                conversation.extras[LIFECYCLE]["files_ready"] = not bool(exports)
                _emit_progress(agent)
            strategy = IncrementalBlake3Flush(registry=ConfigExportRegistry(agent), idempotent=True, before_upload=pending_exports)
            generated = await agent.media_backend.flush_exports(agent.agent_uuid, strategy=strategy)
            generated.extend(await agent.provider.collect_api_files(agent))
            for media in generated:
                agent.agent_config.media_registry[media.media_id] = media
            conversation.generated_files = generated
            journal["exports_done"] = True
            conversation.extras[LIFECYCLE]["files_ready"] = True
            _progress(agent, stage="checkpoint")
            await _save(agent, journal)
        if conversation.generated_files:
            agent._hook_emit(FilesUpdated(files=[f.to_dict() for f in conversation.generated_files]))
        _emit_progress(agent)

        if not journal.get("memory_done"):
            if agent.memory_store is not None:
                try:
                    await agent.memory_store.update(agent._memory_hook_context(), conversation.conversation_log, conversation.stop_reason)
                except Exception:
                    agent._logger.warning("memory_update_failed", exc_info=True)
            journal["memory_done"] = True

        if not journal.get("checkpoint_done"):
            _progress(agent, stage="checkpoint")
            # A reset/fork checkpoint must never carry a recovery journal for
            # the original run (and must never replay its billing callback).
            checkpoint_config = copy.deepcopy(agent.agent_config)
            checkpoint_config.extras.pop(JOURNAL, None)
            await agent.capture_checkpoint(config_snapshot=checkpoint_config)
            journal["checkpoint_done"] = True
            await _save(agent, journal)

        agent._finalized_run_id = agent._run_id
        conversation.completed_at = _now()
        _progress(agent, status="complete", stage="complete")
        # Row first, then clear journal. A crash in between replays only the
        # idempotent completion projection, never a model request or charge.
        await agent.conversation_adapter.save(conversation)
        agent.agent_config.extras.pop(JOURNAL, None)
        try:
            await agent.config_adapter.save(agent.agent_config)
        except BaseException:
            agent.agent_config.extras[JOURNAL] = journal
            raise
        _emit_progress(agent)
        result = agent._build_agent_result(conversation.final_response, conversation.stop_reason)
        result.generated_files = conversation.generated_files
        result.settlement = settlement
        from .anthropic_agent import _strip_binary_data
        agent._hook_emit(RunCompleted(
            stop_reason=result.stop_reason, total_steps=result.total_steps,
            generated_files=[f.to_dict() for f in conversation.generated_files] or None,
            cost=settlement.turn_cost.to_dict(), cumulative_usage=conversation.usage.to_dict(),
            conversation_log=_strip_binary_data(result.conversation_log.to_dict()) if agent.stream_meta_history_and_tool_results else None,
        ))
        return result
    except Exception as exc:
        stage = conversation.extras[LIFECYCLE]["stage"]
        message = str(exc) if isinstance(exc, WorkspaceStateLost) else f"Your answer is saved, but {stage} could not finish. Retry to finish preparing this response."
        _progress(agent, status="failed", stage=stage, error=message)
        conversation.extras[LIFECYCLE]["retryable"] = not isinstance(exc, WorkspaceStateLost)
        try:
            await _save(agent, journal)
        except Exception:
            agent._logger.warning("finalization_failure_save_failed", exc_info=True)
        _emit_progress(agent)
        if isinstance(exc, WorkspaceStateLost):
            raise
        raise FinalizationFailed(message) from exc
