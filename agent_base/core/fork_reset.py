"""Fork / reset-to-checkpoint verbs (SPEC §5).

Module-level functions over a ``StorageHandles`` bundle — they work COLD, with no
live runtime. ``fork_session`` starts a new owned session from a past checkpoint
(non-destructive to the source); ``reset_session`` rolls an existing session back
to a past checkpoint, **unconditionally** restoring agent state + sandbox (the
tail is archived, never deleted — reversible). Both operate only at completed turn
boundaries (the boundary a checkpoint was captured at).

Divergence is NOT decided here: the library reset of agent + sandbox is
deterministic and needs no divergence input. The workbook (a Nova concern) is
decided in the backend AROUND these calls via the opaque ``consumer_payload``
(SPEC §F4).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agent_base.core.checkpoint import Checkpoint, CheckpointRef
from agent_base.core.config import LLMConfig
from agent_base.core.messages import Usage
from agent_base.storage.checkpoint_codec import assemble_config_from_checkpoint

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_base.core.config import Conversation
    from agent_base.core.identity import SessionPrincipal
    from agent_base.sandbox.sandbox_types import Sandbox
    from agent_base.storage.handles import StorageHandles


class ForkResetError(Exception):
    """Base for fork/reset verb failures."""


class CheckpointNotFound(ForkResetError):
    """No checkpoint exists at the requested ``(agent_uuid, sequence_number)``."""

    def __init__(self, agent_uuid: str, sequence_number: int) -> None:
        super().__init__(
            f"no checkpoint for agent {agent_uuid!r} at sequence {sequence_number}"
        )
        self.agent_uuid = agent_uuid
        self.sequence_number = sequence_number


class SessionBusy(ForkResetError):
    """A reset was requested on a resident session that could not be evicted
    (a turn is in flight or an await is parked)."""

    def __init__(self, agent_uuid: str) -> None:
        super().__init__(f"session {agent_uuid!r} is busy — cannot reset mid-turn")
        self.agent_uuid = agent_uuid


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(handles: "StorageHandles") -> None:
    if handles.checkpoint is None:
        raise ForkResetError("StorageHandles.checkpoint is not wired (feature off)")


async def _history_through(
    handles: "StorageHandles",
    agent_uuid: str,
    at_sequence: int,
    principal: "SessionPrincipal",
) -> list["Conversation"]:
    """Every non-archived conversation run with ``sequence_number <= at_sequence``."""
    adapter = handles.conversation.for_principal(principal)
    out: list["Conversation"] = []
    offset = 0
    page_size = 200
    while True:
        page = await adapter.load_history(agent_uuid, limit=page_size, offset=offset)
        if not page:
            break
        out.extend(c for c in page if (c.sequence_number or 0) <= at_sequence)
        if len(page) < page_size:
            break
        offset += len(page)
    return out


async def fork_session(
    handles: "StorageHandles",
    *,
    source_uuid: str,
    at_sequence: int,
    new_uuid: str,
    principal: "SessionPrincipal",
    llm_config_class: type[LLMConfig] = LLMConfig,
    copy_history: bool = True,
) -> str:
    """Create a NEW session ``new_uuid`` from ``source_uuid``'s checkpoint at
    ``at_sequence``. Non-destructive to the source. Returns ``new_uuid``.

    The transcript segments + sandbox manifest are pointer-copied (CAS shared by
    reference — a 0-byte copy); copied history rows carry zeroed usage/cost so
    analytics never double-counts forked spend.
    """
    _require(handles)
    cp = await handles.checkpoint.for_principal(principal).load(source_uuid, at_sequence)
    if cp is None:
        raise CheckpointNotFound(source_uuid, at_sequence)

    # 1) agent state: assemble the config, re-stamp identity/ownership, persist.
    cfg = await assemble_config_from_checkpoint(cp, handles.blobs, llm_config_class)
    cfg.agent_uuid = new_uuid
    cfg.parent_agent_uuid = None        # a fork is a new ROOT, not a sub-agent
    cfg.pending_relay = None            # the boundary is quiescent
    cfg.owner_tenant = principal.tenant
    cfg.owner_subject = principal.subject
    cfg.extras = dict(cfg.extras)
    cfg.extras["forked_from"] = source_uuid
    cfg.extras["forked_at_seq"] = at_sequence
    await handles.config.for_principal(principal).save(cfg)

    # 2) history: copy runs <= at_sequence, usage/cost zeroed (no double-count).
    if copy_history and handles.conversation is not None:
        conv_adapter = handles.conversation.for_principal(principal)
        for conv in await _history_through(handles, source_uuid, at_sequence, principal):
            conv.agent_uuid = new_uuid
            conv.usage = Usage()
            conv.cost = None
            conv.archived = False
            conv.extras = dict(conv.extras)
            conv.extras["forked_from_run_id"] = conv.run_id
            await conv_adapter.save(conv)       # explicit sequence_number preserved

    # 3) seed checkpoint: pointer-copy the CAS refs forward (0-byte copy) under
    #    the forked config base. The consumer_payload rides along so the consumer
    #    can open the turn-N artifact (e.g. Nova's workbook).
    seed_base = dict(cp.config_base)
    seed_base["agent_uuid"] = new_uuid
    seed_base["parent_agent_uuid"] = None
    seed_base["owner_tenant"] = principal.tenant
    seed_base["owner_subject"] = principal.subject
    seed = Checkpoint(
        ref=CheckpointRef(
            agent_uuid=new_uuid,
            sequence_number=at_sequence,
            run_id=cp.ref.run_id,
            created_at=_now_iso(),
            fidelity=cp.ref.fidelity,
        ),
        config_base=seed_base,
        transcript_segments=list(cp.transcript_segments),
        log_segments=list(cp.log_segments),
        transcript_codec_v=cp.transcript_codec_v,
        sandbox_manifest_ref=cp.sandbox_manifest_ref,
        consumer_payload=dict(cp.consumer_payload),
    )
    await handles.checkpoint.for_principal(principal).save(seed)
    return new_uuid


async def reset_session(
    handles: "StorageHandles",
    *,
    agent_uuid: str,
    to_sequence: int,
    principal: "SessionPrincipal",
    llm_config_class: type[LLMConfig] = LLMConfig,
    sandbox_factory: "Callable[[str], Sandbox] | None" = None,
    sessions: object | None = None,
) -> CheckpointRef:
    """Reset ``agent_uuid`` back to ``to_sequence``. UNCONDITIONAL for agent +
    sandbox. Non-destructive: the conversation + checkpoint tail is ARCHIVED, not
    deleted (so "undo the reset" is a re-point). Returns the restored
    ``CheckpointRef``.

    If ``sessions`` (a SessionManager) is given and the session is resident, it is
    evicted first; an in-flight turn refuses with ``SessionBusy``. If a
    ``sandbox_factory`` is given and the checkpoint has a workspace snapshot, the
    sandbox is materialized back to it; without a factory, agent state is restored
    and the workspace is left untouched.
    """
    _require(handles)

    # 0) quiesce — eviction refuses while a turn is in flight or an await parks.
    if sessions is not None and getattr(sessions, "is_resident", None) is not None:
        if sessions.is_resident(agent_uuid):
            if not await sessions.evict(agent_uuid):
                raise SessionBusy(agent_uuid)

    cp = await handles.checkpoint.for_principal(principal).load(agent_uuid, to_sequence)
    if cp is None:
        raise CheckpointNotFound(agent_uuid, to_sequence)

    # 1) archive the tail (reversible) on BOTH planes — never a delete.
    if handles.conversation is not None:
        await handles.conversation.for_principal(principal).archive_after(
            agent_uuid, to_sequence
        )
    await handles.checkpoint.for_principal(principal).archive_after(agent_uuid, to_sequence)

    # 2) restore agent state in place (latest-wins config row == the checkpoint).
    cfg = await assemble_config_from_checkpoint(cp, handles.blobs, llm_config_class)
    cfg.pending_relay = None
    await handles.config.for_principal(principal).save(cfg)

    # 3) workspace: materialize the sandbox from the checkpoint manifest.
    if cp.sandbox_manifest_ref and handles.blobs is not None and sandbox_factory is not None:
        from agent_base.sandbox.snapshot import SandboxSnapshotter

        tenant = cfg.owner_tenant or "_"
        sandbox = sandbox_factory(agent_uuid)
        await SandboxSnapshotter(
            sandbox, handles.blobs, tenant=tenant
        ).materialize(cp.sandbox_manifest_ref)

    return cp.ref


__all__ = [
    "ForkResetError",
    "CheckpointNotFound",
    "SessionBusy",
    "fork_session",
    "reset_session",
]
