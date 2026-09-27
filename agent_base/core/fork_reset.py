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

import inspect
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from types import SimpleNamespace

from agent_base.sandbox.coordinator import uncoordinated

from agent_base.core.checkpoint import Checkpoint, CheckpointRef
from agent_base.core.config import LLMConfig
from agent_base.core.messages import Usage
from agent_base.storage.checkpoint_codec import assemble_config_from_checkpoint

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_base.core.config import Conversation
    from agent_base.core.identity import SessionPrincipal
    from agent_base.sandbox.sandbox_types import Sandbox
    from agent_base.sandbox.coordinator import SandboxCoordinator
    from agent_base.sandbox.snapshot import SnapshotPolicy
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


async def _coordination_context(handles, agent_uuid, principal, sandbox_factory=None, snapshot_policy=None):
    adapter = handles.config.for_principal(principal)
    return SimpleNamespace(
        agent_uuid=agent_uuid, principal=principal, agent_config=await adapter.load(agent_uuid),
        config_adapter=adapter,
        checkpoint_adapter=handles.checkpoint.for_principal(principal) if handles.checkpoint else None,
        _blobs=handles.blobs, _sandbox=None, _sandbox_factory=sandbox_factory,
        _snapshot_policy=snapshot_policy, _last_sandbox_manifest=None, sandbox_warnings=[],
    )


async def fork_session(
    handles: "StorageHandles", *, source_uuid: str, at_sequence: int,
    new_uuid: str, principal: "SessionPrincipal",
    llm_config_class: type[LLMConfig] = LLMConfig, copy_history: bool = True,
    sandbox_coordinator: "SandboxCoordinator | None" = None,
    snapshot_policy: "SnapshotPolicy | None" = None,
) -> str:
    """Fork an immutable checkpoint while reserving the destination session."""
    context = await _coordination_context(handles, new_uuid, principal, snapshot_policy=snapshot_policy)
    guard = sandbox_coordinator.exclusive(context, reason="fork") if sandbox_coordinator else uncoordinated()
    async with guard:
        return await _fork_session(
            handles, source_uuid=source_uuid, at_sequence=at_sequence, new_uuid=new_uuid,
            principal=principal, llm_config_class=llm_config_class, copy_history=copy_history,
        )


async def _fork_session(
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
    # A fork gets its OWN sandbox: scrub the source's binding (a remote id
    # would otherwise be shared by two sessions). The fork's first initialize
    # provisions one and rehydrates it from the seed checkpoint's manifest.
    cfg.sandbox_config = None
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
    seed_base["sandbox_config"] = None
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
    handles: "StorageHandles", *, agent_uuid: str, to_sequence: int,
    principal: "SessionPrincipal", llm_config_class: type[LLMConfig] = LLMConfig,
    sandbox_factory: "Callable[[str], Sandbox] | None" = None,
    sessions: object | None = None,
    sandbox_coordinator: "SandboxCoordinator | None" = None,
    snapshot_policy: "SnapshotPolicy | None" = None,
) -> CheckpointRef:
    """Reset state and files under exclusive session activity when coordinated."""
    context = await _coordination_context(handles, agent_uuid, principal, sandbox_factory, snapshot_policy)
    guard = sandbox_coordinator.exclusive(context, reason="reset") if sandbox_coordinator else uncoordinated()
    async with guard:
        return await _reset_session(
            handles, agent_uuid=agent_uuid, to_sequence=to_sequence, principal=principal,
            llm_config_class=llm_config_class, sandbox_factory=sandbox_factory, sessions=sessions,
            sandbox_coordinator=sandbox_coordinator, snapshot_policy=snapshot_policy,
            coordination_context=context,
        )


async def _reset_session(
    handles: "StorageHandles",
    *,
    agent_uuid: str,
    to_sequence: int,
    principal: "SessionPrincipal",
    llm_config_class: type[LLMConfig] = LLMConfig,
    sandbox_factory: "Callable[[str], Sandbox] | None" = None,
    sessions: object | None = None,
    sandbox_coordinator=None, snapshot_policy=None, coordination_context=None,
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

    current = await handles.config.for_principal(principal).load(agent_uuid)
    cfg = await assemble_config_from_checkpoint(cp, handles.blobs, llm_config_class)
    cfg.pending_relay = None
    if current is not None and current.sandbox_config is not None:
        cfg.sandbox_config = current.sandbox_config

    # Prepare/validate replacement files before changing the transcript/history.
    # A coordinator reserves durable pending recovery and prepares a replacement.
    # It commits readiness only after all transcript/history persistence succeeds.
    sandbox = await _sandbox_for_reset(cfg, agent_uuid, sandbox_factory, setup=False)
    coordinated_remote_reset = (
        sandbox_coordinator is not None and sandbox is not None and sandbox.is_remote
    )
    if coordinated_remote_reset:
        coordination_context.agent_config = current or cfg
        coordination_context._sandbox = sandbox
        sandbox = await sandbox_coordinator.reset(
            coordination_context, manifest_ref=cp.sandbox_manifest_ref,
        )
        cfg.sandbox_config = sandbox.config
    elif cp.sandbox_manifest_ref and handles.blobs is not None and sandbox is not None:
        from agent_base.sandbox.snapshot import SandboxSnapshotter
        if sandbox_coordinator is None:
            sandbox = await _sandbox_for_reset(cfg, agent_uuid, sandbox_factory)
        await SandboxSnapshotter(
            sandbox, handles.blobs, tenant=cfg.owner_tenant or "_", policy=snapshot_policy,
        ).materialize(cp.sandbox_manifest_ref)
        cfg.sandbox_config = sandbox.config

    config_adapter = handles.config.for_principal(principal)
    save_reset = getattr(config_adapter, "save_reset", None)
    # Consumers may merge metadata on ordinary saves while reset must replace
    # it from the checkpoint. Existing adapters keep their normal save behavior.
    if callable(save_reset):
        await save_reset(cfg)
    else:
        await config_adapter.save(cfg)
    if handles.conversation is not None:
        await handles.conversation.for_principal(principal).archive_after(agent_uuid, to_sequence)
    await handles.checkpoint.for_principal(principal).archive_after(agent_uuid, to_sequence)

    if coordinated_remote_reset:
        coordination_context.agent_config = cfg
        coordination_context._sandbox = sandbox
        await sandbox_coordinator.finish_reset(coordination_context)

    return cp.ref


async def _sandbox_for_reset(cfg, agent_uuid: str, sandbox_factory, *, setup: bool = True) -> "Sandbox | None":
    """The sandbox a reset restores into: the persisted binding first (it
    carries a remote id), else the consumer factory (sync or awaitable).
    Returns it set up; a vanished remote is forgotten and re-provisioned."""
    from agent_base.sandbox.registry import sandbox_from_config
    from agent_base.sandbox.sandbox_types import SandboxGone

    sandbox = None
    if cfg.sandbox_config is not None:
        try:
            sandbox = sandbox_from_config(cfg.sandbox_config)
        except Exception:  # noqa: BLE001 — unknown/legacy type: fall back
            sandbox = None
    if sandbox is None and sandbox_factory is not None:
        sandbox = sandbox_factory(agent_uuid)
        if inspect.isawaitable(sandbox):
            sandbox = await sandbox
    if sandbox is None or not setup:
        return sandbox
    try:
        await sandbox.setup()
    except SandboxGone:
        sandbox.forget_remote()
        await sandbox.setup()
    return sandbox


async def destroy_session_sandbox(
    handles: "StorageHandles",
    *,
    agent_uuid: str,
    principal: "SessionPrincipal",
    sessions: object | None = None,
    sandbox_coordinator: "SandboxCoordinator | None" = None,
) -> bool:
    """Kill the backing sandbox of ``agent_uuid`` (session-DELETE semantics)
    and null its persisted binding. Refuses a busy resident session with
    ``SessionBusy``. Returns False when nothing was bound. The next
    ``initialize()`` would provision a fresh sandbox and rehydrate it."""
    if sessions is not None and getattr(sessions, "is_resident", None) is not None:
        if sessions.is_resident(agent_uuid):
            if not await sessions.evict(agent_uuid):
                raise SessionBusy(agent_uuid)
    adapter = handles.config.for_principal(principal)
    cfg = await adapter.load(agent_uuid)
    if cfg is None or cfg.sandbox_config is None:
        return False
    if sandbox_coordinator is not None:
        context = await _coordination_context(handles, agent_uuid, principal)
        await sandbox_coordinator.destroy(context)
        return True
    from agent_base.sandbox.registry import sandbox_from_config
    from agent_base.sandbox.sandbox_types import SandboxGone

    sandbox = sandbox_from_config(cfg.sandbox_config)
    try:
        await sandbox.teardown()
    except SandboxGone:
        pass
    cfg.sandbox_config = None
    await adapter.save(cfg)
    return True


__all__ = [
    "ForkResetError",
    "CheckpointNotFound",
    "SessionBusy",
    "destroy_session_sandbox",
    "fork_session",
    "reset_session",
]
