"""Checkpoint domain model for fork / reset-to-checkpoint.

A ``Checkpoint`` is an immutable restore point for one completed turn boundary.
It carries the serialized ``AgentConfig`` **minus** the two unbounded transcript
fields (``context_messages`` and ``conversation_log``): those are sliced into
per-message segments in the content-addressed blob store, and the checkpoint
holds ordered segment-key arrays. An unchanged transcript prefix dedupes to zero
new blobs, so steady-state checkpoint storage is O(distinct bytes), not O(n²) in
transcript length (see ``agent_base/storage/checkpoint_codec.py``).

The ``CheckpointAdapter`` ABC lives in ``agent_base.storage.base`` alongside the
other storage adapters (it parallels ``ConversationAdapter``); these are the
entity dataclasses it persists. The library is workbook-agnostic — a consumer
(e.g. Nova) rides its own refs in the opaque ``consumer_payload`` slot, which the
library never parses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Allowed values for ``CheckpointRef.fidelity`` — how complete the restore is.
#: ``"full"`` everything captured; ``"degraded"`` some sandbox files skipped
#: (oversize); ``"none"`` no workspace snapshot (config-only).
CheckpointFidelity = str


@dataclass(frozen=True)
class CheckpointRef:
    """Lightweight pointer to a checkpoint — what the picker lists.

    Cheap to load in bulk (``list_refs``) without materializing the heavy
    ``config_snapshot`` / segment arrays.
    """

    agent_uuid: str
    sequence_number: int
    run_id: str
    created_at: str
    fidelity: CheckpointFidelity = "full"


@dataclass
class Checkpoint:
    """An immutable restore point captured at a completed turn boundary.

    ``config_base`` is ``serialize_config()`` with ``context_messages`` and
    ``conversation_log`` removed; ``transcript_segments`` / ``log_segments`` are
    the ordered content-addressed keys those two fields were sliced into.
    ``transcript_codec_v`` is ``0`` when stored full-inline (no blob store wired)
    and ``1`` when CAS-segmented.
    """

    ref: CheckpointRef
    config_base: dict[str, Any]
    transcript_segments: list[str] = field(default_factory=list)
    log_segments: list[str] = field(default_factory=list)
    transcript_codec_v: int = 1
    sandbox_manifest_ref: str | None = None
    #: OPAQUE to the library — the consumer reads/writes its own refs here.
    consumer_payload: dict[str, Any] = field(default_factory=dict)
    #: Reset archives the tail by flipping this flag; rows are NEVER deleted.
    archived: bool = False
