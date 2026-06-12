"""Red-suite spec: the provider-agnostic compaction interface + hook context.

Covers interface_plan/subsystems/core.md:
  - §2.3 — ``agent_base/core/compaction_types.py``: ``CompactionConfig``
    (declarative, Serializable convention), ``CompactionStats`` (typed result,
    replaces the ``last_compaction_meta`` dict), and the ``Compactor``
    protocol; ``agent_base/hooks/compaction.py``: ``CompactionContext``.
  - I10 — ``trigger`` vocabulary gains ``"overflow"`` on config/stats/context.
  - Fork L (V1) — veto semantics are enforced by the runtime (loop subsystem);
    here we spec only the typed seam the loop consumes (trigger + stats on
    the context, dataclasses.replace for the after_compact view).

``HookContext`` (agent_loop_hooks subsystem) is a collaborator: we construct
its R4 canonical field set with in-file fakes, never deep-test it.
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Protocol

import pytest

from agent_base.core.compaction_types import (
    CompactionConfig,
    CompactionStats,
    Compactor,
)
from agent_base.core.config import AgentConfig
from agent_base.core.hooks.context import CompactionContext, HookContext
from agent_base.core.serializable import CORE_SCHEMA_VERSION, SCHEMA_VERSION_KEY

# ── collaborator fakes (R4 canonical HookContext field set) ─────────────────


class _FakeStorageHandles:
    config = None
    conversation = None
    run = None


class _FakeLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass


def _emit(body: Any, *, correlation_id: str | None = None, expects_reply: bool = False) -> None:
    pass


async def _once(key: str, fn: Any) -> Any:
    return await fn()


def _base_hook_kwargs() -> dict[str, Any]:
    return dict(
        run_id="run-1",
        agent_id="agent-1",
        parent_agent_id=None,
        principal=None,
        executor="backend",
        sandbox=None,
        storage=_FakeStorageHandles(),
        media=None,
        memory=None,
        agent_config=AgentConfig(agent_uuid="agent-1"),
        conversation=None,
        emit=_emit,
        once=_once,
        logger=_FakeLogger(),
    )


def _stats(trigger: str = "auto") -> CompactionStats:
    return CompactionStats(
        trigger=trigger,
        applied=True,
        messages_compacted=12,
        messages_preserved=4,
        summary_tokens=512,
        tokens_before=180_000,
        tokens_after=60_000,
    )


# ── CompactionConfig ────────────────────────────────────────────────────────


def test_compaction_config_defaults():
    cfg = CompactionConfig()
    assert cfg.threshold_tokens == 160_000
    assert cfg.preserve_recent_tokens == 40_000
    assert cfg.summary_prompt is None
    assert cfg.model is None


def test_compaction_config_to_dict_stamps_core_schema_version():
    d = CompactionConfig().to_dict()
    assert d[SCHEMA_VERSION_KEY] == CORE_SCHEMA_VERSION


def test_compaction_config_round_trips():
    cfg = CompactionConfig(
        threshold_tokens=120_000,
        preserve_recent_tokens=30_000,
        summary_prompt="Summarize tersely.",
        model="claude-sonnet-4-5",
    )
    back = CompactionConfig.from_dict(cfg.to_dict())
    assert back.threshold_tokens == 120_000
    assert back.preserve_recent_tokens == 30_000
    assert back.summary_prompt == "Summarize tersely."
    assert back.model == "claude-sonnet-4-5"


def test_compaction_config_from_dict_missing_keys_take_defaults():
    # §2.1 Serializable convention: "missing keys take defaults".
    cfg = CompactionConfig.from_dict({})
    assert cfg.threshold_tokens == 160_000
    assert cfg.preserve_recent_tokens == 40_000
    assert cfg.summary_prompt is None
    assert cfg.model is None


def test_compaction_config_from_dict_accepts_a_stamp_only_dict():
    # The `_v` stamp alone is tolerated (it is metadata, not a config field).
    cfg = CompactionConfig.from_dict({SCHEMA_VERSION_KEY: CORE_SCHEMA_VERSION})
    assert cfg.threshold_tokens == 160_000
    assert cfg.preserve_recent_tokens == 40_000
    assert cfg.summary_prompt is None
    assert cfg.model is None


# ── CompactionStats ─────────────────────────────────────────────────────────


def test_compaction_stats_shape_and_token_defaults():
    stats = CompactionStats(
        trigger="manual",
        applied=False,
        messages_compacted=0,
        messages_preserved=9,
        summary_tokens=0,
    )
    assert stats.trigger == "manual"
    assert stats.applied is False
    assert stats.messages_compacted == 0
    assert stats.messages_preserved == 9
    assert stats.summary_tokens == 0
    assert stats.tokens_before == 0
    assert stats.tokens_after == 0


def test_compaction_stats_is_frozen():
    stats = _stats()
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.applied = False  # type: ignore[misc]


def test_compaction_stats_accepts_every_trigger_value():
    # I10: "overflow" joined the vocabulary.
    for trigger in ("auto", "manual", "overflow"):
        assert _stats(trigger).trigger == trigger


# ── Compactor protocol ──────────────────────────────────────────────────────


def test_compactor_is_a_protocol_with_the_compaction_members():
    assert getattr(Compactor, "_is_protocol", False) is True
    assert Protocol in Compactor.__mro__
    assert "should_compact" in dir(Compactor)
    assert "compact" in dir(Compactor)
    assert "config" in getattr(Compactor, "__annotations__", {})


def test_compactor_protocol_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Compactor()  # type: ignore[misc]


def test_compactor_compact_signature_is_async_keyword_only_with_auto_default():
    # §2.3 pins the concrete seam: async compact(context_messages, *, model,
    # ctx, trigger="auto") -> (messages, CompactionStats). The keyword-only
    # params and the trigger default are the I10 surface the loop drives.
    assert inspect.iscoroutinefunction(Compactor.compact)
    params = inspect.signature(Compactor.compact).parameters
    assert list(params) == ["self", "context_messages", "model", "ctx", "trigger"]
    assert params["context_messages"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["model"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["ctx"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["trigger"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["trigger"].default == "auto"


def test_compactor_should_compact_signature_is_sync_two_positional():
    # §2.3: should_compact(context_messages, estimated_tokens) -> bool — a
    # plain sync predicate, no keyword-only ceremony.
    assert not inspect.iscoroutinefunction(Compactor.should_compact)
    params = inspect.signature(Compactor.should_compact).parameters
    assert list(params) == ["self", "context_messages", "estimated_tokens"]
    assert params["context_messages"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["estimated_tokens"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


# ── CompactionContext (the hook context this subsystem defines) ─────────────


def test_compaction_context_subclasses_hook_context():
    assert issubclass(CompactionContext, HookContext)


def test_compaction_context_before_compact_defaults():
    # before_compact view: trigger set, stats=None.
    # estimated_tokens defaults to None per agent-loop-hooks.md §2.2 — the doc
    # that OWNS the HookContext hierarchy (R4: superset is canonical) — and its
    # suite pins `is None`; core.md §2.3's `int = 0` was the stale draft line.
    ctx = CompactionContext(**_base_hook_kwargs())
    assert ctx.trigger == "auto"
    assert ctx.estimated_tokens is None
    assert ctx.stats is None
    # Inherited identity threading (contract §1.2): stamped, not hand-passed.
    assert ctx.run_id == "run-1"
    assert ctx.agent_id == "agent-1"
    assert callable(ctx.emit)
    assert callable(ctx.once)


def test_compaction_context_accepts_the_overflow_trigger():
    # I10: overflow routes through before_compact(trigger="overflow").
    ctx = CompactionContext(
        trigger="overflow", estimated_tokens=205_000, **_base_hook_kwargs()
    )
    assert ctx.trigger == "overflow"
    assert ctx.estimated_tokens == 205_000


def test_compaction_context_after_compact_view_carries_typed_stats():
    # §2.3 wiring: after_compact receives replace(cc, stats=stats).
    before = CompactionContext(
        trigger="auto", estimated_tokens=170_000, **_base_hook_kwargs()
    )
    stats = _stats()
    after = dataclasses.replace(before, stats=stats)
    assert after.stats is stats
    assert after.stats.tokens_before - after.stats.tokens_after == 120_000
    assert before.stats is None
