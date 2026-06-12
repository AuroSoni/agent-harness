"""SubAgentSpec field-aware deepcopy snapshot contract (tools.md §2.5, P8-G1).

The runtime snapshots a ``SubAgentSpec`` via ``copy.deepcopy`` in two places:
``SubAgentTool._coerce_spec`` (an explicitly-passed spec) and
``SubAgentSpec.from_template_agent`` (nested specs). That snapshot is
**field-aware**:

- runtime-resource fields — ``tools`` / ``frontend_tools`` / ``memory_store`` —
  are kept by REFERENCE. A spec whose tool holds a live resource (e.g. an
  asyncpg pool, whose ``__deepcopy__`` raises ``TypeError``) snapshots without
  blowing up, and the snapshot's tool IS the original instance (identity);
- the ``tools`` / ``frontend_tools`` LIST CONTAINERS are copied fresh, so
  mutating a snapshot's list never touches the original;
- plain DATA fields (prompts, model, config, limits, ``retry_policy``, nested
  ``subagents``) are independent deep copies.

This is the upstreamed P8-G1 fix: the consumer's ``NovaSubAgentSpec`` subclass
(which workarounded this) collapses back onto ``SubAgentSpec`` with zero
behavior change.
"""
import copy

import pytest

from agent_base.common_tools.sub_agent_tool import SubAgentSpec, SubAgentTool


class _PoolBackedTool:
    """A tool instance whose deepcopy explodes — like Nova's skill tools that
    carry the skills registry's process-wide asyncpg pool (``TypeError: no
    default __reduce__ due to non-trivial __cinit__``)."""

    def __init__(self, name: str = "skill_tool"):
        self.name = name

    def __deepcopy__(self, memo):  # noqa: D401 - simulate the pool
        raise TypeError(
            "cannot deepcopy a live connection pool (no default __reduce__)"
        )


def test_deepcopy_keeps_pool_backed_tool_by_reference():
    """A spec whose ``tools`` hold a deepcopy-hostile runtime object survives
    ``copy.deepcopy`` and the cloned tool IS the original instance (identity)."""
    pool_tool = _PoolBackedTool()
    spec = SubAgentSpec(
        name="researcher",
        description="researches",
        tools=[pool_tool],
    )

    clone = copy.deepcopy(spec)

    assert clone is not spec
    assert clone.tools is not spec.tools          # fresh list container
    assert clone.tools[0] is pool_tool            # SAME instance (by reference)


def test_coerce_spec_preserves_tool_identity():
    """``SubAgentTool._coerce_spec`` deepcopies an explicit spec — the pool-
    backed tool must survive and stay the SAME instance after coercion."""
    pool_tool = _PoolBackedTool()
    spec = SubAgentSpec(
        name="researcher",
        description="researches",
        tools=[pool_tool],
    )

    coerced = SubAgentTool._coerce_spec("researcher", spec)

    assert coerced is not spec
    assert coerced.tools[0] is pool_tool


def test_subagent_tool_construction_does_not_explode_on_pool_tool():
    """The whole ``SubAgentTool`` build path (which coerces every spec) does not
    500 on a pool-backed tool — the live-smoke regression this fix closes."""
    pool_tool = _PoolBackedTool()
    tool = SubAgentTool(
        agents={
            "researcher": SubAgentSpec(
                name="researcher",
                description="researches",
                tools=[pool_tool],
            )
        }
    )

    assert tool.specs["researcher"].tools[0] is pool_tool


def test_frontend_tools_and_memory_store_kept_by_reference():
    """``frontend_tools`` and ``memory_store`` are runtime-resource fields too —
    kept by reference (fresh list container for the list field)."""
    fe_tool = _PoolBackedTool("frontend_tool")
    sentinel_store = object()  # a stand-in for a shared MemoryStore singleton
    spec = SubAgentSpec(
        name="researcher",
        description="researches",
        frontend_tools=[fe_tool],
        memory_store=sentinel_store,
    )

    clone = copy.deepcopy(spec)

    assert clone.frontend_tools is not spec.frontend_tools  # fresh container
    assert clone.frontend_tools[0] is fe_tool               # shared member
    assert clone.memory_store is sentinel_store             # shared singleton


def test_snapshot_list_containers_are_isolated():
    """Mutating a snapshot's ``tools`` list does NOT leak into the original —
    the container is copied even though the members are shared."""
    tool_a = _PoolBackedTool("a")
    tool_b = _PoolBackedTool("b")
    spec = SubAgentSpec(name="researcher", description="x", tools=[tool_a])

    clone = copy.deepcopy(spec)
    clone.tools.append(tool_b)

    assert spec.tools == [tool_a]            # original untouched
    assert clone.tools == [tool_a, tool_b]


def test_data_fields_are_independent_copies():
    """Plain data fields ARE deep-copied — mutating the snapshot's nested
    mutable data leaves the original alone."""
    spec = SubAgentSpec(
        name="researcher",
        description="x",
        system_prompt="ORIG",
        config={"nested": {"limit": 1}},
    )

    clone = copy.deepcopy(spec)
    clone.system_prompt = "CHANGED"
    clone.config["nested"]["limit"] = 999

    assert spec.system_prompt == "ORIG"
    assert spec.config["nested"]["limit"] == 1          # deep, not shared


def test_nested_subagents_deepcopied_but_keep_their_tool_identity():
    """``subagents`` is a DATA field (deep-copied to an independent dict), yet
    each nested spec recurses through this same field-aware ``__deepcopy__`` so
    its OWN tool instances stay shared by reference."""
    nested_pool_tool = _PoolBackedTool("nested")
    spec = SubAgentSpec(
        name="parent",
        description="x",
        subagents={
            "child": SubAgentSpec(
                name="child", description="y", tools=[nested_pool_tool]
            )
        },
    )

    clone = copy.deepcopy(spec)

    assert clone.subagents is not spec.subagents                  # fresh dict
    assert clone.subagents["child"] is not spec.subagents["child"]  # fresh spec
    # ...but the nested runtime tool is still the SAME instance.
    assert clone.subagents["child"].tools[0] is nested_pool_tool


def test_retry_policy_is_data_and_deepcopied():
    """``retry_policy`` is a DATA field (a provider value object), so it is
    deep-copied — NOT shared by reference like the runtime-resource fields."""
    class _RetryPolicy:
        def __init__(self, max_retries: int):
            self.max_retries = max_retries

    policy = _RetryPolicy(3)
    spec = SubAgentSpec(name="r", description="x", retry_policy=policy)

    clone = copy.deepcopy(spec)

    assert clone.retry_policy is not policy
    assert clone.retry_policy.max_retries == 3
