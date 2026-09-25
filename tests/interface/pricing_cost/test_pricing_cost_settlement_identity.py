"""Settlement identity + child usage-report propagation (GF-P8G2 / GF-P7G1).

Covers interface_plan/subsystems/pricing-cost.md §2.4 + tenancy-principal.md
§A.1 + the AMENDMENTS "Open-gap fixes (2026-06-12)" GF-P8G2/GF-P7G1 entries:

  - **Settlement carries the runtime's NAMED principal** once identity is
    threaded — via the ``principal=`` ctor passthrough on ``AnthropicAgent``
    (forwarded to the ``AgentRuntime`` base) OR via ``set_principal`` after
    build (the ``SessionManager`` path). Before GF-P8G2 every resident agent
    settled ANONYMOUS and consumer billing callbacks skipped (the live smoke
    billed ZERO turns).
  - **GF-P7G1 (ratified D2): child sub-agents get the parent's
    ``on_usage_report`` subscribers propagated AT BUILD TIME, recursively** —
    a child turn's ``TurnSettlement`` reaches the parent-registered
    subscriber, stamped with the child's ``agent_id``, the parent's id as
    ``parent_agent_id``, and the (inherited) named principal. A child's own
    ``SubAgentTool`` propagates again to grandchildren. Timing contract:
    subscribers registered on the parent AFTER a child was built do NOT
    retro-attach to that child — the next spawn picks them up (the same
    semantics the consumer's ``_propagate_to_subagents`` workaround had).
    No ``SettlementAggregator`` involvement (I9 RESOLVED: deleted 2026-07-14).

The provider generation step is stubbed (one ``end_turn`` assistant message);
everything else — spawn, propagation, settle, emit — is the real path.
"""
from __future__ import annotations

from agent_base.common_tools.sub_agent_tool import (
    SubAgentParentContext,
    SubAgentSpec,
    SubAgentTool,
)
from agent_base.core.cost import TurnSettlement
from agent_base.core.identity import SessionPrincipal
from agent_base.core.messages import Message, Usage
from agent_base.core.provider import ProviderTurn
from agent_base.providers.anthropic import AnthropicAgent

OWNER = SessionPrincipal(tenant="org_1", subject="member_1")


def _stub_provider(agent: AnthropicAgent, reply_text: str) -> None:
    """Fake the generation step: one end_turn assistant message."""

    async def _fake_generate(**kwargs):
        msg = Message.assistant(reply_text)
        msg.stop_reason = "end_turn"
        msg.usage = Usage(input_tokens=10, output_tokens=5)
        return ProviderTurn(message=msg)

    agent.provider.generate = _fake_generate          # type: ignore[method-assign]
    agent.provider.generate_stream = _fake_generate   # type: ignore[method-assign]


def _stubbing_builder(tool: SubAgentTool, built: list[AnthropicAgent]):
    """Wrap the DEFAULT builder so spawned children get a stubbed provider.

    Propagation lives in ``SubAgentTool.run`` AFTER the builder returns, so
    wrapping the builder (as a test seam for the provider stub) exercises the
    same library propagation path as the default builder.
    """
    inner = tool._default_child_agent_builder

    def builder(spec, resume_uuid, parent_context):
        child = inner(spec, resume_uuid, parent_context)
        _stub_provider(child, "child done")
        built.append(child)
        return child

    return builder


def _wire_subagent_tool(parent: AnthropicAgent, tool: SubAgentTool) -> None:
    tool.set_parent_context(
        SubAgentParentContext(
            parent_agent_uuid=parent.agent_uuid,
            parent_agent=parent,
        )
    )


# ─── settlement carries the named principal (GF-P8G2) ─────────────────────


async def test_settlement_carries_the_ctor_principal():
    # AnthropicAgent(principal=...) forwards to the AgentRuntime base — the
    # exposed passthrough GF-P8G2 specs.
    agent = AnthropicAgent(system_prompt="t", principal=OWNER)
    _stub_provider(agent, "done")
    await agent.initialize()

    result = await agent.run("hello")

    settlement = result.settlement
    assert isinstance(settlement, TurnSettlement)
    assert settlement.principal == OWNER
    assert settlement.principal is not None
    assert not settlement.principal.is_anonymous()


async def test_settlement_carries_the_set_principal_identity():
    # The SessionManager path (the live-smoke regression): identity threaded
    # AFTER build via set_principal must reach the NEXT settlement — and the
    # on_usage_report subscriber must see it (consumer billing keys off the
    # settlement principal, not the request context).
    agent = AnthropicAgent(system_prompt="t")
    _stub_provider(agent, "done")
    await agent.initialize()
    agent.set_principal(OWNER)

    seen: list[TurnSettlement] = []
    agent.on_usage_report(seen.append)

    result = await agent.run("hello")

    assert result.settlement.principal == OWNER
    assert len(seen) == 1
    assert seen[0].principal == OWNER
    assert seen[0] is result.settlement       # identical bytes, no re-extraction


async def test_anonymous_runtime_still_settles_anonymous():
    # Pre-identity baseline: nothing threaded → the settlement principal is
    # the anonymous ambient default (never None on a constructed runtime).
    agent = AnthropicAgent(system_prompt="t")
    _stub_provider(agent, "done")
    await agent.initialize()

    result = await agent.run("hello")

    assert result.settlement.principal is not None
    assert result.settlement.principal.is_anonymous()


# ─── GF-P7G1: child propagation reaches the parent subscriber ─────────────


async def test_child_settlement_reaches_the_parent_registered_subscriber():
    parent = AnthropicAgent(system_prompt="p", principal=OWNER)
    await parent.initialize()

    seen: list[TurnSettlement] = []
    parent.on_usage_report(seen.append)

    built: list[AnthropicAgent] = []
    tool = SubAgentTool(
        agents={"helper": SubAgentSpec(description="helps", system_prompt="h")}
    )
    tool._child_agent_builder = _stubbing_builder(tool, built)
    _wire_subagent_tool(parent, tool)

    envelope = await tool.run("helper", "do the thing")

    assert not envelope.is_error
    assert len(built) == 1
    child = built[0]
    assert len(seen) == 1                      # the child's turn settled once
    settlement = seen[0]
    assert settlement.agent_id == child.agent_uuid
    assert settlement.parent_agent_id == parent.agent_uuid
    # G2 x G1: the child inherited the parent's NAMED principal at spawn, so
    # the propagated settlement is billable (not anonymous-skipped).
    assert settlement.principal == OWNER


async def test_propagation_is_recursive_through_the_childs_own_subagent_tool():
    # D2: a child's own SubAgentTool propagates AGAIN — the parent-registered
    # subscriber receives the GRANDCHILD's settlement too.
    parent = AnthropicAgent(system_prompt="p", principal=OWNER)
    await parent.initialize()

    seen: list[TurnSettlement] = []
    parent.on_usage_report(seen.append)

    built: list[AnthropicAgent] = []
    tool = SubAgentTool(
        agents={
            "helper": SubAgentSpec(
                description="helps",
                system_prompt="h",
                subagents={
                    "grand": SubAgentSpec(description="grand", system_prompt="g")
                },
            )
        }
    )
    tool._child_agent_builder = _stubbing_builder(tool, built)
    _wire_subagent_tool(parent, tool)

    await tool.run("helper", "level one")
    child = built[0]

    # The propagated callback is ON the child's list — that is exactly what
    # the child's own SubAgentTool reads when it spawns the next level.
    child_tool = child._sub_agent_tool
    assert child_tool is not None
    grand_built: list[AnthropicAgent] = []
    child_tool._child_agent_builder = _stubbing_builder(child_tool, grand_built)

    await child_tool.run("grand", "level two")

    assert len(grand_built) == 1
    grandchild = grand_built[0]
    assert len(seen) == 2                      # child turn + grandchild turn
    grand_settlement = seen[-1]
    assert grand_settlement.agent_id == grandchild.agent_uuid
    assert grand_settlement.parent_agent_id == child.agent_uuid
    assert grand_settlement.principal == OWNER


async def test_late_parent_subscribers_do_not_retro_attach():
    # Timing contract (D2): propagation happens AT BUILD TIME. A subscriber
    # registered on the parent after a child was built does not retro-attach
    # to that child; the NEXT spawn picks it up.
    parent = AnthropicAgent(system_prompt="p", principal=OWNER)
    await parent.initialize()

    early: list[TurnSettlement] = []
    parent.on_usage_report(early.append)

    built: list[AnthropicAgent] = []
    tool = SubAgentTool(
        agents={"helper": SubAgentSpec(description="helps", system_prompt="h")}
    )
    tool._child_agent_builder = _stubbing_builder(tool, built)
    _wire_subagent_tool(parent, tool)

    await tool.run("helper", "first spawn")
    first_child = built[0]

    late: list[TurnSettlement] = []
    parent.on_usage_report(late.append)        # AFTER the first child was built

    assert late.append not in first_child._usage_report_callbacks
    assert early.append in first_child._usage_report_callbacks

    await tool.run("helper", "second spawn")
    second_child = built[1]

    assert late.append in second_child._usage_report_callbacks
    assert len(late) == 1                      # only the second child's turn
    assert late[0].agent_id == second_child.agent_uuid
    assert len(early) == 2                     # both children's turns
