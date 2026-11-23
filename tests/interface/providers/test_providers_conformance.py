"""Interface spec: the conformance tier, which every provider passes, so
"supported" means the same thing for every model
(mental_model/planned_items/multi-provider-models.md › The conformance tier).

Each spec runs once per format, against a scripted server for it: Claude through
``AnthropicProvider``, every other model through ``OpenAIChatProvider`` on Vercel
AI Gateway, both built by ``ModelRouter``.

- A plain reply streams as text frames, closed by a final one, and the turn
  ends with the text, ``end_turn`` and usage.
- Parallel tool calls in one step run, and their results, an error among them,
  go back matched to their calls by id.
- A frontend tool pauses the run; the reply resumes it, and the model gets it.
- A rate limit and a server error are retried within the ``RetryPolicy``.
- A prompt over the context window is ``CONTEXT_OVERFLOW``, not retried, so the
  turn loop compacts.
- An image in the user's message reaches the model.
- A sub-agent on the other format answers its parent.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_base.common_tools.sub_agent_tool import SubAgentSpec
from agent_base.core.errors import ErrorCode
from agent_base.core.messages import Message
from agent_base.core.types import ImageContent, TextContent, ToolResultContent
from agent_base.providers.anthropic import AnthropicAgent
from agent_base.streaming.types import TextDelta
from agent_base.tools import tool

from ._two_formats import ROUTES, Sink, WireAPI, wait_until

pytestmark = pytest.mark.parametrize("route", ROUTES, ids=[r.name for r in ROUTES])


@tool
def lookup(word: str) -> str:
    """Look a word up."""
    return "A small animal."


@tool
def broken() -> str:
    """A tool that fails."""
    raise RuntimeError("the lookup service is down")


@tool(executor="frontend")
def confirm(question: str) -> str:
    """Ask the user to confirm."""
    return ""


def _agent(route, wire, **kw):
    kw.setdefault("system_prompt", "Be brief.")
    return AnthropicAgent(model=route.model, provider_value=route.provider(wire), **kw)


async def test_a_plain_reply_streams_its_text_and_ends_the_turn(route):
    wire = WireAPI(**{route.api: [route.text("Hello there.")]})
    sink = Sink()
    turn = await route.provider(wire).generate_stream(
        system_prompt="Be brief.", messages=[Message.user("hi")], tool_schemas=[], llm_config=None,
        model=route.model, sink=sink, agent_uuid="a1")

    texts = [d for d in sink.deltas if isinstance(d, TextDelta)]
    assert "".join(d.text for d in texts) == "Hello there." and texts[-1].is_final
    assert [b.text for b in turn.message.content if isinstance(b, TextContent)] == ["Hello there."]
    assert turn.message.stop_reason == "end_turn"
    assert turn.message.usage.input_tokens > 0 and turn.message.usage.output_tokens > 0


async def test_parallel_tool_calls_and_a_tool_error_go_back_matched_by_id(route):
    one, two = route.call_ids
    wire = WireAPI(**{route.api: [route.calls((one, "lookup", {"word": "cat"}), (two, "broken", {})),
                                  route.text("Done.")]})
    result = await _agent(route, wire, tools=[lookup, broken]).run("Look up 'cat', and try the other tool.")

    assert result.final_answer == "Done."
    returned = route.tool_results(wire.bodies(route.api)[1])
    assert returned[one] == ("A small animal.", False)
    assert returned[two][1] and "the lookup service is down" in returned[two][0]


async def test_a_frontend_tool_pauses_the_run_and_the_reply_resumes_it(route):
    one, _ = route.call_ids
    wire = WireAPI(**{route.api: [route.calls((one, "confirm", {"question": "Go?"})), route.text("Going.")]})
    agent = _agent(route, wire, tools=[confirm])
    run = asyncio.create_task(agent.run("Ask me first."))
    await wait_until(lambda: agent.agent_config is not None and agent.agent_config.pending_relay is not None)

    relay = agent.agent_config.pending_relay
    assert [c.name for c in relay.frontend_calls] == ["confirm"]
    await agent.reply(relay.cid, [ToolResultContent(tool_name="confirm", tool_id=one, tool_result="yes")])
    result = await asyncio.wait_for(run, 10)

    assert result.final_answer == "Going."
    assert route.tool_results(wire.bodies(route.api)[1])[one] == ("yes", False)


async def test_a_rate_limit_and_a_server_error_are_retried_within_the_policy(route):
    wire = WireAPI(**{route.api: [route.rate_limited(), route.overloaded(), route.text("ok")]})
    turn = await route.provider(wire).generate_stream(
        system_prompt=None, messages=[Message.user("hi")], tool_schemas=[], llm_config=None,
        model=route.model, sink=Sink(), agent_uuid="a1")

    assert turn.message.content[0].text == "ok" and len(wire.requests) == 3


async def test_a_prompt_over_the_context_window_is_context_overflow(route):
    wire = WireAPI(**{route.api: [route.overflow()]})
    provider = route.provider(wire)
    with pytest.raises(Exception) as caught:
        await provider.generate_stream(system_prompt=None, messages=[Message.user("hi")], tool_schemas=[],
                                       llm_config=None, model=route.model, sink=Sink(), agent_uuid="a1")

    error = provider.classify_error(caught.value)
    assert error.code == ErrorCode.CONTEXT_OVERFLOW and not error.retriable
    assert len(wire.requests) == 1


async def test_an_image_in_the_users_message_reaches_the_model(route):
    wire = WireAPI(**{route.api: [route.text("A cat.")]})
    image = ImageContent(media_type="image/png", source_type="base64", data="iVBORw0KGgo=")
    await route.provider(wire).generate_stream(
        system_prompt=None, messages=[Message.user([image, TextContent(text="What is it?")])], tool_schemas=[],
        llm_config=None, model=route.model, sink=Sink(), agent_uuid="a1")

    parts = [{k: v for k, v in p.items() if k != "cache_control"}  # Anthropic's prompt-cache marker
             for p in wire.bodies(route.api)[0]["messages"][-1]["content"]]
    if route.api == "anthropic":
        assert {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": "iVBORw0KGgo="}} in parts
    else:
        assert {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}} in parts


async def test_a_sub_agent_on_the_other_format_answers_its_parent(route):
    other = next(r for r in ROUTES if r is not route)
    one, _ = route.call_ids
    wire = WireAPI(**{
        route.api: [route.calls((one, "spawn_subagent", {"agent_name": "helper", "task": "Summarize: the cat sat."})),
                    route.text("The helper says: a cat sat.")],
        other.api: [other.text("A cat sat.")],
    })
    parent = _agent(route, wire, subagents={"helper": SubAgentSpec(
        name="helper", description="Summarizes.", system_prompt="Summarize in four words.", model=other.model)})
    result = await parent.run("Summarize: the cat sat.")

    assert result.final_answer == "The helper says: a cat sat."
    [child_request] = wire.bodies(other.api)
    assert child_request["model"] == other.model
    assert route.tool_results(wire.bodies(route.api)[1])[one][0].find("A cat sat.") >= 0
