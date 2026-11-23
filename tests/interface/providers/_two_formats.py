"""Collaborators for the provider specs: one scripted server for both formats a
provider in this library speaks, and a ``Route`` per format, so a spec states a
behaviour once and every provider has to pass it.

- ``WireAPI`` answers Chat Completions at ``…/chat/completions`` and Anthropic
  Messages at ``…/messages``, each request taking the next scripted response
  for its API. A streaming request to either gets the response as server-sent
  events; a non-streaming one gets it as one JSON body.
- ``Route`` builds the provider for its format through a ``ModelRouter``: Claude
  through ``AnthropicProvider`` (the SDK's own retries off, so the provider's
  ``RetryPolicy`` is the one retrying), every other model through
  ``VercelGateway``. It scripts replies in its format.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

import anthropic
import httpx

from agent_base.core.provider import RetryPolicy
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.providers.router import ModelRouter
from agent_base.providers.vercel_gateway import VercelGateway

FAST = RetryPolicy(max_retries=3, base_delay=0.0)
GATEWAY_KEY = "gateway-key"
DIRECT_KEY = "direct-key"


# ── Chat Completions ─────────────────────────────────────────────────────────


def chat_chunk(*, content: str | None = None, reasoning: str | None = None, reasoning_details: list | None = None,
               tool_calls: list | None = None, finish: str | None = None, usage: dict | None = None,
               model: str = "google/gemini-2.5-flash") -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in (("content", content), ("reasoning", reasoning), ("reasoning_details", reasoning_details),
                       ("tool_calls", tool_calls)):
        if value is not None:
            delta[key] = value
    data: dict[str, Any] = {"id": "gen_1", "object": "chat.completion.chunk", "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        data["usage"] = usage
    return data


def chat_stream(*chunks: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _chat_as_json(response: httpx.Response) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": None}
    calls: dict[int, dict[str, Any]] = {}
    out: dict[str, Any] = {"object": "chat.completion"}
    finish = None
    for line in response.content.decode().splitlines():
        if not line.startswith("data: {"):
            continue
        data = json.loads(line[len("data: "):])
        out["id"], out["model"] = data.get("id"), data.get("model")
        if data.get("usage"):
            out["usage"] = data["usage"]
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning"):
                if delta.get(key):
                    message[key] = (message.get(key) or "") + delta[key]
            if delta.get("reasoning_details"):
                message.setdefault("reasoning_details", []).extend(delta["reasoning_details"])
            for raw in delta.get("tool_calls") or []:
                call = calls.setdefault(raw.get("index", 0), {"type": "function", "function": {"arguments": ""}})
                call["id"] = raw.get("id") or call.get("id")
                fn = raw.get("function") or {}
                call["function"]["name"] = fn.get("name") or call["function"].get("name")
                call["function"]["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    out["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
    return httpx.Response(200, json=out)


# ── Anthropic Messages ───────────────────────────────────────────────────────


def anthropic_reply(blocks: list[dict[str, Any]], *, stop_reason: str, model: str = "claude-sonnet-5-5",
                    usage: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": blocks,
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": usage or {"input_tokens": 10, "output_tokens": 20},
    })


def _anthropic_as_events(response: httpx.Response) -> httpx.Response:
    """A scripted Messages reply as the server-sent events a streaming request gets."""
    message = response.json()
    events: list[tuple[str, dict[str, Any]]] = [("message_start", {"type": "message_start", "message": {
        **{k: v for k, v in message.items() if k != "content"}, "content": [], "stop_reason": None,
        "usage": {**message["usage"], "output_tokens": 1}}})]
    for index, block in enumerate(message["content"]):
        kind = block["type"]
        if kind == "text":
            start, deltas = {"type": "text", "text": ""}, [{"type": "text_delta", "text": block["text"]}]
        elif kind == "tool_use":
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})}]
        elif kind == "thinking":
            start = {"type": "thinking", "thinking": ""}
            deltas = [{"type": "thinking_delta", "thinking": block["thinking"]},
                      {"type": "signature_delta", "signature": block["signature"]}]
        else:
            raise AssertionError(f"no stream form scripted for {kind!r} blocks")
        events.append(("content_block_start", {"type": "content_block_start", "index": index, "content_block": start}))
        events += [("content_block_delta", {"type": "content_block_delta", "index": index, "delta": d}) for d in deltas]
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                                     "usage": {"output_tokens": message["usage"]["output_tokens"]}}))
    events.append(("message_stop", {"type": "message_stop"}))
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


# ── the server ───────────────────────────────────────────────────────────────


class WireAPI:
    """Routes each request to the next scripted response for its API."""

    def __init__(self, *, chat: list[httpx.Response] | None = None,
                 anthropic: list[httpx.Response] | None = None) -> None:
        self.scripts = {"chat": list(chat or []), "anthropic": list(anthropic or [])}
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        kind = "chat" if request.url.path.endswith("/chat/completions") else "anthropic"
        if not self.scripts[kind]:
            raise AssertionError(f"no scripted response left for {request.url.path}")
        response = self.scripts[kind].pop(0)
        if response.status_code != 200:
            return response
        streaming = bool(json.loads(request.content or b"{}").get("stream"))
        is_events = "event-stream" in response.headers.get("content-type", "")
        if kind == "chat" and not streaming and is_events:
            return _chat_as_json(response)
        if kind == "anthropic" and streaming and not is_events:
            return _anthropic_as_events(response)
        return response

    def bodies(self, kind: str) -> list[dict[str, Any]]:
        suffix = "/chat/completions" if kind == "chat" else "/messages"
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith(suffix)]

    def router(self) -> ModelRouter:
        direct = anthropic.AsyncAnthropic(api_key=DIRECT_KEY, max_retries=0,
                                          http_client=anthropic.DefaultAsyncHttpxClient(transport=self.transport))
        return ModelRouter(claude=lambda model, retry: AnthropicProvider(client=direct, retry_policy=retry or FAST),
                           others=VercelGateway(api_key=GATEWAY_KEY, transport=self.transport))


# ── one route per format ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Route:
    """One format: its model, its API's name in ``WireAPI``, and its replies."""

    name: str
    model: str
    api: str
    call_ids: tuple[str, str]
    overflow_message: str

    def provider(self, wire: WireAPI, retry: RetryPolicy = FAST) -> Any:
        return wire.router().provider_for(self.model, retry_policy=retry)

    def text(self, text: str) -> httpx.Response:
        if self.api == "anthropic":
            return anthropic_reply([{"type": "text", "text": text}], stop_reason="end_turn", model=self.model)
        return chat_stream(chat_chunk(content=text, model=self.model),
                           chat_chunk(finish="stop", usage={"prompt_tokens": 12, "completion_tokens": 3},
                                      model=self.model))

    def calls(self, *calls: tuple[str, str, dict[str, Any]]) -> httpx.Response:
        if self.api == "anthropic":
            return anthropic_reply([{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in calls],
                                   stop_reason="tool_use", model=self.model)
        raw = [{"index": k, "id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
               for k, (i, n, a) in enumerate(calls)]
        return chat_stream(chat_chunk(tool_calls=raw, model=self.model),
                           chat_chunk(finish="tool_calls", usage={"prompt_tokens": 20, "completion_tokens": 9},
                                      model=self.model))

    def error(self, status: int, message: str, kind: str) -> httpx.Response:
        if self.api == "anthropic":
            return httpx.Response(status, json={"type": "error", "error": {"type": kind, "message": message}})
        return httpx.Response(status, json={"error": {"message": message, "type": kind, "code": kind}})

    def rate_limited(self) -> httpx.Response:
        return self.error(429, "Too many requests.", "rate_limit_error" if self.api == "anthropic" else "rate_limited")

    def overloaded(self) -> httpx.Response:
        if self.api == "anthropic":
            return self.error(529, "Overloaded.", "overloaded_error")
        return self.error(503, "Service temporarily unavailable.", "service_unavailable_error")

    def overflow(self) -> httpx.Response:
        return self.error(400, self.overflow_message, "invalid_request_error")

    def tool_results(self, body: dict[str, Any]) -> dict[str, tuple[str, bool]]:
        """The tool results a request carried: call id → (text, marked as an error)."""
        if self.api == "anthropic":
            out = {}
            for message in body["messages"]:
                for block in message["content"] if isinstance(message["content"], list) else []:
                    if block.get("type") == "tool_result":
                        content = block.get("content")
                        text = content if isinstance(content, str) else "".join(
                            p.get("text", "") for p in content or [] if isinstance(p, dict))
                        out[block["tool_use_id"]] = (text, bool(block.get("is_error")))
            return out
        return {m["tool_call_id"]: (m["content"], m["content"].startswith("Error"))
                for m in body["messages"] if m["role"] == "tool"}


ROUTES = [
    Route("anthropic", "claude-sonnet-5-5", "anthropic", ("toolu_1", "toolu_2"),
          "prompt is too long: 250000 tokens > 200000 maximum"),
    Route("openai_chat", "google/gemini-2.5-flash", "chat", ("call_1", "call_2"),
          "This model's maximum context length is 1000000 tokens."),
]


class Sink:
    """A ``DeltaSink`` that keeps every frame; ``on_frame`` sees each one as it lands."""

    def __init__(self, on_frame: Callable[[Any], None] | None = None) -> None:
        self.deltas: list[Any] = []
        self.on_frame = on_frame

    def emit(self, delta: Any) -> None:
        self.deltas.append(delta)
        if self.on_frame is not None:
            self.on_frame(delta)

    def emit_meta(self, body: Any) -> None:
        pass


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(poll(), timeout)
