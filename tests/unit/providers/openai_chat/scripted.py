"""Scripted stand-ins for the Chat Completions and Anthropic Messages APIs, as
``httpx`` mock transports: each request takes the next scripted response for its
route, and every request is kept for the test to inspect."""
from __future__ import annotations

import json
from typing import Any

import httpx


# -- Chat Completions --------------------------------------------------------------


def chunk(*, content: str | None = None, reasoning: str | None = None, reasoning_details: list | None = None,
          tool_calls: list | None = None, finish: str | None = None, usage: dict | None = None,
          gen_id: str = "gen_1", model: str = "zai/glm-5.3-flash") -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    if reasoning_details is not None:
        delta["reasoning_details"] = reasoning_details
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    data: dict[str, Any] = {"id": gen_id, "object": "chat.completion.chunk", "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        data["usage"] = usage
    return data


def chat_sse(*chunks: dict[str, Any], done: bool = True) -> httpx.Response:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + ("data: [DONE]\n\n" if done else "")
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def chat_text_reply(text: str) -> httpx.Response:
    return chat_sse(chunk(content=text), chunk(finish="stop", usage={"prompt_tokens": 12, "completion_tokens": 3}))


def chat_error(status: int, message: str, *, error_type: str = "", retry_after: str | None = None) -> httpx.Response:
    headers = {"retry-after": retry_after} if retry_after else {}
    return httpx.Response(status, headers=headers,
                          json={"error": {"message": message, "type": error_type, "code": error_type}})


def _sse_as_one_response(response: httpx.Response) -> httpx.Response:
    """A scripted stream as the one JSON response a non-streaming request gets."""
    message: dict[str, Any] = {"role": "assistant", "content": None}
    calls: dict[int, dict[str, Any]] = {}
    out: dict[str, Any] = {"object": "chat.completion"}
    finish = None
    for line in response.content.decode().splitlines():
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        data = json.loads(line[len("data:"):])
        if "error" in data:
            return httpx.Response(int(data["error"].get("code") or 500), json=data)
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
                if raw.get("id"):
                    call["id"] = raw["id"]
                fn = raw.get("function") or {}
                if fn.get("name"):
                    call["function"]["name"] = fn["name"]
                call["function"]["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    out["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
    return httpx.Response(200, json=out)


# -- Anthropic Messages ------------------------------------------------------------


def anthropic_message(blocks: list[dict[str, Any]], *, stop_reason: str,
                      model: str = "claude-sonnet-5-5") -> httpx.Response:
    """A non-streaming Messages API response with ``blocks``."""
    return httpx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": blocks,
        "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 20},
    })


# -- the router ------------------------------------------------------------------------


class ScriptedAPI:
    """Routes each request to the next scripted response for its API."""

    def __init__(self, *, chat: list[httpx.Response] | None = None,
                 anthropic: list[httpx.Response] | None = None) -> None:
        self.chat = list(chat or [])
        self.anthropic = list(anthropic or [])
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/chat/completions"):
            if not self.chat:
                raise AssertionError(f"no scripted response left for {path}")
            response = self.chat.pop(0)
            streaming = json.loads(request.content or b"{}").get("stream", False)
            if not streaming and "event-stream" in response.headers.get("content-type", ""):
                return _sse_as_one_response(response)
            return response
        if path.endswith("/messages"):
            if not self.anthropic:
                raise AssertionError(f"no scripted response left for {path}")
            return self.anthropic.pop(0)
        return httpx.Response(404, json={"error": {"message": f"no route for {path}"}})

    def bodies(self, kind: str) -> list[dict[str, Any]]:
        """The JSON bodies sent to ``kind`` ("chat" or "anthropic"), in order."""
        suffix = "/chat/completions" if kind == "chat" else "/messages"
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith(suffix)]


class RecordingSink:
    def __init__(self) -> None:
        self.deltas: list[Any] = []
        self.metas: list[Any] = []

    def emit(self, delta: Any) -> None:
        self.deltas.append(delta)

    def emit_meta(self, body: Any) -> None:
        self.metas.append(body)
