"""Scripted stand-ins for the Gemini and Anthropic HTTP APIs, as ``httpx``
mock transports: each request takes the next scripted response for its route,
and every request is kept for the test to inspect."""
from __future__ import annotations

import json
from typing import Any

import httpx


def gemini_sse(*chunks: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def gemini_chunk(*parts: dict[str, Any], finish: str | None = None, usage: dict | None = None) -> dict:
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": list(parts)}}
    if finish:
        candidate["finishReason"] = finish
    chunk: dict[str, Any] = {"candidates": [candidate], "modelVersion": "gemini-3.8-flash"}
    if usage:
        chunk["usageMetadata"] = usage
    return chunk


def gemini_text_reply(text: str) -> httpx.Response:
    return gemini_sse(gemini_chunk({"text": text}, finish="STOP",
                                   usage={"promptTokenCount": 12, "candidatesTokenCount": 3}))


def gemini_error(status: int, message: str, *, code_status: str = "", retry_delay: str | None = None) -> httpx.Response:
    error: dict[str, Any] = {"code": status, "message": message, "status": code_status}
    if retry_delay:
        error["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    return httpx.Response(status, json={"error": error})


def anthropic_sse(blocks: list[dict[str, Any]], *, stop_reason: str, model: str = "claude-haiku-5-5") -> httpx.Response:
    """A Messages API stream with ``blocks`` (``{"type": "text", "text": ...}``
    or ``{"type": "tool_use", "id", "name", "input"}``)."""
    events: list[tuple[str, dict[str, Any]]] = [("message_start", {
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1}},
    })]
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                   "content_block": {"type": "text", "text": ""}}))
            events.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "text_delta", "text": block["text"]}}))
        else:
            events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {
                "type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}))
            events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(block["input"])}}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                     "usage": {"output_tokens": 20}}))
    events.append(("message_stop", {"type": "message_stop"}))
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def anthropic_message(blocks: list[dict[str, Any]], *, stop_reason: str,
                      model: str = "claude-haiku-5-5") -> httpx.Response:
    """A non-streaming Messages API response with ``blocks``."""
    return httpx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": blocks,
        "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 20},
    })


class ScriptedAPI:
    """Routes each request to the next scripted response for its API."""

    def __init__(self, *, gemini: list[httpx.Response] | None = None,
                 anthropic: list[httpx.Response] | None = None) -> None:
        self.gemini = list(gemini or [])
        self.anthropic = list(anthropic or [])
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if ":streamGenerateContent" in path or ":generateContent" in path:
            script = self.gemini
        elif path.endswith("/messages"):
            script = self.anthropic
        else:
            return httpx.Response(404, json={"error": {"message": f"no route for {path}"}})
        if not script:
            raise AssertionError(f"no scripted response left for {path}")
        response = script.pop(0)
        if path.endswith(":generateContent") and "event-stream" in response.headers.get("content-type", ""):
            return _sse_as_one_response(response)
        return response

    def bodies(self, kind: str) -> list[dict[str, Any]]:
        """The JSON bodies sent to ``kind`` ("gemini" or "anthropic"), in order."""
        def is_kind(r: httpx.Request) -> bool:
            gemini = "generatecontent" in r.url.path.lower()
            return gemini if kind == "gemini" else r.url.path.endswith("/messages")
        return [json.loads(r.content) for r in self.requests if is_kind(r)]


def _sse_as_one_response(response: httpx.Response) -> httpx.Response:
    """A scripted stream as the one JSON response ``generateContent`` returns."""
    parts: list[dict[str, Any]] = []
    merged: dict[str, Any] = {}
    for line in response.content.decode().splitlines():
        if not line.startswith("data:"):
            continue
        chunk = json.loads(line[len("data:"):])
        if "error" in chunk:
            return httpx.Response(chunk["error"].get("code", 500), json=chunk)
        candidate = (chunk.get("candidates") or [{}])[0]
        parts.extend((candidate.get("content") or {}).get("parts") or [])
        if candidate.get("finishReason"):
            merged["finishReason"] = candidate["finishReason"]
        for key in ("usageMetadata", "modelVersion"):
            if key in chunk:
                merged[key] = chunk[key]
    body: dict[str, Any] = {"candidates": [{"content": {"role": "model", "parts": parts}}]}
    if "finishReason" in merged:
        body["candidates"][0]["finishReason"] = merged.pop("finishReason")
    body.update(merged)
    return httpx.Response(200, json=body)


class RecordingSink:
    def __init__(self) -> None:
        self.deltas: list[Any] = []
        self.metas: list[Any] = []

    def emit(self, delta: Any) -> None:
        self.deltas.append(delta)

    def emit_meta(self, body: Any) -> None:
        self.metas.append(body)
