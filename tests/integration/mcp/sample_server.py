"""A tiny self-contained MCP server authored with agent_base primitives.

Used by ``test_server_roundtrip.py``: imported for its ``SAMPLE`` bundle (the
in-memory test) and launched as a subprocess ``__main__`` (the stdio test). Kept
free of any network/credentials so the round-trip is deterministic. NOT a test
module itself (no ``test_`` prefix → pytest won't collect it).
"""
from __future__ import annotations

from agent_base.tools import ToolBundle, tool


@tool
async def echo_struct(text: str, times: int = 1) -> dict:
    """Echo text back as structured data.

    Args:
        text: the text to echo
        times: how many times to repeat it
    """
    return {"echo": text * times, "n": times}


@tool
async def kaboom(why: str) -> dict:
    """Always raises — exercises the error path across the wire.

    Args:
        why: reason for the failure
    """
    raise RuntimeError(f"boom: {why}")


SAMPLE = ToolBundle("sample", [echo_struct, kaboom])


if __name__ == "__main__":
    import asyncio

    from agent_base.mcp import serve_stdio

    asyncio.run(serve_stdio(SAMPLE, name="sample"))
