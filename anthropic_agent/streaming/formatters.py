import asyncio
from typing import Any, Literal, Callable, Awaitable
import json
import html
from anthropic.lib.streaming._beta_messages import BetaAsyncMessageStream
from anthropic.types.beta import BetaMessage

# Type alias for formatter names
FormatterType = Literal["xml", "raw", "json"]

# Maximum bytes for a single SSE data: line (prevents transport-layer fragmentation)
MAX_SSE_CHUNK_BYTES = 2048


def escape_xml_attr(value: str) -> str:
    """Escape a string for safe use as an XML attribute value.
    
    Uses html.escape to convert special characters to XML entities.
    
    Args:
        value: The string to escape
        
    Returns:
        The escaped string safe for use in XML attributes
    """
    return html.escape(str(value), quote=True)


async def stream_to_aqueue(chunk: Any, queue: asyncio.Queue) -> None:
    """Send a chunk to an async queue with simple SSE-safe escaping.
    
    Helper function to put items onto an async queue for SSE streaming.
    Only escapes actual newlines to prevent breaking SSE framing.
    
    Used by raw_formatter where JSON already has its own escaping.
    
    Args:
        chunk: The data chunk to send (can be any type)
        queue: The async queue to put the chunk into
    """
    if isinstance(chunk, str):
        chunk = chunk.replace('\n', '\\n')  # Only escape actual newlines
    await queue.put(chunk)


async def stream_xml_to_aqueue(chunk: Any, queue: asyncio.Queue) -> None:
    """Send a chunk to an async queue with double-escaping for XML format.
    
    Helper function for XML formatter that uses double-escaping to preserve
    literal backslashes in content:
    1. First escapes backslashes: \\ → \\\\
    2. Then escapes newlines: newline → \\n
    
    This ensures the frontend can distinguish:
    - \\n (two chars) → actual newline
    - \\\\n (three chars) → literal backslash-n
    
    Args:
        chunk: The data chunk to send (can be any type)
        queue: The async queue to put the chunk into
    """
    if isinstance(chunk, str):
        chunk = chunk.replace('\\', '\\\\')  # First: escape backslashes
        chunk = chunk.replace('\n', '\\n')   # Then: escape newlines
    await queue.put(chunk)


async def xml_formatter(
    stream: BetaAsyncMessageStream,
    queue: asyncio.Queue,
    stream_tool_results: bool = True
) -> BetaMessage:
    """Format Anthropic streaming response with custom XML tags.

    .. deprecated::
        The xml_formatter is deprecated. Use :func:`json_formatter` instead.
        It will be removed in a future release.

    This formatter wraps different content types in custom XML tags:
    - Thinking: <content-block-thinking>...</content-block-thinking>
    - Text: <content-block-text>...</content-block-text>
    - Client tool calls: <content-block-tool_call id="..." name="..." arguments="..."></content-block-tool_call>
    - Server tool calls: <content-block-server_tool_call id="..." name="..." arguments="..."></content-block-server_tool_call>
    - Server tool results: <content-block-server_tool_result id="..." name="..."><![CDATA[...]]></content-block-server_tool_result>
    - Errors: <content-block-error><![CDATA[...]]></content-block-error>
    
    Supported event types (per Anthropic spec):
    - message_start, message_delta, message_stop (handled gracefully)
    - content_block_start, content_block_delta, content_block_stop
    - ping (ignored)
    - error (propagated)
    
    Supported delta types:
    - text_delta (streamed as received)
    - thinking_delta (streamed as received)
    - signature_delta (captured but not streamed)
    - input_json_delta (buffered until complete)
    
    Supported content block types:
    - text
    - thinking
    - tool_use (client tools)
    - server_tool_use (server-side tools)
    - *_tool_result (any server tool result type)
    
    Args:
        stream: Anthropic streaming context manager
        queue: Async queue to send formatted output chunks
        stream_tool_results: Whether to stream server tool results (default: True)
        
    Returns:
        The final accumulated message from stream.get_final_message()
        
    Note:
        - Content blocks are tracked by index to properly open/close tags
        - Tool calls are buffered and streamed as attribute-based XML when complete
        - Tool results use CDATA sections for safe content wrapping
        - We process raw delta events, not simplified helper events to avoid duplication
    """
    # Track content blocks by index
    content_blocks = {}  # index -> {"type": str, "is_open": bool, "tool_data": dict, "signature": str}
    
    # Iterate through streaming events with async for to yield control to event loop
    async for event in stream:
        event_type = event.type
        
        # Message-level events (handle gracefully per Anthropic versioning policy)
        if event_type == "message_start":
            # Message initialization - contains empty message object
            continue
        
        elif event_type == "message_delta":
            # Top-level message changes (usage stats, stop_reason)
            # Not streamed to user, but available in final message
            continue
        
        elif event_type == "message_stop":
            # End of stream
            continue
        
        elif event_type == "ping":
            # Keep-alive ping - ignore
            continue
        
        elif event_type == "error":
            # Error event - propagate to user with CDATA wrapping
            error_data = getattr(event, 'error', event)
            error_content = json.dumps(error_data, default=str)
            await stream_xml_to_aqueue(
                f'<content-block-error><![CDATA[{error_content}]]></content-block-error>',
                queue
            )
            continue
        
        # Content block start
        elif event_type == "content_block_start":
            idx = event.index
            block_type = event.content_block.type
            content_blocks[idx] = {
                "type": block_type,
                "is_open": False,
                "tool_data": None,
                "signature": None
            }
            
            # Open appropriate tag based on content type
            if block_type == "thinking":
                await stream_xml_to_aqueue("<content-block-thinking>", queue)
                content_blocks[idx]["is_open"] = True
            
            elif block_type == "text":
                await stream_xml_to_aqueue("<content-block-text>", queue)
                content_blocks[idx]["is_open"] = True
            
            elif block_type == "tool_use":
                # Start buffering client tool call data
                content_blocks[idx]["tool_data"] = {
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input": {}
                }
            
            elif block_type == "server_tool_use":
                # Start buffering server tool call data (e.g., web_search, code_execution)
                content_blocks[idx]["tool_data"] = {
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input": {}
                }
            
            elif block_type.endswith("_tool_result"):
                # Handle all server tool result types (*_tool_result)
                # e.g., web_search_tool_result, bash_code_execution_tool_result, 
                # text_editor_code_execution_tool_result, etc.
                content_blocks[idx]["tool_data"] = {
                    "tool_use_id": getattr(event.content_block, "tool_use_id", ""),
                    "content": getattr(event.content_block, "content", None)
                }
        
        # Content block deltas
        elif event_type == "content_block_delta":
            idx = event.index
            delta = event.delta
            block_info = content_blocks.get(idx)
            
            if not hasattr(delta, 'type'):
                continue
            
            delta_type = delta.type
            
            # Text delta - stream immediately
            if delta_type == "text_delta":
                text_content = getattr(delta, 'text', '')
                if text_content:
                    await stream_xml_to_aqueue(text_content, queue)
            
            # Thinking delta - stream immediately
            elif delta_type == "thinking_delta":
                thinking_content = getattr(delta, 'thinking', '')
                if thinking_content:
                    await stream_xml_to_aqueue(thinking_content, queue)
            
            # Signature delta - buffer (sent before content_block_stop for thinking)
            elif delta_type == "signature_delta":
                if block_info:
                    block_info["signature"] = getattr(delta, 'signature', '')
            
            # Input JSON delta - buffer for tool_use/server_tool_use
            elif delta_type == "input_json_delta":
                if block_info and block_info["tool_data"]:
                    partial_json = getattr(delta, 'partial_json', '')
                    if "input_json" not in block_info["tool_data"]:
                        block_info["tool_data"]["input_json"] = ""
                    block_info["tool_data"]["input_json"] += partial_json
        
        # NOTE: We skip simplified event types (text, thinking, input_json, citation, signature)
        # because we already process them via content_block_delta events above.
        # Processing both would cause duplication in the output.
        
        # Content block stop
        elif event_type == "content_block_stop":
            idx = event.index
            block_info = content_blocks.get(idx)
            
            if not block_info:
                continue
            
            block_type = block_info["type"]
            
            # Close text/thinking blocks
            if block_info["is_open"]:
                if block_type == "thinking":
                    await stream_xml_to_aqueue("</content-block-thinking>", queue)
                elif block_type == "text":
                    await stream_xml_to_aqueue("</content-block-text>", queue)
                    # Emit citations from content_block if present
                    content_block = getattr(event, 'content_block', None)
                    citations = getattr(content_block, 'citations', None) if content_block else None
                    if citations:
                        citations_xml = '<citations>'
                        for cit in citations:
                            cit_dict = cit.model_dump() if hasattr(cit, 'model_dump') else cit
                            cit_type = escape_xml_attr(cit_dict.get("type", "unknown"))
                            doc_idx = escape_xml_attr(str(cit_dict.get("document_index", "")))
                            doc_title = escape_xml_attr(cit_dict.get("document_title") or "")
                            cited_text = cit_dict.get("cited_text", "")
                            # Build attributes based on citation type
                            if cit_dict.get("type") == "char_location":
                                start = escape_xml_attr(str(cit_dict.get("start_char_index", "")))
                                end = escape_xml_attr(str(cit_dict.get("end_char_index", "")))
                                citations_xml += f'<citation type="{cit_type}" document_index="{doc_idx}" document_title="{doc_title}" start_char_index="{start}" end_char_index="{end}"><![CDATA[{cited_text}]]></citation>'
                            elif cit_dict.get("type") == "page_location":
                                start_page = escape_xml_attr(str(cit_dict.get("start_page_number", "")))
                                end_page = escape_xml_attr(str(cit_dict.get("end_page_number", "")))
                                citations_xml += f'<citation type="{cit_type}" document_index="{doc_idx}" document_title="{doc_title}" start_page_number="{start_page}" end_page_number="{end_page}"><![CDATA[{cited_text}]]></citation>'
                            elif cit_dict.get("type") == "web_search_result_location":
                                url = escape_xml_attr(cit_dict.get("url") or "")
                                title = escape_xml_attr(cit_dict.get("title") or "")
                                citations_xml += f'<citation type="{cit_type}" url="{url}" title="{title}"><![CDATA[{cited_text}]]></citation>'
                            else:
                                citations_xml += f'<citation type="{cit_type}" document_index="{doc_idx}"><![CDATA[{cited_text}]]></citation>'
                        citations_xml += '</citations>'
                        await stream_xml_to_aqueue(citations_xml, queue)
                block_info["is_open"] = False
            
            # Stream complete tool call (client tools)
            elif block_type == "tool_use" and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                try:
                    # Parse the accumulated input JSON
                    if "input_json" in tool_data:
                        tool_data["input"] = json.loads(tool_data["input_json"])
                        del tool_data["input_json"]
                except json.JSONDecodeError:
                    # If parsing fails, keep as string
                    if "input_json" in tool_data:
                        tool_data["input"] = tool_data["input_json"]
                        del tool_data["input_json"]
                
                # Format as attribute-based XML
                tool_id = escape_xml_attr(tool_data["id"])
                tool_name = escape_xml_attr(tool_data["name"])
                arguments = escape_xml_attr(json.dumps(tool_data["input"]))
                await stream_xml_to_aqueue(
                    f'<content-block-tool_call id="{tool_id}" name="{tool_name}" arguments="{arguments}"></content-block-tool_call>',
                    queue
                )
            
            # Stream complete server tool call (e.g., web_search, code_execution)
            elif block_type == "server_tool_use" and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                try:
                    # Parse the accumulated input JSON
                    if "input_json" in tool_data:
                        tool_data["input"] = json.loads(tool_data["input_json"])
                        del tool_data["input_json"]
                except json.JSONDecodeError:
                    if "input_json" in tool_data:
                        tool_data["input"] = tool_data["input_json"]
                        del tool_data["input_json"]
                
                # Format as attribute-based XML (distinct from client tools)
                tool_id = escape_xml_attr(tool_data["id"])
                tool_name = escape_xml_attr(tool_data["name"])
                arguments = escape_xml_attr(json.dumps(tool_data["input"]))
                await stream_xml_to_aqueue(
                    f'<content-block-server_tool_call id="{tool_id}" name="{tool_name}" arguments="{arguments}"></content-block-server_tool_call>',
                    queue
                )
            
            # Stream server tool result (handles all *_tool_result types)
            elif block_type.endswith("_tool_result") and block_info["tool_data"]:
                # Only stream tool results if stream_tool_results is True
                if stream_tool_results:
                    tool_data = block_info["tool_data"]
                    tool_use_id = escape_xml_attr(tool_data["tool_use_id"])
                    result_name = escape_xml_attr(block_type)  # Use block type as name
                    
                    # Serialize content to string for CDATA
                    content = tool_data.get("content")
                    if content is None:
                        content_str = ""
                    elif isinstance(content, str):
                        content_str = content
                    else:
                        content_str = json.dumps(content, default=str)
                    
                    await stream_xml_to_aqueue(
                        f'<content-block-server_tool_result id="{tool_use_id}" name="{result_name}"><![CDATA[{content_str}]]></content-block-server_tool_result>',
                        queue
                    )
        
        # Handle unknown event types gracefully (per Anthropic versioning policy)
        else:
            # Unknown event type - log but don't fail
            pass
    
    # Safety: close any open blocks at the end
    for idx, block_info in content_blocks.items():
        if block_info.get("is_open"):
            if block_info["type"] == "thinking":
                await stream_xml_to_aqueue("</content-block-thinking>", queue)
            elif block_info["type"] == "text":
                await stream_xml_to_aqueue("</content-block-text>", queue)
            block_info["is_open"] = False
    
    # Get and return the final accumulated message (async method)
    return await stream.get_final_message()


# ---------------------------------------------------------------------------
# JSON Envelope helpers
# ---------------------------------------------------------------------------

def _build_envelope(
    msg_type: str,
    agent: str,
    final: bool,
    delta: str,
    **extra: Any,
) -> str:
    """Build a JSON envelope string from base fields plus optional extras.

    Returns a compact JSON string (no trailing newline) ready for SSE emission.
    """
    obj: dict[str, Any] = {
        "type": msg_type,
        "agent": agent,
        "final": final,
        "delta": delta,
    }
    obj.update(extra)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _utf8_safe_split(payload_bytes: bytes, max_bytes: int) -> list[bytes]:
    """Split *payload_bytes* into chunks of at most *max_bytes* each.

    Never splits inside a multi-byte UTF-8 character.
    """
    chunks: list[bytes] = []
    offset = 0
    length = len(payload_bytes)
    while offset < length:
        end = min(offset + max_bytes, length)
        if end < length:
            # Back up if we landed on a continuation byte (0b10xxxxxx)
            while end > offset and (payload_bytes[end] & 0xC0) == 0x80:
                end -= 1
        chunks.append(payload_bytes[offset:end])
        offset = end
    return chunks


async def _chunk_and_emit(
    queue: asyncio.Queue,
    msg_type: str,
    agent: str,
    payload: str,
    final_on_last: bool,
    **extra: Any,
) -> None:
    """Proactively chunk *payload* and emit JSON envelopes to *queue*.

    If *final_on_last* is True the last emitted chunk carries ``final=true``. Used for buffered blocks.
    """
    # Measure overhead (envelope with empty delta)
    overhead = len(_build_envelope(msg_type, agent, False, "", **extra).encode("utf-8"))
    max_delta_bytes = MAX_SSE_CHUNK_BYTES - overhead
    if max_delta_bytes < 64:
        # Degenerate case – just emit full payload in one message
        # TODO: Won't happen in real world but insert a warning log here.
        max_delta_bytes = MAX_SSE_CHUNK_BYTES

    payload_bytes = payload.encode("utf-8")

    if len(payload_bytes) <= max_delta_bytes:
        # Fits in a single message
        env = _build_envelope(msg_type, agent, final_on_last, payload, **extra)
        await queue.put(env)
        return

    chunks = _utf8_safe_split(payload_bytes, max_delta_bytes)
    for i, chunk_bytes in enumerate(chunks):
        is_last = i == len(chunks) - 1
        delta_str = chunk_bytes.decode("utf-8")
        env = _build_envelope(
            msg_type, agent,
            final=(final_on_last and is_last),
            delta=delta_str,
            **extra,
        )
        await queue.put(env)


# ---------------------------------------------------------------------------
# JSON Envelope formatter
# ---------------------------------------------------------------------------

async def json_formatter(
    stream: BetaAsyncMessageStream,
    queue: asyncio.Queue,
    stream_tool_results: bool = True,
    agent_uuid: str = "",
) -> BetaMessage:
    """Format Anthropic streaming response as self-contained JSON envelopes.

    Each emitted message is a compact JSON object containing full routing
    context (type, agent, final, delta) as specified in
    ``new_streaming_paradigm.md``.

    Messages arrive in strict order over SSE, so the consumer can reconstruct
    content by appending deltas in arrival order. The ``final`` flag marks
    the last message for a given content block.

    Args:
        stream: Anthropic streaming context manager.
        queue: Async queue to send formatted output chunks.
        stream_tool_results: Whether to stream server tool results (default True).
        agent_uuid: UUID of the emitting agent (stamped on every message).

    Returns:
        The final accumulated message from ``stream.get_final_message()``.
    """
    # Per-block type tracking keyed by Anthropic content-block index
    block_types: dict[int, str] = {}
    # Buffered tool data keyed by Anthropic content-block index
    tool_buffers: dict[int, dict[str, Any]] = {}

    async for event in stream:
        event_type = event.type

        # ---- Message-level events (skip) ----
        if event_type in ("message_start", "message_delta", "message_stop", "ping"):
            continue

        # ---- Error ----
        if event_type == "error":
            error_data = getattr(event, "error", event)
            error_payload = json.dumps(error_data, default=str)
            await _chunk_and_emit(queue, "error", agent_uuid, error_payload, final_on_last=True)
            continue

        # ---- Content block start ----
        if event_type == "content_block_start":
            api_idx = event.index
            block_type: str = event.content_block.type
            block_types[api_idx] = block_type

            if block_type in ("thinking", "text"):
                # No opening tag emitted – deltas will carry content
                pass
            elif block_type == "tool_use":
                tool_buffers[api_idx] = {
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input_json": "",
                }
            elif block_type == "server_tool_use":
                tool_buffers[api_idx] = {
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input_json": "",
                }
            elif block_type.endswith("_tool_result"):
                tool_buffers[api_idx] = {
                    "tool_use_id": getattr(event.content_block, "tool_use_id", ""),
                    "content": getattr(event.content_block, "content", None),
                    "block_type": block_type,
                }
            continue

        # ---- Content block delta ----
        if event_type == "content_block_delta":
            api_idx = event.index
            delta = event.delta
            if api_idx not in block_types or not hasattr(delta, "type"):
                continue

            delta_type = delta.type

            if delta_type == "text_delta":
                text_content = getattr(delta, "text", "")
                if text_content:
                    env = _build_envelope("text", agent_uuid, False, text_content)
                    await queue.put(env)

            elif delta_type == "thinking_delta":
                thinking_content = getattr(delta, "thinking", "")
                if thinking_content:
                    env = _build_envelope("thinking", agent_uuid, False, thinking_content)
                    await queue.put(env)

            elif delta_type == "signature_delta":
                # Captured but not streamed
                pass

            elif delta_type == "input_json_delta":
                buf = tool_buffers.get(api_idx)
                if buf is not None:
                    partial = getattr(delta, "partial_json", "")
                    buf["input_json"] = buf.get("input_json", "") + partial
            continue

        # ---- Content block stop ----
        if event_type == "content_block_stop":
            api_idx = event.index
            bt = block_types.get(api_idx, "")
            if not bt:
                continue

            # --- Streamed blocks: emit empty-delta final marker ---
            if bt in ("thinking", "text"):
                env = _build_envelope(bt, agent_uuid, True, "")
                await queue.put(env)

                # Emit citations after text final marker
                if bt == "text":
                    content_block = getattr(event, "content_block", None)
                    citations = getattr(content_block, "citations", None) if content_block else None
                    if citations:
                        for i, cit in enumerate(citations):
                            cit_dict = cit.model_dump() if hasattr(cit, "model_dump") else cit
                            cited_text = cit_dict.get("cited_text", "")
                            extras: dict[str, Any] = {
                                "citation_type": cit_dict.get("type", "unknown"),
                            }
                            # Add type-specific fields
                            for key in (
                                "document_index", "document_title",
                                "start_char_index", "end_char_index",
                                "start_page_number", "end_page_number",
                                "url", "title",
                            ):
                                if key in cit_dict and cit_dict[key] is not None:
                                    extras[key] = cit_dict[key]
                            is_last_citation = i == len(citations) - 1
                            env = _build_envelope(
                                "citation", agent_uuid,
                                is_last_citation, cited_text, **extras,
                            )
                            await queue.put(env)

            # --- Buffered blocks: tool_use ---
            elif bt == "tool_use":
                buf = tool_buffers.pop(api_idx, None)
                if buf:
                    try:
                        parsed_input = json.loads(buf["input_json"]) if buf["input_json"] else {}
                    except json.JSONDecodeError:
                        parsed_input = buf["input_json"]
                    payload = json.dumps(parsed_input, ensure_ascii=False)
                    await _chunk_and_emit(
                        queue, "tool_call", agent_uuid, payload,
                        final_on_last=True, id=buf["id"], name=buf["name"],
                    )

            # --- Buffered blocks: server_tool_use ---
            elif bt == "server_tool_use":
                buf = tool_buffers.pop(api_idx, None)
                if buf:
                    try:
                        parsed_input = json.loads(buf["input_json"]) if buf["input_json"] else {}
                    except json.JSONDecodeError:
                        parsed_input = buf["input_json"]
                    payload = json.dumps(parsed_input, ensure_ascii=False)
                    await _chunk_and_emit(
                        queue, "server_tool_call", agent_uuid, payload,
                        final_on_last=True, id=buf["id"], name=buf["name"],
                    )

            # --- Buffered blocks: *_tool_result ---
            elif bt.endswith("_tool_result"):
                if stream_tool_results:
                    buf = tool_buffers.pop(api_idx, None)
                    if buf:
                        content = buf.get("content")
                        if content is None:
                            content_str = ""
                        elif isinstance(content, str):
                            content_str = content
                        else:
                            content_str = json.dumps(content, default=str)
                        result_name = buf.get("block_type", bt)
                        await _chunk_and_emit(
                            queue, "server_tool_result", agent_uuid,
                            content_str, final_on_last=True,
                            id=buf["tool_use_id"], name=result_name,
                        )
                else:
                    tool_buffers.pop(api_idx, None)
            continue

        # Handle unknown event types gracefully
        pass

    return await stream.get_final_message()


async def raw_formatter(
    stream: BetaAsyncMessageStream,
    queue: asyncio.Queue,
    stream_tool_results: bool = True
) -> BetaMessage:
    """Format Anthropic streaming response with raw/minimal formatting.

    .. deprecated::
        The raw_formatter is deprecated. Use :func:`json_formatter` instead.
        It will be removed in a future release.

    This formatter streams raw Anthropic events with minimal processing.
    When stream_tool_results is False, all events related to *_tool_result blocks
    are filtered out (content_block_start, content_block_delta, content_block_stop).
    
    Args:
        stream: Anthropic streaming context manager
        queue: Async queue to send formatted output chunks
        stream_tool_results: Whether to stream tool results (default: True).
            When False, content_block events for *_tool_result types are skipped.
        
    Returns:
        The final accumulated message from stream.get_final_message()
    """
    # Track block indices that are tool results (to skip their deltas and stops)
    tool_result_block_indices: set[int] = set()
    
    async for event in stream:
        event_type = event.type
        
        # Only process relevant event types
        if event_type not in ['message_start', 'message_delta', 'message_stop', 
                              'content_block_start', 'content_block_delta', 'content_block_stop']:
            continue
        
        # Handle tool result filtering when stream_tool_results is False
        if not stream_tool_results:
            if event_type == 'content_block_start':
                idx = getattr(event, 'index', None)
                block_type = getattr(event.content_block, 'type', '')
                # Track this block if it's a tool result type
                if block_type.endswith('_tool_result'):
                    if idx is not None:
                        tool_result_block_indices.add(idx)
                    continue  # Skip streaming this event
            
            elif event_type == 'content_block_delta':
                idx = getattr(event, 'index', None)
                if idx in tool_result_block_indices:
                    continue  # Skip deltas for tool result blocks
            
            elif event_type == 'content_block_stop':
                idx = getattr(event, 'index', None)
                if idx in tool_result_block_indices:
                    # Clean up tracking and skip this event
                    tool_result_block_indices.discard(idx)
                    continue
        
        # Stream the event
        event_str = event.model_dump_json(warnings=False)
        await stream_to_aqueue(event_str, queue)

    # Return final message
    return await stream.get_final_message()


# Formatter registry mapping string names to formatter functions
FORMATTERS: dict[str, Callable[..., Awaitable[BetaMessage]]] = {
    "xml": xml_formatter,
    "raw": raw_formatter,
    "json": json_formatter,
}


def get_formatter(name: FormatterType) -> Callable[..., Awaitable[BetaMessage]]:
    """Get a formatter function by name.

    Args:
        name: Formatter name ("json", "xml", or "raw"). The "xml" and "raw"
            formatters are deprecated; prefer "json".

    Returns:
        The formatter function

    Raises:
        ValueError: If formatter name is not recognized
    """
    if name not in FORMATTERS:
        raise ValueError(
            f"Unknown formatter '{name}'. Available formatters: {list(FORMATTERS.keys())}"
        )
    if name in ("xml", "raw"):
        import warnings
        warnings.warn(
            f"The '{name}' formatter is deprecated. Use 'json' instead. "
            "It will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    return FORMATTERS[name]

