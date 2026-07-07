"""Dynamic registration + declarative reconcile (mcp.md §3/§7; MC-D8/MC-D14, E14).

add/remove on a live source, duplicate rejection, needs_auth-on-add (the
registration still succeeds), the reconcile diff matrix (add-only /
remove-only / mixed / no-op / unchanged-key-spec-change), diff + notice
semantics (boot baseline emits nothing; MC-D13 notice queues on change).
"""
from __future__ import annotations

import pytest

from agent_base.mcp import McpHttpSpec, McpServerSpec
from agent_base.mcp.source import McpToolSource

from ._fakes import JsonHttpMcpHandler, make_fastmcp, use_fake_server


def _spec() -> McpServerSpec:
    return McpServerSpec(transport=McpHttpSpec(url="http://fake/mcp"))


async def _started(monkeypatch, servers) -> McpToolSource:
    use_fake_server(monkeypatch)
    source = McpToolSource(servers)
    await source.start()
    return source


async def test_add_server_connects_and_flags_surface_change(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec()})
    changes = []
    source.on_surface_changed = lambda: changes.append(True)
    try:
        status = await source.add_server("b", _spec())
        assert status.state == "connected"
        assert "b" in source.current_surface()
        assert changes  # boundary discipline notified
    finally:
        await source.aclose()


async def test_add_duplicate_key_raises_value_error(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec()})
    try:
        with pytest.raises(ValueError):
            await source.add_server("a", _spec())
        with pytest.raises(ValueError):
            await source.add_server("bad__key", _spec())
    finally:
        await source.aclose()


async def test_add_with_401_parks_needs_auth_but_registration_succeeds(monkeypatch):
    use_fake_server(monkeypatch, passthrough=("locked",))
    source = McpToolSource({"a": _spec()})
    await source.start()
    handler = JsonHttpMcpHandler(require_token="nope")
    locked = McpServerSpec(
        transport=McpHttpSpec(
            url="http://fake/mcp", httpx_transport_factory=handler.transport_factory()
        )
    )
    try:
        status = await source.add_server("locked", locked)
        assert status.state == "needs_auth"
        assert status.auth_challenge is not None
        # visible in statuses() and recoverable — the registration held
        assert {s.name for s in source.statuses()} == {"a", "locked"}
    finally:
        await source.aclose()


async def test_remove_server_drops_handle_and_tools(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec(), "b": _spec()})
    try:
        await source.remove_server("b")
        assert {s.name for s in source.statuses()} == {"a"}
        assert "b" not in source.current_surface()
        with pytest.raises(KeyError):
            await source.remove_server("b")
    finally:
        await source.aclose()


async def test_boot_commit_is_baseline_no_notice(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec()})
    try:
        diff = source.commit_applied()
        assert diff.is_empty()  # boot surface is NOT a change (MC-D13)
        assert source.consume_pending_notice() is None
    finally:
        await source.aclose()


async def test_change_after_baseline_queues_exactly_one_notice(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec()})
    try:
        source.commit_applied()  # baseline
        await source.add_server("b", _spec())
        diff = source.commit_applied()
        assert set(diff.added) == {"b"} and not diff.removed
        notice = source.consume_pending_notice()
        assert notice is not None and "+ b connected" in notice
        assert source.consume_pending_notice() is None  # consumed exactly once
        # a no-op re-commit queues nothing
        assert source.commit_applied().is_empty()
        assert source.consume_pending_notice() is None
    finally:
        await source.aclose()


async def test_removal_notice_tells_model_not_to_claim_access(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec(), "b": _spec()})
    try:
        source.commit_applied()
        await source.remove_server("b")
        source.commit_applied()
        notice = source.consume_pending_notice()
        assert "- b disconnected" in notice and "do not claim access" in notice
    finally:
        await source.aclose()


# ── reconcile (MC-D14) diff matrix ───────────────────────────────────


async def test_reconcile_add_only(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec()})
    try:
        statuses = await source.reconcile({"a": _spec(), "b": _spec()})
        assert {s.name for s in statuses} == {"a", "b"}
        assert source._handles["b"].state == "connected"
    finally:
        await source.aclose()


async def test_reconcile_remove_only_and_mixed(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec(), "b": _spec()})
    try:
        statuses = await source.reconcile({"a": _spec(), "c": _spec()})
        assert {s.name for s in statuses} == {"a", "c"}
    finally:
        await source.aclose()


async def test_reconcile_noop_notifies_nothing(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec()})
    changes = []
    source.on_surface_changed = lambda: changes.append(True)
    events = []
    source.on_event = lambda body: events.append(body)
    try:
        await source.reconcile({"a": _spec()})
        assert changes == [] and events == []  # zero frames, zero notices
    finally:
        await source.aclose()


async def test_reconcile_unchanged_key_with_changed_spec_is_untouched(monkeypatch):
    """Keys are the identity (MC-D14): a different spec under an existing
    key does NOT reconnect or replace the handle."""
    source = await _started(monkeypatch, {"a": _spec()})
    original_handle = source._handles["a"]
    try:
        changed = McpServerSpec(
            transport=McpHttpSpec(url="http://elsewhere/mcp"), tool_timeout_s=5.0
        )
        await source.reconcile({"a": changed})
        assert source._handles["a"] is original_handle
        assert source._handles["a"].spec.tool_timeout_s == 60.0
    finally:
        await source.aclose()


async def test_reconcile_to_empty_removes_everything(monkeypatch):
    source = await _started(monkeypatch, {"a": _spec(), "b": _spec()})
    try:
        statuses = await source.reconcile({})
        assert statuses == []
        assert source.statuses() == []
        assert source.current_surface() == {}
    finally:
        await source.aclose()
