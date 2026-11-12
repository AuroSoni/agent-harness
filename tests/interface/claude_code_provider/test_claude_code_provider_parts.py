"""Interface spec: the Claude Code provider's parts — config, the CLI's environment, the request fields, the
transcript store and ledger, the process slots, the plan's meter and the URL guard.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk.testing import run_session_store_conformance

from agent_base.core.messages import Message
from agent_base.providers.anthropic import AnthropicLLMConfig
from agent_base.providers.anthropic.formatters import AnthropicMessageFormatter
from agent_base.providers.claude_code import (DEFAULT_SECRET_PATTERNS, ClaudeCodeConfig, ClaudeCodeConfigError,
                                              MemoryTranscriptLedger, MemoryTranscriptStore, PlanMeter, ProcessSlots,
                                              TranscriptLedger, TranscriptStore, check_llm_config, cli_env,
                                              make_public_url_guard, prefix_fingerprints, prune_prompt_files,
                                              web_tools_for)


def test_config_defaults_and_validation(tmp_path):
    cfg = ClaudeCodeConfig(workdir=tmp_path / "cc")
    assert cfg.workdir == (tmp_path / "cc").resolve() and cfg.workdir.is_absolute()
    assert cfg.server_name == "agent_base" and cfg.tool_prefix == "mcp__agent_base__"
    assert cfg.fallback == "direct" and cfg.max_processes == 4 and cfg.max_child_processes == 4
    with pytest.raises(ValueError):
        ClaudeCodeConfig(workdir=tmp_path, fallback="sometimes")
    with pytest.raises(ValueError):
        ClaudeCodeConfig(workdir=tmp_path, server_name="has spaces")
    with pytest.raises(ValueError):
        ClaudeCodeConfig(workdir=tmp_path, max_processes=0)
    with pytest.raises(ValueError):
        ClaudeCodeConfig(workdir=tmp_path, max_child_processes=0)


def test_the_clis_environment_carries_no_secret_but_the_login(monkeypatch):
    for name in ("SOME_API_KEY", "DATABASE_URL", "AWS_REGION_X", "E2B_THING", "CLAUDE_CODE_OAUTH_TOKEN", "PLAIN"):
        monkeypatch.setenv(name, "x")
    env = cli_env(max_tokens=64000)
    assert env["SOME_API_KEY"] == env["DATABASE_URL"] == env["AWS_REGION_X"] == env["E2B_THING"] == ""
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env and "PLAIN" not in env
    assert env["ANTHROPIC_API_KEY"] == "" and env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"
    monkeypatch.setenv("VENDOR_ONLY", "x")
    assert cli_env(DEFAULT_SECRET_PATTERNS + ("^VENDOR_",))["VENDOR_ONLY"] == ""


def test_request_fields_refused_or_ignored():
    with pytest.raises(ClaudeCodeConfigError, match="skills"):
        check_llm_config(AnthropicLLMConfig(skills=[{"type": "anthropic", "skill_id": "xlsx"}]))
    with pytest.raises(ClaudeCodeConfigError, match="container_id"):
        check_llm_config(AnthropicLLMConfig(container_id="c1"))
    web_only = AnthropicLLMConfig(server_tools=[{"type": "web_search_20250305", "name": "web_search"}])
    assert check_llm_config(web_only) == []
    tuned = AnthropicLLMConfig(thinking_tokens=1024, beta_headers=["x"], enable_caching=False)
    assert set(check_llm_config(tuned)) == {"thinking_tokens", "beta_headers", "enable_caching"}


def test_web_tools_follow_the_server_tools():
    cfg = AnthropicLLMConfig(server_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
                                           {"type": "web_fetch_20250910", "name": "web_fetch"}])
    assert web_tools_for(cfg, None) == {"WebSearch": 5, "WebFetch": None}
    assert web_tools_for(AnthropicLLMConfig(), None) == {}
    assert web_tools_for(AnthropicLLMConfig(), True) == {"WebSearch": None, "WebFetch": None}
    assert web_tools_for(cfg, False) == {}


async def test_the_memory_store_meets_the_sdks_session_store_contracts():
    assert isinstance(MemoryTranscriptStore(), TranscriptStore)
    await run_session_store_conformance(MemoryTranscriptStore)
    store = MemoryTranscriptStore()
    key = {"project_key": "p", "session_id": "s"}
    await store.append(key, [{"type": "user", "uuid": "u1"}, {"type": "assistant", "uuid": "a1"},
                             {"type": "summary"}])
    assert await store.last_message_uuid("p", "s") == "a1"


async def test_the_ledger_finds_the_longest_known_prefix():
    ledger = MemoryTranscriptLedger()
    assert isinstance(ledger, TranscriptLedger)
    await ledger.record(agent_uuid="a", fingerprint="f2", prefix_len=2, session_id="s1", entry_uuid="e2")
    await ledger.record(agent_uuid="a", fingerprint="f4", prefix_len=4, session_id="s1", entry_uuid="e4")
    assert await ledger.longest_match({2: "f2", 4: "f4", 6: "f6"}) == (4, "s1", "e4")
    assert await ledger.longest_match({6: "f6"}) is None


def test_a_fingerprint_survives_a_save_and_names_one_exact_history():
    fmt = AnthropicMessageFormatter()
    history = [Message.user("What is 2 + 3?"), Message.assistant("5.")]
    reloaded = [Message.from_dict(m.to_dict()) for m in history]
    assert prefix_fingerprints(fmt, reloaded) == prefix_fingerprints(fmt, history)
    edited = [Message.user("What is 2 + 4?"), history[1]]
    assert all(a != b for a, b in zip(prefix_fingerprints(fmt, edited), prefix_fingerprints(fmt, history)))


async def test_process_slots_cap_how_many_processes_run():
    slots, a, b = ProcessSlots(), object(), object()
    assert await slots.acquire(a, limit=1, wait_s=1)
    assert not await slots.acquire(b, limit=1, wait_s=0.05)
    waiting = asyncio.create_task(slots.acquire(b, limit=1, wait_s=5))
    await asyncio.sleep(0.01)
    slots.release(a)
    assert await waiting and slots.in_use == 1


async def test_the_memory_store_knows_which_entries_a_main_transcript_holds():
    store = MemoryTranscriptStore()
    main = {"project_key": "p", "session_id": "s"}
    await store.append(main, [{"type": "user", "uuid": "u1"}, {"type": "summary"}])
    await store.append({**main, "subpath": "subagents/a"}, [{"type": "assistant", "uuid": "a1"}])
    assert await store.has_entry("p", "s", "u1")
    assert not await store.has_entry("p", "s", "a1")  # a sub-transcript's entry is not the main transcript's
    assert not await store.has_entry("p", "other", "u1")


async def test_sub_agents_draw_from_a_count_of_their_own_per_level():
    """A full count of runs leaves the children's untouched: children draw from ``slots.children``, grandchildren
    from ``slots.children.children``. ``total_in_use`` counts a level and every level below it."""
    slots, parent, child, grandchild = ProcessSlots(), object(), object(), object()
    assert await slots.acquire(parent, limit=1, wait_s=0)
    assert not await slots.acquire(child, limit=1, wait_s=0)
    assert await slots.children.acquire(child, limit=1, wait_s=0)
    assert await slots.children.children.acquire(grandchild, limit=1, wait_s=0)
    assert slots.children is slots.children and slots.in_use == 1 and slots.children.in_use == 1
    assert slots.total_in_use == 3 and slots.children.total_in_use == 2
    slots.children.release(child)
    slots.children.children.release(grandchild)
    slots.release(parent)
    assert slots.total_in_use == 0


def test_the_plans_meter_holds_a_warning_until_its_window_resets():
    meter, now = PlanMeter(), 1_000_000.0
    state = meter.note(SimpleNamespace(status="allowed_warning", rate_limit_type="five_hour", utilization=0.9,
                                       resets_at=now + 3600), now=now)
    assert state.window == "five_hour" and meter.near_limit(now=now).status == "allowed_warning"
    assert meter.near_limit(now=now + 3601) is None
    meter.note(SimpleNamespace(status="rejected", rate_limit_type="seven_day"), now=now)
    meter.note(SimpleNamespace(status="allowed", rate_limit_type="seven_day"), now=now)
    assert meter.near_limit(now=now) is None


async def test_the_url_guard_refuses_what_is_not_public():
    table = {"example.com": ["93.184.215.14"], "intranet": ["10.0.0.5"], "mixed": ["93.184.215.14", "127.0.0.1"],
             "v6local": ["::ffff:127.0.0.1"]}

    async def resolver(host, port):
        if host not in table:
            raise OSError("no such host")
        return table[host]

    guard = make_public_url_guard(resolver)
    assert await guard("https://example.com/page") is None
    for url in ("http://intranet/", "https://mixed/", "http://v6local/", "file:///etc/passwd", "ftp://example.com",
                "https://nowhere.invalid/", ""):
        assert await guard(url) is not None, url


def test_prompt_files_are_pruned_after_the_retention(tmp_path):
    folder = tmp_path / "prompts"
    folder.mkdir()
    old, new = folder / "old.txt", folder / "new.txt"
    old.write_text("x")
    new.write_text("y")
    import os
    import time

    os.utime(old, (time.time() - 40 * 86400,) * 2)
    assert prune_prompt_files(tmp_path, days=30) == 1
    assert not old.exists() and new.exists()
    assert prune_prompt_files(Path(tmp_path), days=0) == 0
