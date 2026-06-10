"""Red-suite specs — storage §2.5: media-metadata-by-id lookup (fixes E5).

Covers:
- interface_plan/subsystems/storage.md §2.5: ``get_media_metadata`` /
  ``find_generated_file`` ship as CONCRETE default implementations on the
  ABCs (R26 — never bare @abstractmethod, so existing custom adapters keep
  working), with optimized Postgres overrides.
- "Newest run wins" for ``find_generated_file``; canonical ``media_id`` only
  (legacy key spellings are normalized upstream by MediaMetadata — a media
  subsystem dependency, not re-tested here).
"""
from __future__ import annotations

import json

from agent_base.core.config import AgentConfig, Conversation
from agent_base.media_backend import MediaMetadata
from agent_base.storage.base import AgentConfigAdapter, ConversationAdapter
from agent_base.storage.pg import PgConfigAdapterBase, PgConversationAdapterBase


def _media(media_id: str = "m1", location: str = "local://m1") -> MediaMetadata:
    return MediaMetadata(
        media_id=media_id,
        media_mime_type="image/png",
        media_filename=f"{media_id}.png",
        media_extension="png",
        media_size=1024,
        storage_type="local",
        storage_location=location,
    )


# ---------------------------------------------------------------------------
# Fake adapters exercising the ABC concrete defaults
# ---------------------------------------------------------------------------

class _MemoryConfigAdapter(AgentConfigAdapter):
    def __init__(self, configs: dict[str, AgentConfig]):
        self._configs = configs

    async def save(self, config):
        self._configs[config.agent_uuid] = config

    async def load(self, agent_uuid):
        return self._configs.get(agent_uuid)

    async def delete(self, agent_uuid):
        return self._configs.pop(agent_uuid, None) is not None

    async def update_title(self, agent_uuid, title):
        return False

    async def list_sessions(self, limit=50, offset=0):
        return ([], 0)


class _MemoryConversationAdapter(ConversationAdapter):
    def __init__(self, conversations: list[Conversation]):
        # stored newest-first by sequence_number, as load_history returns them
        self._conversations = sorted(
            conversations, key=lambda c: c.sequence_number or 0, reverse=True,
        )

    async def save(self, conversation):
        self._conversations.insert(0, conversation)

    async def load_history(self, agent_uuid, limit=20, offset=0):
        matching = [c for c in self._conversations if c.agent_uuid == agent_uuid]
        return matching[offset:offset + limit]

    async def load_by_run_id(self, agent_uuid, run_id):
        for c in self._conversations:
            if c.agent_uuid == agent_uuid and c.run_id == run_id:
                return c
        return None

    async def load_cursor(self, agent_uuid, before=None, limit=20):
        matching = [
            c for c in self._conversations
            if c.agent_uuid == agent_uuid
            and (before is None or (c.sequence_number or 0) < before)
        ]
        return matching[:limit], len(matching) > limit


# ---------------------------------------------------------------------------
# Concrete default: AgentConfigAdapter.get_media_metadata (R26)
# ---------------------------------------------------------------------------

async def test_default_get_media_metadata_reads_media_registry():
    assert "get_media_metadata" not in _MemoryConfigAdapter.__dict__
    config = AgentConfig(agent_uuid="agent-1", media_registry={"m1": _media("m1")})
    adapter = _MemoryConfigAdapter({"agent-1": config})
    found = await adapter.get_media_metadata("agent-1", "m1")
    assert isinstance(found, MediaMetadata)
    assert found.media_id == "m1"


async def test_default_get_media_metadata_unknown_id_returns_none():
    config = AgentConfig(agent_uuid="agent-1", media_registry={"m1": _media("m1")})
    adapter = _MemoryConfigAdapter({"agent-1": config})
    assert await adapter.get_media_metadata("agent-1", "nope") is None


async def test_default_get_media_metadata_missing_agent_returns_none():
    adapter = _MemoryConfigAdapter({})
    assert await adapter.get_media_metadata("ghost", "m1") is None


# ---------------------------------------------------------------------------
# Concrete default: ConversationAdapter.find_generated_file (R26)
# ---------------------------------------------------------------------------

async def test_default_find_generated_file_scans_conversations():
    assert "find_generated_file" not in _MemoryConversationAdapter.__dict__
    conversation = Conversation(
        agent_uuid="agent-1", run_id="run-1",
        sequence_number=1, generated_files=[_media("m1")],
    )
    adapter = _MemoryConversationAdapter([conversation])
    found = await adapter.find_generated_file("agent-1", "m1")
    assert isinstance(found, MediaMetadata)
    assert found.media_id == "m1"


async def test_default_find_generated_file_newest_run_wins():
    old = Conversation(
        agent_uuid="agent-1", run_id="run-1", sequence_number=1,
        generated_files=[_media("m1", location="local://old")],
    )
    new = Conversation(
        agent_uuid="agent-1", run_id="run-2", sequence_number=2,
        generated_files=[_media("m1", location="local://new")],
    )
    adapter = _MemoryConversationAdapter([old, new])
    found = await adapter.find_generated_file("agent-1", "m1")
    assert found is not None
    assert found.storage_location == "local://new"


async def test_default_find_generated_file_absent_returns_none():
    conversation = Conversation(
        agent_uuid="agent-1", run_id="run-1",
        sequence_number=1, generated_files=[_media("m1")],
    )
    adapter = _MemoryConversationAdapter([conversation])
    assert await adapter.find_generated_file("agent-1", "missing") is None


# ---------------------------------------------------------------------------
# Pg overrides (the optimized single-query forms)
# ---------------------------------------------------------------------------

class _FakeTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self):
        self.calls: list[tuple[str, str, tuple]] = []
        self.fetchrow_result = None
        self.fetchval_result = None
        self.fetch_result: list = []

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args))
        return "OK"

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        return self.fetch_result

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", sql, args))
        return self.fetchrow_result

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args))
        return self.fetchval_result

    def transaction(self):
        return _FakeTxn()


class _Acquired:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn=None):
        self.conn = conn or _FakeConn()
        self.closed = False

    def acquire(self):
        return _Acquired(self.conn)

    async def close(self):
        self.closed = True


async def test_pg_find_generated_file_returns_typed_metadata_by_media_id():
    conn = _FakeConn()
    conn.fetchrow_result = {"gf": json.dumps(_media("m1", location="s3://hit").to_dict())}
    adapter = PgConversationAdapterBase(_FakePool(conn))
    found = await adapter.find_generated_file("agent-1", "m1")
    assert isinstance(found, MediaMetadata)
    assert found.media_id == "m1"
    assert found.storage_location == "s3://hit"
    # the canonical media_id is bound into the query (no key-spelling
    # reconciliation in the consumer — E5)
    bound_args = [a for _, _, args in conn.calls for a in args]
    assert "m1" in bound_args


async def test_pg_find_generated_file_none_when_no_row():
    adapter = PgConversationAdapterBase(_FakePool())
    assert await adapter.find_generated_file("agent-1", "missing") is None


async def test_pg_get_media_metadata_none_when_absent():
    adapter = PgConfigAdapterBase(_FakePool())   # every fetch returns None
    assert await adapter.get_media_metadata("agent-1", "missing") is None
