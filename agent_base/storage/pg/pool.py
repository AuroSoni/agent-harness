"""Injectable Postgres pool — storage.md §2.3 (fixes E4).

The adapter no longer owns connection management by fiat: consumers inject a
live :data:`PgPool` (e.g. one FastAPI app pool shared across all three
adapters), or use ``from_dsn()`` for the back-compat path where the adapter
creates and OWNS the pool. ``connect()``/``close()`` act iff owned and are
no-ops iff borrowed.
"""
from __future__ import annotations

from dataclasses import dataclass

import asyncpg

#: The unit consumers inject (alias so signatures stay readable).
PgPool = asyncpg.Pool


@dataclass
class PgConnectConfig:
    """Connection settings for a library-created (owned) pool."""

    dsn: str
    min_size: int = 1
    max_size: int = 10
    timezone: str = "UTC"


async def create_pool(cfg: PgConnectConfig) -> PgPool:
    """Create an asyncpg pool from :class:`PgConnectConfig`."""
    return await asyncpg.create_pool(
        cfg.dsn,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        server_settings={"timezone": cfg.timezone},
    )


class _OwnedPool:
    """Wraps a DSN-created pool the adapter owns (``connect()``/``close()`` act)."""

    def __init__(self, cfg: PgConnectConfig) -> None:
        self.cfg = cfg
        self.pool: PgPool | None = None

    async def connect(self) -> PgPool:
        if self.pool is None:
            self.pool = await create_pool(self.cfg)
        return self.pool

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None


class _BorrowedPool:
    """Wraps an externally-managed pool (``connect()``/``close()`` are no-ops)."""

    def __init__(self, pool: PgPool) -> None:
        self.pool = pool

    async def connect(self) -> PgPool:
        return self.pool

    async def close(self) -> None:  # never closes a borrowed pool (E4)
        return None


__all__ = ["PgPool", "PgConnectConfig", "create_pool"]
