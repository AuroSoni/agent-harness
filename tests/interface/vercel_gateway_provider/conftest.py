from __future__ import annotations

import pytest

from agent_base.await_table import AwaitTable, get_await_table, set_await_table

from ._fakes import Gate


@pytest.fixture(autouse=True)
def fresh_await_table():
    """Each scenario gets its own await table, and the slow tools' gate is reset."""
    original = get_await_table()
    set_await_table(AwaitTable())
    Gate.started = None
    yield
    set_await_table(original)
