from __future__ import annotations

from pathlib import Path

import pytest

# Directory holding the integration suite. The auto-marker below must apply ONLY
# to items under here — a session-wide ``pytest_collection_modifyitems`` runs for
# the whole run, so marking every collected item (as this previously did)
# deselected the unit suite too whenever ``tests/`` was collected as a whole.
_INTEGRATION_DIR = Path(__file__).parent.resolve()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        raw_path = getattr(item, "path", None) or getattr(item, "fspath", None)
        if raw_path is None:
            continue
        try:
            item_path = Path(str(raw_path)).resolve()
        except Exception:
            continue
        if item_path == _INTEGRATION_DIR or _INTEGRATION_DIR in item_path.parents:
            item.add_marker(pytest.mark.integration)
