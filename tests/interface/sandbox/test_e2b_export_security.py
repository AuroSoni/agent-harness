"""EX-1: E2B export discovery and publication refuse symlink paths."""
import sys

import pytest

from agent_base.sandbox import E2BSandbox, SandboxPathEscapeError
from tests.unit.sandbox.fake_e2b import FakeE2BTransport


async def test_e2b_exports_publish_regular_nested_files_but_never_links(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'vm')
    sandbox = E2BSandbox(sandbox_id='exports', transport=transport, python_path=sys.executable)
    await sandbox.setup()
    await sandbox.write_file('.exports/nested/ok.txt', 'ok')
    root = transport.boxes[sandbox.e2b_sandbox_id].host_dir / sandbox.root_path.lstrip('/')
    secret = tmp_path / 'sentinel.txt'
    secret.write_text('not an export')
    (root / '.exports/link.txt').symlink_to(secret)
    assert await sandbox.list_exported_files() == ['nested/ok.txt']
    assert [m.path for m in await sandbox.get_exported_file_metadata()] == ['nested/ok.txt']
    assert b''.join([b async for b in sandbox.get_exported_file('nested/ok.txt')]) == b'ok'
    with pytest.raises(SandboxPathEscapeError):
        _ = [b async for b in sandbox.get_exported_file('link.txt')]
