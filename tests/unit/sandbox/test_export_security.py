"""Export publication must never follow a sandbox pathname through a symlink."""
import os
from pathlib import Path

import pytest

from agent_base.sandbox import SandboxPathEscapeError
from agent_base.sandbox.remote_scripts import export_files
from .fake_e2b import FakeE2BTransport
from .test_e2b_sandbox import _make, _multiroot


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
async def test_exports_skip_symlinks_and_direct_reads_refuse_them(tmp_path, monkeypatch, absolute, fallback):
    transport = FakeE2BTransport(tmp_path / 'remote')
    sb = (_multiroot if absolute else _make)(transport)
    await sb.setup()
    await sb.write_file(f'{sb.exports_dir}/nested/good.txt', 'good')
    box = transport.boxes[sb.e2b_sandbox_id]
    exports = box.host_dir / sb.abs_path(sb.exports_dir).lstrip('/')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'sentinel.txt').write_text('must not publish')
    (exports / 'leaf.txt').symlink_to(outside / 'sentinel.txt')
    (exports / 'parent').symlink_to(outside, target_is_directory=True)
    (exports / 'internal.txt').symlink_to(exports / 'nested/good.txt')
    if fallback:
        async def unavailable():
            return None
        monkeypatch.setattr(sb, '_export_manifest', unavailable)
    assert await sb.list_exported_files() == ['nested/good.txt']
    assert [m.path for m in await sb.get_exported_file_metadata()] == ['nested/good.txt']
    for path in ('leaf.txt', 'parent/sentinel.txt', 'internal.txt', '../outside/sentinel.txt'):
        with pytest.raises(SandboxPathEscapeError):
            _ = [c async for c in sb.get_exported_file(path)]
    assert transport.bytes_read == 0


@pytest.mark.parametrize('parent', [False, True])
async def test_export_root_and_ancestor_symlinks_are_refused(tmp_path, parent):
    transport = FakeE2BTransport(tmp_path / 'remote')
    sb = _multiroot(transport)
    await sb.setup()
    exports = transport.boxes[sb.e2b_sandbox_id].host_dir / 'mnt/user-data/outputs'
    target = exports.parent if parent else exports
    moved = target.with_name(target.name + '-moved')
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=True)
    for operation in (sb.list_exported_files, sb.get_exported_file_metadata):
        with pytest.raises(SandboxPathEscapeError):
            await operation()


async def test_file_swapped_after_manifest_cannot_be_exported(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    sb = _multiroot(transport)
    await sb.setup()
    await sb.write_file(f'{sb.exports_dir}/report.txt', 'report')
    assert len(await sb.get_exported_file_metadata()) == 1
    exports = transport.boxes[sb.e2b_sandbox_id].host_dir / 'mnt/user-data/outputs'
    sentinel = tmp_path / 'sentinel.txt'
    sentinel.write_text('private')
    (exports / 'report.txt').unlink()
    (exports / 'report.txt').symlink_to(sentinel)
    with pytest.raises(SandboxPathEscapeError):
        _ = [c async for c in sb.get_exported_file('report.txt')]


@pytest.mark.parametrize('directory', [False, True])
def test_symlink_swap_between_stat_and_open_is_refused(tmp_path, monkeypatch, directory):
    root = tmp_path / 'exports'
    root.mkdir()
    target = root / 'target'
    outside = tmp_path / 'outside'
    if directory:
        target.mkdir()
        outside.mkdir()
        (outside / 'sentinel.txt').write_text('private')
    else:
        target.write_text('good')
        outside.write_text('private')
    original_open = os.open
    def swap(path, *args, **kwargs):
        if path == 'target':
            target.rename(root / 'moved')
            target.symlink_to(outside, target_is_directory=directory)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(export_files.os, 'open', swap)
    with export_files.open_root(str(root.resolve())) as fd:
        assert list(export_files.walk_files(fd)) == []


async def test_binary_read_larger_than_command_capture_limit_is_complete(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    sb = _multiroot(transport)
    await sb.setup()
    payload = bytes(range(256)) * 40000  # spills the host spool; exceeds 8 MiB
    await sb.write_bytes(f'{sb.exports_dir}/large.bin', payload)
    assert b''.join([c async for c in sb.get_exported_file('large.bin')]) == payload
    assert transport.bytes_read == 0


async def test_unavailable_safe_reader_fails_closed(tmp_path):
    transport = FakeE2BTransport(tmp_path / 'remote')
    sb = _make(transport, python_path='definitely-not-a-python-binary')
    await sb.setup()
    await sb.write_file('.exports/good.txt', 'good')
    with pytest.raises(RuntimeError, match='Safe export listing unavailable'):
        await sb.get_exported_file_metadata()
    assert transport.bytes_read == 0


@pytest.mark.parametrize('protocol', [
    '{"offset":0,"data":"eA=="}\n',
    '{"offset":2,"data":"eA=="}\n',
    '{"offset":0,"data":"eA=="}\n{"size":1,"sha256":"wrong"}\n',
])
async def test_incomplete_or_corrupt_read_never_yields_bytes(tmp_path, monkeypatch, protocol):
    from agent_base.sandbox.sandbox_types import ExecResult
    sb = _make(FakeE2BTransport(tmp_path / 'remote'))
    async def corrupt(command, *, on_output, **kwargs):
        on_output(protocol)
        return ExecResult(exit_code=0)
    monkeypatch.setattr(sb, 'run_streaming', corrupt)
    received = []
    with pytest.raises(RuntimeError):
        async for chunk in sb.get_exported_file('a.txt'):
            received.append(chunk)
    assert received == []
