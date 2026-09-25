"""Upload errors release both consumers before releasing the caller's guard."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_base.media_backend.local import LocalMediaBackend


@pytest.mark.parametrize('failure', ['store', 'sandbox'])
async def test_failed_upload_cancels_and_joins_sibling(tmp_path, failure):
    backend = LocalMediaBackend(tmp_path)
    reader_started = asyncio.Event()
    queue_filled = asyncio.Event()
    finished = set()
    never = asyncio.Event()

    async def content():
        for index in range(20):
            if index == 4:
                queue_filled.set()
            yield bytes([index])

    async def store(stream, *args):
        try:
            await reader_started.wait()
            if failure == 'store':
                await anext(stream)
                raise OSError('store failed')
            async for _ in stream:
                pass
        finally:
            await asyncio.sleep(0)
            finished.add('store')

    async def import_file(filename, stream):
        reader_started.set()
        try:
            if failure == 'sandbox':
                # The producer is blocked writing a fifth chunk. Its cleanup
                # must not attempt to put an EOF sentinel into this full queue.
                await queue_filled.wait()
                raise OSError('sandbox failed')
            await anext(stream)
            await never.wait()
        finally:
            await asyncio.sleep(0)
            finished.add('sandbox')

    backend.store = store
    backend.attach_sandbox(SimpleNamespace(import_file=import_file))
    source = content()
    try:
        with pytest.raises(OSError, match=failure + ' failed'):
            await asyncio.wait_for(backend.user_upload(source, 'x.csv', 'text/csv', 'a'), 2)
        assert finished == {'store', 'sandbox'}
    finally:
        await source.aclose()


async def test_caller_cancellation_joins_both_upload_consumers(tmp_path):
    backend = LocalMediaBackend(tmp_path)
    started = {side: asyncio.Event() for side in ['store', 'sandbox']}
    finished = set()
    never = asyncio.Event()

    async def content():
        yield b'x'

    async def consume(side):
        started[side].set()
        try:
            await never.wait()
        finally:
            await asyncio.sleep(0)
            finished.add(side)

    async def store(stream, *args):
        await consume('store')

    async def import_file(filename, stream):
        await consume('sandbox')

    backend.store = store
    backend.attach_sandbox(SimpleNamespace(import_file=import_file))
    upload = asyncio.create_task(backend.user_upload(content(), 'x', 'text/plain', 'a'))
    await asyncio.gather(*(event.wait() for event in started.values()))
    upload.cancel()
    with pytest.raises(asyncio.CancelledError):
        await upload
    assert finished == {'store', 'sandbox'}


async def test_second_cancellation_does_not_interrupt_sibling_cleanup(tmp_path):
    backend = LocalMediaBackend(tmp_path)
    started, cleaning, release, finished = (asyncio.Event() for _ in range(4))

    async def content():
        yield b'x'

    async def store(stream, *args):
        await started.wait()
        raise OSError('store failed')

    async def import_file(filename, stream):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()

    backend.store = store
    backend.attach_sandbox(SimpleNamespace(import_file=import_file))
    upload = asyncio.create_task(backend.user_upload(content(), 'x', 'text/plain', 'a'))
    await cleaning.wait()
    upload.cancel()
    await asyncio.sleep(0)
    assert not upload.done()
    release.set()
    with pytest.raises(OSError, match='store failed'):
        await upload
    assert finished.is_set()


async def test_upload_success_streams_identical_bytes_to_both_destinations(tmp_path):
    backend = LocalMediaBackend(tmp_path)
    received = {}
    metadata = object()

    async def content():
        for index in range(20):
            yield bytes([index])

    async def store(stream, *args):
        received['store'] = b''.join([chunk async for chunk in stream])
        return metadata

    async def import_file(filename, stream):
        received['sandbox'] = b''.join([chunk async for chunk in stream])
        return 'workspace/x'

    backend.store = store
    backend.attach_sandbox(SimpleNamespace(import_file=import_file))
    assert await backend.user_upload(content(), 'x', 'text/plain', 'a') == (metadata, 'workspace/x')
    assert received['store'] == received['sandbox'] == bytes(range(20))
