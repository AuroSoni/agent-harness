"""Export-only filesystem operations. Sent as trusted inline code, even to old VMs.

All paths are opened relative to directory descriptors with O_NOFOLLOW. No SDK
read may follow a pathname after this helper checks it. The read protocol carries
offsets and a final checksum so dropped/replayed command output fails closed.
"""
import base64
import contextlib
import errno
import hashlib
import json
import os
import stat
import sys


class UnsafePath(Exception):
    pass


def _open(name, parent, directory=False):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if directory:
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(name, flags, dir_fd=parent)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafePath() from exc
        raise
    kind = os.fstat(fd).st_mode
    if not (stat.S_ISDIR(kind) if directory else stat.S_ISREG(kind)):
        os.close(fd)
        raise UnsafePath()
    return fd


@contextlib.contextmanager
def open_root(path):
    if not os.path.isabs(path) or path != os.path.normpath(path):
        raise UnsafePath()
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.split('/'):
            if not part:
                continue
            nxt = _open(part, fd, directory=True)
            os.close(fd)
            fd = nxt
        yield fd
    finally:
        os.close(fd)


@contextlib.contextmanager
def open_file(root_fd, path):
    parts = path.split('/')
    if not parts or any(p in ('', '.', '..') for p in parts):
        raise UnsafePath()
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            nxt = _open(part, fd, directory=True)
            os.close(fd)
            fd = nxt
        leaf = _open(parts[-1], fd)
        try:
            yield leaf
        finally:
            os.close(leaf)
    finally:
        os.close(fd)


def walk_files(fd, prefix='', rejected=None):
    """Hold parent descriptors through recursion; never traverse symlinks."""
    for name in sorted(os.listdir(fd)):
        path = prefix + name
        try:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or not (
                stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
            ):
                raise UnsafePath()
            child = _open(name, fd, directory=stat.S_ISDIR(info.st_mode))
            try:
                if stat.S_ISDIR(info.st_mode):
                    yield from walk_files(child, path + '/', rejected)
                else:
                    yield path, child
            finally:
                os.close(child)
        except FileNotFoundError:
            continue  # a file removed during discovery is no longer an export
        except UnsafePath:
            if rejected is not None:
                rejected.append(path)


def emit(record):
    print(json.dumps(record, separators=(',', ':')), flush=True)


def main():
    mode = os.environ['SBX_EXPORT_MODE']
    root = os.environ['SBX_EXPORT_ROOT']
    make_hash = None
    if mode == 'manifest':
        try:
            from blake3 import blake3
            make_hash = blake3
        except ImportError:
            emit({'error': 'hash_unavailable'})
            return 3
    try:
        with open_root(root) as root_fd:
            if mode == 'read':
                with open_file(root_fd, os.environ['SBX_EXPORT_PATH']) as fd:
                    offset = 0
                    digest = hashlib.sha256()
                    while True:
                        data = os.read(fd, 32768)
                        if not data:
                            break
                        emit({'offset': offset, 'data': base64.b64encode(data).decode('ascii')})
                        digest.update(data)
                        offset += len(data)
                    emit({'size': offset, 'sha256': digest.hexdigest()})
            elif mode in ('manifest', 'list'):
                files = {}
                rejected = []
                for path, fd in walk_files(root_fd, rejected=rejected):
                    size = os.fstat(fd).st_size
                    digest = None
                    if make_hash:
                        digest = make_hash()
                        size = 0
                        while True:
                            data = os.read(fd, 1 << 20)
                            if not data:
                                break
                            size += len(data)
                            digest.update(data)
                    files[path] = [digest.hexdigest() if digest else None, size]
                emit({'files': files, 'rejected_count': len(rejected)})
            else:
                raise ValueError('invalid export operation')
    except FileNotFoundError:
        if mode != 'read':
            emit({'files': {}, 'rejected_count': 0})
            return 0
        emit({'error': 'not_found'})
        return 5
    except UnsafePath:
        emit({'error': 'unsafe_path'})
        return 4
    except OSError:
        emit({'error': 'export_io_failed'})
        return 6
    return 0


if __name__ == '__main__':
    sys.exit(main())
