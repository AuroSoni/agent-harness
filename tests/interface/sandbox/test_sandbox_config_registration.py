"""Red-suite specs for auto-derived config/from_config + @register_sandbox (resolves F9).

Covers sandbox.md:
  - §2.5 `ConfigDrivenSandbox` mixin — auto-derived `config` property + `from_config` classmethod
    from the paired SandboxConfig dataclass (no per-field copy); round-trips.
  - §2.5 I12(b): `__init_subclass__` VALIDATES the config-field↔attribute mapping at class
    creation and RAISES (TypeError) when a config field has no matching __init__ parameter.
    An intermediate base without `config_class` is tolerated.
  - §2.5 `register_sandbox(sandbox_type, *, config_class=None)` decorator — registers the pair
    in one line via the existing registry fn; stamps `cls.sandbox_type`; raises if no config_class
    can be resolved.
  - The underlying `register_sandbox_type` registry fn (agent_base/sandbox/registry.py) stays
    public and is what the decorator wires into (verified via deserialize/sandbox_from_config).

`SandboxConfig` is used as a collaborator/base; the types under test are `ConfigDrivenSandbox`
and `register_sandbox`.
"""

from __future__ import annotations

import dataclasses
from typing import AsyncIterator

import pytest

from agent_base.sandbox import (
    ConfigDrivenSandbox,
    SandboxConfig,
    register_sandbox,
)
from agent_base.sandbox.registry import (
    deserialize_sandbox_config,
    register_sandbox_type,
    sandbox_from_config,
)


# ─── Reusable inert filesystem/exec stubs for a concrete ConfigDrivenSandbox ─


class _StubBody:
    """Inert abstract-method stubs so subclasses are instantiable. Not under test."""

    async def setup(self) -> None:  # pragma: no cover
        return None

    async def teardown(self) -> None:  # pragma: no cover
        return None

    async def read_file(self, path, offset=0, limit=None):  # pragma: no cover
        return ""

    async def write_file(self, path, content):  # pragma: no cover
        return None

    async def read_file_bytes(self, path) -> AsyncIterator[bytes]:  # pragma: no cover
        yield b""

    async def write_file_bytes(self, path, data) -> None:  # pragma: no cover
        return None

    async def list_dir(self, path="."):  # pragma: no cover
        return []

    async def file_exists(self, path):  # pragma: no cover
        return (False, None)

    async def delete(self, path):  # pragma: no cover
        return False

    async def import_file(self, filename, data):  # pragma: no cover
        return ""

    async def list_exported_files(self):  # pragma: no cover
        return []

    async def get_exported_file(self, path) -> AsyncIterator[bytes]:  # pragma: no cover
        yield b""

    async def get_exported_file_metadata(self):  # pragma: no cover
        return []

    async def exec(self, command, timeout=30.0, cwd=None, env=None):  # pragma: no cover
        raise NotImplementedError

    async def exec_stream(self, command, timeout=30.0, cwd=None, env=None) -> AsyncIterator[str]:  # pragma: no cover
        yield ""


@dataclasses.dataclass
class _DemoConfig(SandboxConfig):
    sandbox_type: str = "demo"
    sandbox_id: str = ""
    base_dir: str = ""
    default_timeout: float = 30.0


# ─── config / from_config auto-derivation ────────────────────────────────


def test_config_property_auto_derived_from_attributes():
    class _DemoSandbox(_StubBody, ConfigDrivenSandbox):
        config_class = _DemoConfig

        def __init__(self, sandbox_id="", base_dir="", default_timeout=30.0):
            self.sandbox_id = sandbox_id
            self.base_dir = base_dir
            self.default_timeout = default_timeout

    sb = _DemoSandbox(sandbox_id="s1", base_dir="/b", default_timeout=12.0)
    cfg = sb.config
    assert isinstance(cfg, _DemoConfig)
    assert cfg.sandbox_id == "s1"
    assert cfg.base_dir == "/b"
    assert cfg.default_timeout == 12.0
    # sandbox_type comes from the dataclass default (the dispatch key is not copied per-field).
    assert cfg.sandbox_type == "demo"


def test_from_config_recreates_instance():
    class _DemoSandbox(_StubBody, ConfigDrivenSandbox):
        config_class = _DemoConfig

        def __init__(self, sandbox_id="", base_dir="", default_timeout=30.0):
            self.sandbox_id = sandbox_id
            self.base_dir = base_dir
            self.default_timeout = default_timeout

    cfg = _DemoConfig(sandbox_id="s2", base_dir="/c", default_timeout=7.0)
    sb = _DemoSandbox.from_config(cfg)
    assert isinstance(sb, _DemoSandbox)
    assert sb.sandbox_id == "s2"
    assert sb.base_dir == "/c"
    assert sb.default_timeout == 7.0


def test_config_round_trips_through_from_config():
    class _DemoSandbox(_StubBody, ConfigDrivenSandbox):
        config_class = _DemoConfig

        def __init__(self, sandbox_id="", base_dir="", default_timeout=30.0):
            self.sandbox_id = sandbox_id
            self.base_dir = base_dir
            self.default_timeout = default_timeout

    original = _DemoSandbox(sandbox_id="r1", base_dir="/r", default_timeout=3.0)
    rebuilt = _DemoSandbox.from_config(original.config)
    assert rebuilt.config == original.config


# ─── I12(b): __init_subclass__ validation ────────────────────────────────


def test_init_subclass_raises_on_unmapped_config_field():
    @dataclasses.dataclass
    class _GapConfig(SandboxConfig):
        sandbox_type: str = "gap"
        present: str = ""
        orphan: str = ""  # no matching __init__ param → must raise at class creation

    with pytest.raises(TypeError):

        class _GapSandbox(_StubBody, ConfigDrivenSandbox):  # noqa: F811
            config_class = _GapConfig

            def __init__(self, present=""):
                self.present = present


def test_init_subclass_tolerates_intermediate_base_without_config_class():
    # An intermediate base may set config_class later; defining it without one must NOT raise.
    class _Intermediate(_StubBody, ConfigDrivenSandbox):
        pass

    assert issubclass(_Intermediate, ConfigDrivenSandbox)


def test_init_subclass_ignores_sandbox_type_field():
    # The sandbox_type dispatch key is exempt from the field↔param check (it's the dataclass default).
    @dataclasses.dataclass
    class _OkConfig(SandboxConfig):
        sandbox_type: str = "ok"
        sandbox_id: str = ""

    class _OkSandbox(_StubBody, ConfigDrivenSandbox):
        config_class = _OkConfig

        def __init__(self, sandbox_id=""):
            self.sandbox_id = sandbox_id

    assert _OkSandbox(sandbox_id="z").config.sandbox_id == "z"


# ─── @register_sandbox decorator ─────────────────────────────────────────


def test_register_sandbox_stamps_type_and_returns_class():
    @dataclasses.dataclass
    class _RegConfig(SandboxConfig):
        sandbox_type: str = "reg_demo_a"
        sandbox_id: str = ""

    @register_sandbox("reg_demo_a")
    class _RegSandbox(_StubBody, ConfigDrivenSandbox):
        config_class = _RegConfig

        def __init__(self, sandbox_id=""):
            self.sandbox_id = sandbox_id

    # Decorator returns the class and stamps the dispatch key on it.
    assert _RegSandbox.sandbox_type == "reg_demo_a"


def test_register_sandbox_wires_into_registry():
    @dataclasses.dataclass
    class _RegConfig(SandboxConfig):
        sandbox_type: str = "reg_demo_b"
        sandbox_id: str = ""

    @register_sandbox("reg_demo_b")
    class _RegSandbox(_StubBody, ConfigDrivenSandbox):
        config_class = _RegConfig

        def __init__(self, sandbox_id=""):
            self.sandbox_id = sandbox_id

    # The decorator calls the existing registry fn → deserialize/instantiate work end-to-end.
    cfg = deserialize_sandbox_config({"sandbox_type": "reg_demo_b", "sandbox_id": "x"})
    assert isinstance(cfg, _RegConfig)
    sb = sandbox_from_config(cfg)
    assert isinstance(sb, _RegSandbox)
    assert sb.sandbox_id == "x"


def test_register_sandbox_uses_explicit_config_class_argument():
    @dataclasses.dataclass
    class _ExplicitConfig(SandboxConfig):
        sandbox_type: str = "reg_demo_c"
        sandbox_id: str = ""

    @register_sandbox("reg_demo_c", config_class=_ExplicitConfig)
    class _RegSandbox(_StubBody, ConfigDrivenSandbox):
        config_class = _ExplicitConfig

        def __init__(self, sandbox_id=""):
            self.sandbox_id = sandbox_id

    cfg = deserialize_sandbox_config({"sandbox_type": "reg_demo_c", "sandbox_id": "y"})
    assert sandbox_from_config(cfg).sandbox_id == "y"


def test_register_sandbox_raises_without_resolvable_config_class():
    # No config_class arg and no config_class ClassVar → TypeError.
    with pytest.raises(TypeError):

        @register_sandbox("reg_demo_d")
        class _NoConfigSandbox(_StubBody, ConfigDrivenSandbox):  # noqa: F811
            def __init__(self):
                pass


def test_register_sandbox_type_is_still_public_registry_fn():
    # Migration note (still-true): the underlying registry fn stays public.
    @dataclasses.dataclass
    class _DirectConfig(SandboxConfig):
        sandbox_type: str = "reg_direct"
        sandbox_id: str = ""

    class _DirectSandbox(_StubBody, ConfigDrivenSandbox):
        config_class = _DirectConfig

        def __init__(self, sandbox_id=""):
            self.sandbox_id = sandbox_id

    register_sandbox_type("reg_direct", _DirectConfig, _DirectSandbox)
    cfg = deserialize_sandbox_config({"sandbox_type": "reg_direct", "sandbox_id": "d"})
    assert sandbox_from_config(cfg).sandbox_id == "d"
