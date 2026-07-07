"""Auto-derived config/from_config + @register_sandbox decorator (resolves F9).

``ConfigDrivenSandbox`` derives ``config``/``from_config`` from the paired
``SandboxConfig`` dataclass so a subclass does not hand-copy fields. ``register_sandbox``
is a class decorator that wires ``(sandbox_type → config_class, sandbox_class)`` in one
place, replacing the module-bottom side-effect.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, ClassVar

from .registry import register_sandbox_type
from .sandbox_types import Sandbox, SandboxConfig


class ConfigDrivenSandbox(Sandbox):
    """Mixin: derive ``config()``/``from_config()`` from the paired ``SandboxConfig``.

    A subclass declares ``config_class`` and stores each config field as an attribute of
    the same name. ``config()``/``from_config()`` are then auto-generated — no per-field
    copying (F9).

    I12(b): ``__init_subclass__`` VALIDATES the config-field↔attribute mapping at class
    creation and RAISES — a config field with no matching constructor parameter (so
    ``from_config`` could silently drop it) is a hard error at import time.
    """

    config_class: ClassVar[type[SandboxConfig]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cfg = getattr(cls, "config_class", None)
        if cfg is None:
            return  # an intermediate base may set config_class later
        accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
        missing = {
            f.name
            for f in dataclasses.fields(cfg)
            if f.name != "sandbox_type" and f.name not in accepted
        }
        if missing:
            raise TypeError(
                f"{cls.__name__}: config fields {sorted(missing)} have no matching "
                f"__init__ parameter — from_config would silently drop them (I12(b))."
            )

    @property
    def config(self) -> SandboxConfig:
        fields = {
            f.name
            for f in dataclasses.fields(self.config_class)
            if f.name != "sandbox_type"
        }
        values = {name: getattr(self, name) for name in fields if hasattr(self, name)}
        return self.config_class(**values)

    @classmethod
    def from_config(cls, config: SandboxConfig) -> Sandbox:
        sig = inspect.signature(cls.__init__)
        accepted = {p for p in sig.parameters if p != "self"}
        kwargs = {
            k: v
            for k, v in dataclasses.asdict(config).items()
            if k in accepted and k != "sandbox_type"
        }
        return cls(**kwargs)


def register_sandbox(
    sandbox_type: str,
    *,
    config_class: type[SandboxConfig] | None = None,
):
    """Class decorator: register a Sandbox subclass under ``sandbox_type`` in one line.

    Replaces the hand-written module-bottom ``register_sandbox_type(...)`` side-effect.
    If ``config_class`` is omitted, uses the subclass's ``config_class`` ClassVar.
    """

    def _wrap(cls: type[Sandbox]) -> type[Sandbox]:
        cfg = config_class or getattr(cls, "config_class", None)
        if cfg is None:
            raise TypeError(
                f"{cls.__name__} must set config_class or pass config_class="
            )
        register_sandbox_type(sandbox_type, cfg, cls)
        cls.sandbox_type = sandbox_type  # keep dispatch key on the class too
        return cls

    return _wrap
