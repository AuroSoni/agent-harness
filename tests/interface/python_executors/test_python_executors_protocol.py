"""Red-suite specs for the ``PythonExecutor`` Protocol (U1 promotion).

Covers python-executors.md:
  - §2.3 ``PythonExecutor`` promoted from an empty marker to a
    ``@runtime_checkable`` ``Protocol`` with method surface
    ``bind_tools``/``bind_variables``/``run``/``arun``/``reset`` and a
    ``policy`` attribute.
  - §3.4 / §6: a custom backend that structurally satisfies the surface passes
    ``isinstance(x, PythonExecutor)``; ``LocalPythonExecutor`` does too.

``PythonExecutor`` is owned by this subsystem, so its protocol surface is
deep-tested here. The custom ``E2BPythonExecutor`` in §3.4 is modelled as a
small in-file fake collaborator. The implementation does not exist yet.
"""

import typing

from agent_base.python_executors import PythonExecutor as PythonExecutorFromPkg
from agent_base.python_executors.base import ExecutorPolicy
from agent_base.python_executors.local_python_executor import (
    ExecutorResult,
    LocalPythonExecutor,
    PythonExecutor,
)


def test_python_executor_is_a_protocol():
    assert issubclass(PythonExecutor, typing.Protocol)  # type: ignore[arg-type]


def test_python_executor_is_runtime_checkable():
    # §2.3: @runtime_checkable so isinstance works for structural backends.
    assert getattr(PythonExecutor, "_is_runtime_protocol", False) is True


def test_package_reexports_the_same_protocol():
    assert PythonExecutorFromPkg is PythonExecutor


def test_protocol_declares_the_lifecycle_methods():
    for method in ("bind_tools", "bind_variables", "run", "arun", "reset"):
        assert hasattr(PythonExecutor, method)


def test_protocol_members_include_policy_attribute():
    members = getattr(PythonExecutor, "__protocol_attrs__", None)
    if members is None:
        # Fallback for runtimes that don't expose __protocol_attrs__: the
        # annotation must declare `policy`.
        members = set(getattr(PythonExecutor, "__annotations__", {}))
    assert "policy" in members


def test_structural_backend_passes_isinstance():
    # §3.4 E2BPythonExecutor-shaped fake: structurally satisfies the surface.
    class _StructuralBackend:
        def __init__(self) -> None:
            self.policy = ExecutorPolicy()

        def bind_tools(self, tools, *, replace=False):
            return None

        def bind_variables(self, variables):
            return None

        def reset(self):
            return None

        def run(self, code, *, ctx=None):
            return ExecutorResult(output=None, logs="", is_final_answer=False)

        async def arun(self, code, *, ctx=None):
            return self.run(code, ctx=ctx)

    assert isinstance(_StructuralBackend(), PythonExecutor)


def test_incomplete_backend_fails_isinstance():
    # Missing run/arun/reset — not structurally a PythonExecutor.
    class _Incomplete:
        def bind_tools(self, tools, *, replace=False):
            return None

    assert not isinstance(_Incomplete(), PythonExecutor)


def test_local_executor_satisfies_protocol_isinstance():
    # §6 migration row 1 (load-bearing): the SHIPPED LocalPythonExecutor must
    # satisfy its own runtime-checkable Protocol. Default construction works
    # because the default policy's effective imports are all stdlib.
    assert isinstance(LocalPythonExecutor(), PythonExecutor)


def test_local_executor_is_subclass_of_protocol():
    # §2.4: LocalPythonExecutor(PythonExecutor) explicit-subclass path also
    # holds under issubclass against the runtime-checkable Protocol.
    assert issubclass(LocalPythonExecutor, PythonExecutor)
