"""Public E2B policy/capture contract (2026-09 reliability amendment)."""
import inspect
from agent_base.sandbox import E2BSandbox, SandboxCoordinator, SnapshotPolicy
from agent_base.sandbox.sandbox_types import Sandbox, ExecResult


def test_injectable_consumer_coordination_and_policy():
    from agent_base.providers.anthropic.anthropic_agent import AnthropicAgent
    params = inspect.signature(AnthropicAgent).parameters
    assert params["sandbox_coordinator"].default is None
    assert params["snapshot_policy"].default is None
    assert SnapshotPolicy().per_file_cap == 50 * 1024 * 1024
    assert SnapshotPolicy().total_cap == 500 * 1024 * 1024
    assert hasattr(SandboxCoordinator, "exclusive")


def test_bounded_capture_is_part_of_the_sandbox_surface():
    assert "capture_limit_bytes" in inspect.signature(Sandbox.run_streaming).parameters
    assert "capture_limit_bytes" in inspect.signature(E2BSandbox.run_streaming).parameters
    assert ExecResult().output_truncated is False
    assert ExecResult().stdout_bytes == 0
