"""Conftest for the interface red suite.

Deliberately import-free: every module under tests/interface imports the
not-yet-implemented ``agent_base`` interfaces at module level, and collection
errors there must stay scoped to the individual test file. Shared fixtures
that depend on the new interfaces would turn one missing module into a
whole-suite collection failure.
"""
