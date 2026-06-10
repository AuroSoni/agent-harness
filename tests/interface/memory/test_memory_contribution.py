"""Interface red-suite: ``MemoryContribution`` (memory subsystem).

Covers memory.md:
  - §2.2 "The contributed recall shape (NEW — MemoryContribution)"
  - §5 Produces: MemoryContribution homed at ``agent_base/memory/base.py``
    (blocks + placement Literal["user_suffix","system_suffix"], default "user_suffix").

``MemoryContribution`` is OWNED by the memory subsystem, so it is deep-tested here:
construction, constructor defaults, the placement vocabulary, frozen-ness, and the
fact that it carries the recall blocks the loop will splice.

``ContentBlock`` / ``TextContent`` / ``Message`` are existing, unchanged vocabulary
(memory.md §2.1) and are used here strictly as collaborators.
"""
from __future__ import annotations

import dataclasses

import pytest

from agent_base.core.types import TextContent
from agent_base.memory.base import MemoryContribution


def test_memory_contribution_is_a_dataclass():
    assert dataclasses.is_dataclass(MemoryContribution)


def test_memory_contribution_default_placement_is_user_suffix():
    # §2.2: placement defaults to "user_suffix".
    contribution = MemoryContribution(blocks=[])
    assert contribution.placement == "user_suffix"


def test_memory_contribution_default_blocks_required_but_can_be_empty():
    # §2.2: ``blocks: list[ContentBlock]`` has NO default — it is mandatory. A regression
    # giving it a default (e.g. field(default_factory=list)) must be caught.
    with pytest.raises(TypeError):
        MemoryContribution()  # type: ignore[call-arg]
    # But an explicit empty list is accepted: empty blocks = inject nothing (recall miss).
    contribution = MemoryContribution(blocks=[])
    assert contribution.blocks == []


def test_memory_contribution_carries_recall_blocks_in_order():
    a = TextContent(text="fact-a")
    b = TextContent(text="fact-b")
    contribution = MemoryContribution(blocks=[a, b])
    assert contribution.blocks == [a, b]
    assert contribution.blocks[0] is a
    assert contribution.blocks[1] is b


def test_memory_contribution_accepts_system_suffix_placement():
    # §2.2 / §5: the other legal placement value.
    contribution = MemoryContribution(
        blocks=[TextContent(text="x")], placement="system_suffix"
    )
    assert contribution.placement == "system_suffix"


def test_memory_contribution_accepts_user_suffix_placement_explicitly():
    contribution = MemoryContribution(
        blocks=[TextContent(text="x")], placement="user_suffix"
    )
    assert contribution.placement == "user_suffix"


def test_memory_contribution_is_frozen():
    # §2.2 declares ``@dataclass(frozen=True)``.
    contribution = MemoryContribution(blocks=[])
    with pytest.raises(dataclasses.FrozenInstanceError):
        contribution.placement = "system_suffix"  # type: ignore[misc]


def test_memory_contribution_blocks_is_frozen_field_too():
    contribution = MemoryContribution(blocks=[TextContent(text="x")])
    with pytest.raises(dataclasses.FrozenInstanceError):
        contribution.blocks = []  # type: ignore[misc]


def test_memory_contribution_placement_is_keyword_or_positional():
    # Field order is (blocks, placement) per the pseudocode.
    contribution = MemoryContribution([TextContent(text="x")], "system_suffix")
    assert contribution.placement == "system_suffix"


def test_memory_contribution_fields_are_exactly_blocks_and_placement():
    # The store names the placement; the loop owns splicing. No extra surface leaks in.
    names = {f.name for f in dataclasses.fields(MemoryContribution)}
    assert names == {"blocks", "placement"}


def test_memory_contribution_equality_by_value():
    one = MemoryContribution(blocks=[], placement="user_suffix")
    two = MemoryContribution(blocks=[], placement="user_suffix")
    assert one == two


def test_memory_contribution_inequality_on_placement():
    one = MemoryContribution(blocks=[], placement="user_suffix")
    two = MemoryContribution(blocks=[], placement="system_suffix")
    assert one != two
