"""Phase 1 — bounded FIFO mailbox with explicit backpressure + freeze."""
from agent_base.core import Message
from agent_base.core.commands import UserMessage
from agent_base.session.mailbox import Mailbox


def _um(i: int) -> UserMessage:
    return UserMessage(message=Message.user(f"m{i}"))


def test_offer_take_is_fifo_oldest_first():
    mb = Mailbox(capacity=3)
    a, b = _um(1), _um(2)
    assert mb.offer(a) is True
    assert mb.offer(b) is True
    assert len(mb) == 2
    assert mb.take() is a  # oldest first
    assert mb.take() is b
    assert mb.take() is None


def test_capacity_backpressure_no_drop():
    mb = Mailbox(capacity=2)
    assert mb.offer(_um(1)) is True
    assert mb.offer(_um(2)) is True
    assert mb.offer(_um(3)) is False  # full → rejected, not dropped
    assert len(mb) == 2


def test_freeze_blocks_offer_then_drain():
    mb = Mailbox(capacity=5)
    mb.offer(_um(1))
    mb.offer(_um(2))
    mb.freeze()
    assert mb.frozen is True
    assert mb.offer(_um(3)) is False
    items = mb.drain()
    assert len(items) == 2
    assert len(mb) == 0
    mb.unfreeze()
    assert mb.offer(_um(4)) is True
