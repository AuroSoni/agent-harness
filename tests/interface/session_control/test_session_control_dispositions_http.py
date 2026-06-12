"""Extended ``Disposition`` vocabulary + the shared HTTP map (closes A10).

Covers session-control.md §2.1: the additive NOT_RUNNING / NOT_FOUND members, the
reserved-but-unmapped MISDIRECTED member (Fork F, amended O4: the 421 row is REMOVED
from DISPOSITION_HTTP_STATUS until Rung 2), the single-source-of-truth
``DISPOSITION_HTTP_STATUS`` table shipped at ``agent_base.session.http``, and the
``ack_to_http`` reference helper (status, json-body shape, default-500 fall-through).
"""
from __future__ import annotations

from agent_base.core.ack import Ack, Disposition
from agent_base.session.http import DISPOSITION_HTTP_STATUS, ack_to_http


def test_new_disposition_members_exist():
    """§2.1: NOT_RUNNING and NOT_FOUND are additive public members."""
    assert Disposition.NOT_RUNNING.value == "not_running"
    assert Disposition.NOT_FOUND.value == "not_found"
    assert isinstance(Disposition.NOT_RUNNING, str)
    assert isinstance(Disposition.NOT_FOUND, str)


def test_misdirected_member_reserved_for_rung2():
    """Fork F: the enum member is reserved NOW so Rung 2 is not a public-enum break."""
    assert Disposition.MISDIRECTED.value == "misdirected"


def test_http_map_exact_rows():
    """§2.1: the exact disposition → status rows of the shared map."""
    assert DISPOSITION_HTTP_STATUS[Disposition.ACCEPTED] == 202
    assert DISPOSITION_HTTP_STATUS[Disposition.RESOLVED] == 200
    assert DISPOSITION_HTTP_STATUS[Disposition.STEERING] == 202
    assert DISPOSITION_HTTP_STATUS[Disposition.CANCELLING] == 202
    assert DISPOSITION_HTTP_STATUS[Disposition.IGNORED_STALE] == 200  # idempotent no-op
    assert DISPOSITION_HTTP_STATUS[Disposition.IGNORED_DUP] == 200   # idempotent retry
    assert DISPOSITION_HTTP_STATUS[Disposition.NOT_RUNNING] == 409   # Conflict
    assert DISPOSITION_HTTP_STATUS[Disposition.NOT_FOUND] == 404
    assert DISPOSITION_HTTP_STATUS[Disposition.REJECTED] == 422


def test_http_map_omits_misdirected_until_rung2():
    """Amended O4: MISDIRECTED has NO HTTP row at Rung 1; every other member is mapped."""
    assert Disposition.MISDIRECTED not in DISPOSITION_HTTP_STATUS
    assert set(DISPOSITION_HTTP_STATUS) == set(Disposition) - {Disposition.MISDIRECTED}


def test_ack_to_http_shape():
    """§2.1: ack_to_http(ack) -> (status, json body) with seq/disposition/detail keys."""
    status, body = ack_to_http(Ack(seq=7, disposition=Disposition.ACCEPTED))
    assert status == 202
    assert body == {"seq": 7, "disposition": "accepted", "detail": None}


def test_ack_to_http_detail_passthrough():
    status, body = ack_to_http(
        Ack(seq=2, disposition=Disposition.REJECTED, detail="mailbox_full")
    )
    assert status == 422
    assert body["detail"] == "mailbox_full"
    assert body["disposition"] == "rejected"


def test_ack_to_http_not_running_and_not_found():
    """§2.4 endpoint pattern: NOT_RUNNING → 409, NOT_FOUND → 404 via the one shared map."""
    status, _ = ack_to_http(Ack(seq=-1, disposition=Disposition.NOT_RUNNING))
    assert status == 409
    status, _ = ack_to_http(Ack(seq=-1, disposition=Disposition.NOT_FOUND))
    assert status == 404


def test_ack_to_http_unmapped_disposition_falls_through_to_500():
    """§2.1 (O4 note): a stray MISDIRECTED lookup falls through to the default 500,
    which correctly signals 'not a Rung-1 outcome'."""
    status, body = ack_to_http(Ack(seq=1, disposition=Disposition.MISDIRECTED))
    assert status == 500
    assert body["disposition"] == "misdirected"
