from agent_base.observability import bind_context, emit, install_sink, reset_context


def test_noop_and_installed_sink_are_fail_soft():
    install_sink(None)
    emit("ignored", value=1)
    events = []
    install_sink(events.append)
    token = bind_context(interaction_id="ix-1")
    try:
        emit("tool_wait", wait_ms=12)
    finally:
        reset_context(token)
        install_sink(None)
    assert events[0].kind == "tool_wait"
    assert events[0].attributes["interaction_id"] == "ix-1"
