import pytest

from multi_agent_orchestrator.core.events import (
    AgentFinishEvent,
    AgentStartEvent,
    EventHandler,
    OrchestratorEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def test_event_initialization():
    start = AgentStartEvent("s1", "a1", "q1")
    assert start.session_id == "s1"
    assert start.agent_name == "a1"
    assert start.query == "q1"
    assert isinstance(start, OrchestratorEvent)

    finish = AgentFinishEvent("s2", "a2", "r2")
    assert finish.session_id == "s2"
    assert finish.agent_name == "a2"
    assert finish.response == "r2"

    tool_call = ToolCallEvent("s3", "a3", "t3", {"arg": "val"})
    assert tool_call.session_id == "s3"
    assert tool_call.agent_name == "a3"
    assert tool_call.tool_name == "t3"
    assert tool_call.args == {"arg": "val"}

    tool_result = ToolResultEvent("s4", "a4", "t4", "res4")
    assert tool_result.session_id == "s4"
    assert tool_result.agent_name == "a4"
    assert tool_result.tool_name == "t4"
    assert tool_result.result == "res4"


@pytest.mark.asyncio
async def test_event_handler_dispatch():
    class DummyEventHandler(EventHandler):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def on_agent_start(self, event: AgentStartEvent) -> None:
            self.calls.append(("start", event.agent_name))

        async def on_agent_finish(self, event: AgentFinishEvent) -> None:
            self.calls.append(("finish", event.agent_name))

        async def on_tool_call(self, event: ToolCallEvent) -> None:
            self.calls.append(("tool_call", event.tool_name))

        async def on_tool_result(self, event: ToolResultEvent) -> None:
            self.calls.append(("tool_result", event.tool_name))

    handler = DummyEventHandler()
    await handler.on_event(AgentStartEvent("s", "a", "q"))
    await handler.on_event(AgentFinishEvent("s", "a", "r"))
    await handler.on_event(ToolCallEvent("s", "a", "t", {}))
    await handler.on_event(ToolResultEvent("s", "a", "t", "res"))

    assert handler.calls == [
        ("start", "a"),
        ("finish", "a"),
        ("tool_call", "t"),
        ("tool_result", "t"),
    ]
