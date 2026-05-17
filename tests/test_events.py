import pytest

from multi_agent_orchestrator.core.events import (
    AgentFinishEvent,
    AgentStartEvent,
    EventHandler,
    OrchestratorErrorEvent,
    OrchestratorEvent,
    OrchestratorFinishEvent,
    OrchestratorHandoffEvent,
    OrchestratorRouteEvent,
    OrchestratorStartEvent,
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

    # New events initialization
    ostart = OrchestratorStartEvent("s5", "q5")
    assert ostart.session_id == "s5"
    assert ostart.query == "q5"

    oroute = OrchestratorRouteEvent("s6", "q6", "a6")
    assert oroute.session_id == "s6"
    assert oroute.query == "q6"
    assert oroute.agent_name == "a6"

    ohandoff = OrchestratorHandoffEvent("s7", "a7", "a8", "msg7")
    assert ohandoff.session_id == "s7"
    assert ohandoff.source_agent == "a7"
    assert ohandoff.target_agent == "a8"
    assert ohandoff.message == "msg7"

    ofinish = OrchestratorFinishEvent("s8", "resp8")
    assert ofinish.session_id == "s8"
    assert ofinish.response == "resp8"

    err = ValueError("test error")
    oerror = OrchestratorErrorEvent("s9", err)
    assert oerror.session_id == "s9"
    assert oerror.error == err


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

        async def on_orchestrator_start(self, event: OrchestratorStartEvent) -> None:
            self.calls.append(("ostart", event.query))

        async def on_orchestrator_route(self, event: OrchestratorRouteEvent) -> None:
            self.calls.append(("oroute", event.agent_name))

        async def on_orchestrator_handoff(self, event: OrchestratorHandoffEvent) -> None:
            self.calls.append(("ohandoff", event.target_agent))

        async def on_orchestrator_finish(self, event: OrchestratorFinishEvent) -> None:
            self.calls.append(("ofinish", event.response))

        async def on_orchestrator_error(self, event: OrchestratorErrorEvent) -> None:
            self.calls.append(("oerror", type(event.error).__name__))

    handler = DummyEventHandler()
    await handler.on_event(AgentStartEvent("s", "a", "q"))
    await handler.on_event(AgentFinishEvent("s", "a", "r"))
    await handler.on_event(ToolCallEvent("s", "a", "t", {}))
    await handler.on_event(ToolResultEvent("s", "a", "t", "res"))
    await handler.on_event(OrchestratorStartEvent("s", "q"))
    await handler.on_event(OrchestratorRouteEvent("s", "q", "a"))
    await handler.on_event(OrchestratorHandoffEvent("s", "a1", "a2", "msg"))
    await handler.on_event(OrchestratorFinishEvent("s", "resp"))
    await handler.on_event(OrchestratorErrorEvent("s", ValueError("err")))

    assert handler.calls == [
        ("start", "a"),
        ("finish", "a"),
        ("tool_call", "t"),
        ("tool_result", "t"),
        ("ostart", "q"),
        ("oroute", "a"),
        ("ohandoff", "a2"),
        ("ofinish", "resp"),
        ("oerror", "ValueError"),
    ]


@pytest.mark.asyncio
async def test_base_event_handler_defaults():
    handler = EventHandler()
    # These should do nothing and not raise any exceptions
    await handler.on_event(AgentStartEvent("s", "a", "q"))
    await handler.on_event(AgentFinishEvent("s", "a", "r"))
    await handler.on_event(ToolCallEvent("s", "a", "t", {}))
    await handler.on_event(ToolResultEvent("s", "a", "t", "res"))
    await handler.on_event(OrchestratorStartEvent("s", "q"))
    await handler.on_event(OrchestratorRouteEvent("s", "q", "a"))
    await handler.on_event(OrchestratorHandoffEvent("s", "a1", "a2", "msg"))
    await handler.on_event(OrchestratorFinishEvent("s", "resp"))
    await handler.on_event(OrchestratorErrorEvent("s", ValueError("err")))
