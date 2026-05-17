from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class OrchestratorEvent:
    """Base class for all orchestrator events."""


@dataclass(frozen=True, slots=True)
class AgentStartEvent(OrchestratorEvent):
    session_id: str
    agent_name: str
    query: str


@dataclass(frozen=True, slots=True)
class AgentFinishEvent(OrchestratorEvent):
    session_id: str
    agent_name: str
    response: str


@dataclass(frozen=True, slots=True)
class ToolCallEvent(OrchestratorEvent):
    session_id: str
    agent_name: str
    tool_name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultEvent(OrchestratorEvent):
    session_id: str
    agent_name: str
    tool_name: str
    result: str


@dataclass(frozen=True, slots=True)
class OrchestratorStartEvent(OrchestratorEvent):
    session_id: str
    query: str


@dataclass(frozen=True, slots=True)
class OrchestratorRouteEvent(OrchestratorEvent):
    session_id: str
    query: str
    agent_name: str


@dataclass(frozen=True, slots=True)
class OrchestratorHandoffEvent(OrchestratorEvent):
    session_id: str
    source_agent: str
    target_agent: str
    message: str


@dataclass(frozen=True, slots=True)
class OrchestratorFinishEvent(OrchestratorEvent):
    session_id: str
    response: str


@dataclass(frozen=True, slots=True)
class OrchestratorErrorEvent(OrchestratorEvent):
    session_id: str
    error: Exception


class EventHandler:
    """Base class for handling orchestrator events.

    Uses a dispatch registry for O(1) event routing. Subclass and override
    individual ``on_*`` methods to handle specific event types.
    """

    _EVENT_DISPATCH: ClassVar[dict[type, str]] = {
        AgentStartEvent: "on_agent_start",
        AgentFinishEvent: "on_agent_finish",
        ToolCallEvent: "on_tool_call",
        ToolResultEvent: "on_tool_result",
        OrchestratorStartEvent: "on_orchestrator_start",
        OrchestratorRouteEvent: "on_orchestrator_route",
        OrchestratorHandoffEvent: "on_orchestrator_handoff",
        OrchestratorFinishEvent: "on_orchestrator_finish",
        OrchestratorErrorEvent: "on_orchestrator_error",
    }

    async def on_event(self, event: OrchestratorEvent) -> None:
        """Handle an incoming event via the dispatch registry."""
        method_name = self._EVENT_DISPATCH.get(type(event))
        if method_name is not None:
            await getattr(self, method_name)(event)

    async def on_agent_start(self, event: AgentStartEvent) -> None:
        pass

    async def on_agent_finish(self, event: AgentFinishEvent) -> None:
        pass

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        pass

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        pass

    async def on_orchestrator_start(self, event: OrchestratorStartEvent) -> None:
        pass

    async def on_orchestrator_route(self, event: OrchestratorRouteEvent) -> None:
        pass

    async def on_orchestrator_handoff(self, event: OrchestratorHandoffEvent) -> None:
        pass

    async def on_orchestrator_finish(self, event: OrchestratorFinishEvent) -> None:
        pass

    async def on_orchestrator_error(self, event: OrchestratorErrorEvent) -> None:
        pass
