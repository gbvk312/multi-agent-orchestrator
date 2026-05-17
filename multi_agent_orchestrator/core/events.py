from typing import Any


class OrchestratorEvent:
    """Base class for all orchestrator events."""

    pass


class AgentStartEvent(OrchestratorEvent):
    def __init__(self, session_id: str, agent_name: str, query: str):
        self.session_id = session_id
        self.agent_name = agent_name
        self.query = query


class AgentFinishEvent(OrchestratorEvent):
    def __init__(self, session_id: str, agent_name: str, response: str):
        self.session_id = session_id
        self.agent_name = agent_name
        self.response = response


class ToolCallEvent(OrchestratorEvent):
    def __init__(self, session_id: str, agent_name: str, tool_name: str, args: dict[str, Any]):
        self.session_id = session_id
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.args = args


class ToolResultEvent(OrchestratorEvent):
    def __init__(self, session_id: str, agent_name: str, tool_name: str, result: str):
        self.session_id = session_id
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.result = result


class OrchestratorStartEvent(OrchestratorEvent):
    def __init__(self, session_id: str, query: str):
        self.session_id = session_id
        self.query = query


class OrchestratorRouteEvent(OrchestratorEvent):
    def __init__(self, session_id: str, query: str, agent_name: str):
        self.session_id = session_id
        self.query = query
        self.agent_name = agent_name


class OrchestratorHandoffEvent(OrchestratorEvent):
    def __init__(self, session_id: str, source_agent: str, target_agent: str, message: str):
        self.session_id = session_id
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.message = message


class OrchestratorFinishEvent(OrchestratorEvent):
    def __init__(self, session_id: str, response: str):
        self.session_id = session_id
        self.response = response


class OrchestratorErrorEvent(OrchestratorEvent):
    def __init__(self, session_id: str, error: Exception):
        self.session_id = session_id
        self.error = error


class EventHandler:
    """Base class for handling orchestrator events."""

    async def on_event(self, event: OrchestratorEvent) -> None:
        """Handle an incoming event."""
        if isinstance(event, AgentStartEvent):
            await self.on_agent_start(event)
        elif isinstance(event, AgentFinishEvent):
            await self.on_agent_finish(event)
        elif isinstance(event, ToolCallEvent):
            await self.on_tool_call(event)
        elif isinstance(event, ToolResultEvent):
            await self.on_tool_result(event)
        elif isinstance(event, OrchestratorStartEvent):
            await self.on_orchestrator_start(event)
        elif isinstance(event, OrchestratorRouteEvent):
            await self.on_orchestrator_route(event)
        elif isinstance(event, OrchestratorHandoffEvent):
            await self.on_orchestrator_handoff(event)
        elif isinstance(event, OrchestratorFinishEvent):
            await self.on_orchestrator_finish(event)
        elif isinstance(event, OrchestratorErrorEvent):
            await self.on_orchestrator_error(event)

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
