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

    async def on_agent_start(self, event: AgentStartEvent) -> None:
        pass

    async def on_agent_finish(self, event: AgentFinishEvent) -> None:
        pass

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        pass

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        pass
