"""Structured JSON Logging Event Handler.

Demonstrates how to capture and trace every orchestrator lifecycle event
with structured logging for production observability.

Usage:
    orchestrator = Orchestrator(event_handler=StructuredLoggingHandler())
"""

import json
import logging
import sys
from typing import Any

from multi_agent_orchestrator import (
    AgentFinishEvent,
    AgentStartEvent,
    EventHandler,
    OrchestratorErrorEvent,
    OrchestratorFinishEvent,
    OrchestratorHandoffEvent,
    OrchestratorRouteEvent,
    OrchestratorStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)

# Configure structured JSON logger
logger = logging.getLogger("orchestrator.trace")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _log(event_type: str, **fields: Any) -> None:
    """Emit a structured JSON log line."""
    logger.info(json.dumps({"event": event_type, **fields}, default=str))


class StructuredLoggingHandler(EventHandler):
    """Logs every orchestrator event as a structured JSON line.

    This is useful for ingestion by log aggregators (e.g. ELK, Datadog, CloudWatch).
    The ``session_id`` field serves as the correlation key across all events
    in a single user session.
    """

    async def on_orchestrator_start(self, event: OrchestratorStartEvent) -> None:
        _log("orchestrator.start", session_id=event.session_id, query=event.query)

    async def on_orchestrator_route(self, event: OrchestratorRouteEvent) -> None:
        _log("orchestrator.route", session_id=event.session_id, agent=event.agent_name)

    async def on_orchestrator_handoff(self, event: OrchestratorHandoffEvent) -> None:
        _log(
            "orchestrator.handoff",
            session_id=event.session_id,
            source=event.source_agent,
            target=event.target_agent,
            message=event.message,
        )

    async def on_orchestrator_finish(self, event: OrchestratorFinishEvent) -> None:
        _log(
            "orchestrator.finish",
            session_id=event.session_id,
            response_length=len(event.response),
        )

    async def on_orchestrator_error(self, event: OrchestratorErrorEvent) -> None:
        _log(
            "orchestrator.error",
            session_id=event.session_id,
            error_type=type(event.error).__name__,
            error_message=str(event.error),
        )

    async def on_agent_start(self, event: AgentStartEvent) -> None:
        _log("agent.start", session_id=event.session_id, agent=event.agent_name, query=event.query)

    async def on_agent_finish(self, event: AgentFinishEvent) -> None:
        _log(
            "agent.finish",
            session_id=event.session_id,
            agent=event.agent_name,
            response_length=len(event.response),
        )

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        _log(
            "tool.call",
            session_id=event.session_id,
            agent=event.agent_name,
            tool=event.tool_name,
            args=event.args,
        )

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        _log(
            "tool.result",
            session_id=event.session_id,
            agent=event.agent_name,
            tool=event.tool_name,
            result_length=len(event.result),
        )
