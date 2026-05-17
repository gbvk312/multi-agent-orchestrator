"""Core framework for the Multi-Agent Orchestrator."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseAgent",
    "AgentError",
    "AgentHandoff",
    "HumanApprovalRequired",
    "Orchestrator",
    "OrchestratorError",
    "MemoryManager",
    "MemoryBackend",
    "InMemoryBackend",
    "RedisMemoryBackend",
    "SQLiteMemoryBackend",
    "OrchestratorConfig",
    "EventHandler",
    "AgentStartEvent",
    "AgentFinishEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "OrchestratorStartEvent",
    "OrchestratorRouteEvent",
    "OrchestratorHandoffEvent",
    "OrchestratorFinishEvent",
    "OrchestratorErrorEvent",
]


_EXPORT_MAP = {
    "BaseAgent": ".agent",
    "AgentError": ".agent",
    "AgentHandoff": ".agent",
    "HumanApprovalRequired": ".agent",
    "Orchestrator": ".orchestrator",
    "OrchestratorError": ".orchestrator",
    "MemoryManager": ".memory",
    "MemoryBackend": ".memory",
    "InMemoryBackend": ".memory",
    "RedisMemoryBackend": ".redis_backend",
    "SQLiteMemoryBackend": ".sqlite_backend",
    "OrchestratorConfig": ".config",
    "EventHandler": ".events",
    "AgentStartEvent": ".events",
    "AgentFinishEvent": ".events",
    "ToolCallEvent": ".events",
    "ToolResultEvent": ".events",
    "OrchestratorStartEvent": ".events",
    "OrchestratorRouteEvent": ".events",
    "OrchestratorHandoffEvent": ".events",
    "OrchestratorFinishEvent": ".events",
    "OrchestratorErrorEvent": ".events",
}


def __getattr__(name: str) -> Any:
    """Lazily import exports to avoid importing optional dependencies at package import time."""
    module_path = _EXPORT_MAP.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
