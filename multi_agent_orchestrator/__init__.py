"""Multi-Agent Orchestrator: An async, Gemini-powered multi-agent framework."""

from __future__ import annotations

__version__ = "0.1.0"

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("multi-agent-orchestrator")
except Exception:  # pragma: no cover
    pass  # Fallback to hardcoded version above

from importlib import import_module
from typing import Any

__all__ = [
    "__version__",
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
    "BaseAgent": ".core",
    "AgentError": ".core",
    "AgentHandoff": ".core",
    "HumanApprovalRequired": ".core",
    "Orchestrator": ".core",
    "OrchestratorError": ".core",
    "MemoryManager": ".core",
    "MemoryBackend": ".core",
    "InMemoryBackend": ".core",
    "RedisMemoryBackend": ".core",
    "SQLiteMemoryBackend": ".core",
    "OrchestratorConfig": ".core",
    "EventHandler": ".core",
    "AgentStartEvent": ".core",
    "AgentFinishEvent": ".core",
    "ToolCallEvent": ".core",
    "ToolResultEvent": ".core",
    "OrchestratorStartEvent": ".core",
    "OrchestratorRouteEvent": ".core",
    "OrchestratorHandoffEvent": ".core",
    "OrchestratorFinishEvent": ".core",
    "OrchestratorErrorEvent": ".core",
}


def __getattr__(name: str) -> Any:
    """Lazily import exports to avoid requiring optional dependencies on package import."""
    module_path = _EXPORT_MAP.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
