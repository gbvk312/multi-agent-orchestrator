"""Multi-Agent Orchestrator: An async, Gemini-powered multi-agent framework."""

__version__ = "0.1.0"

from .core import BaseAgent, AgentError, Orchestrator, MemoryManager
from .core.config import OrchestratorConfig
from .core.memory import MemoryBackend, InMemoryBackend

__all__ = [
    "__version__",
    "BaseAgent",
    "AgentError",
    "Orchestrator",
    "MemoryManager",
    "MemoryBackend",
    "InMemoryBackend",
    "OrchestratorConfig",
]
