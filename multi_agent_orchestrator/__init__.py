"""Multi-Agent Orchestrator: An async, Gemini-powered multi-agent framework."""

__version__ = "0.1.0"

from .core import AgentError, BaseAgent, MemoryManager, Orchestrator, OrchestratorError
from .core.config import OrchestratorConfig
from .core.memory import InMemoryBackend, MemoryBackend

__all__ = [
    "__version__",
    "BaseAgent",
    "AgentError",
    "Orchestrator",
    "OrchestratorError",
    "MemoryManager",
    "MemoryBackend",
    "InMemoryBackend",
    "OrchestratorConfig",
]
