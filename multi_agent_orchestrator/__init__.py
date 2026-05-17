"""Multi-Agent Orchestrator: An async, Gemini-powered multi-agent framework."""

__version__ = "0.1.0"

from .core import (
    AgentError,
    BaseAgent,
    InMemoryBackend,
    MemoryBackend,
    MemoryManager,
    Orchestrator,
    OrchestratorError,
    RedisMemoryBackend,
    SQLiteMemoryBackend,
)
from .core.config import OrchestratorConfig

__all__ = [
    "__version__",
    "BaseAgent",
    "AgentError",
    "Orchestrator",
    "OrchestratorError",
    "MemoryManager",
    "MemoryBackend",
    "InMemoryBackend",
    "RedisMemoryBackend",
    "SQLiteMemoryBackend",
    "OrchestratorConfig",
]
