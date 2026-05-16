import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class MemoryBackend(ABC):
    """Abstract base for pluggable memory storage backends.

    Implement this to provide Redis, SQLite, or other persistent storage.
    """

    @abstractmethod
    async def load(self, session_id: str) -> List[Dict[str, Any]]:
        """Load conversation history for a session."""

    @abstractmethod
    async def save(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        """Persist conversation history for a session."""

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete conversation history for a session."""


class InMemoryBackend(MemoryBackend):
    """In-process dictionary-based memory backend (default)."""

    def __init__(self):
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    async def load(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._store.get(session_id, []))

    async def save(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        self._store[session_id] = history

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class MemoryManager:
    """Manages conversation context across different agents.

    Provides bounded history management with pluggable storage backends.
    Thread-safe for concurrent async access via asyncio.Lock.
    """

    def __init__(self, max_history: int = 50, backend: MemoryBackend | None = None):
        self.max_history = max_history
        self._backend = backend or InMemoryBackend()
        self._lock = asyncio.Lock()

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Adds a message to the session's history."""
        async with self._lock:
            history = await self._backend.load(session_id)
            history.append({"role": role, "content": content})

            # Enforce max history to prevent unbounded memory growth
            if len(history) > self.max_history:
                history = history[-self.max_history :]

            await self._backend.save(session_id, history)

    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves the history for a given session."""
        async with self._lock:
            return await self._backend.load(session_id)

    async def clear(self, session_id: str) -> None:
        """Clears the history for a session."""
        async with self._lock:
            await self._backend.delete(session_id)

    def __repr__(self) -> str:
        return f"MemoryManager(max_history={self.max_history}, backend={self._backend.__class__.__name__})"
