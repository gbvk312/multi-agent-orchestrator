import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any


class MemoryBackend(ABC):
    """Abstract base for pluggable memory storage backends.

    Implement this to provide Redis, SQLite, or other persistent storage.
    """

    @abstractmethod
    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """Load conversation history for a session."""

    @abstractmethod
    async def save(self, session_id: str, history: list[dict[str, Any]]) -> None:
        """Persist conversation history for a session."""

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete conversation history for a session."""

    @abstractmethod
    async def load_state(self, session_id: str) -> dict[str, Any]:
        """Load shared state for a session."""

    @abstractmethod
    async def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist shared state for a session."""

    async def close(self) -> None:
        """Cleanly close connection pools, file descriptors, or sockets."""
        # Default implementation is a no-op
        _ = self


class InMemoryBackend(MemoryBackend):
    """In-process dictionary-based memory backend (default)."""

    def __init__(self, ttl_seconds: int | None = None) -> None:
        self.ttl = ttl_seconds
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._state_store: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, float] = {}

    def _is_expired(self, session_id: str) -> bool:
        if self.ttl is None:
            return False
        last_updated = self._timestamps.get(session_id)
        if last_updated is None:
            return False
        return (time.time() - last_updated) > self.ttl

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        if self._is_expired(session_id):
            await self.delete(session_id)
            return []
        return list(self._store.get(session_id, []))

    async def save(self, session_id: str, history: list[dict[str, Any]]) -> None:
        self._store[session_id] = history
        if self.ttl is not None:
            self._timestamps[session_id] = time.time()

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        self._state_store.pop(session_id, None)
        self._timestamps.pop(session_id, None)

    async def load_state(self, session_id: str) -> dict[str, Any]:
        if self._is_expired(session_id):
            await self.delete(session_id)
            return {}
        return dict(self._state_store.get(session_id, {}))

    async def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        self._state_store[session_id] = state
        if self.ttl is not None:
            self._timestamps[session_id] = time.time()


class MemoryManager:
    """Manages conversation context across different agents.

    Provides bounded history management with pluggable storage backends.
    Uses per-session ``asyncio.Lock`` instances so concurrent sessions
    never block each other.
    """

    def __init__(self, max_history: int = 50, backend: MemoryBackend | None = None):
        self.max_history = max_history
        self._backend = backend or InMemoryBackend()
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Return the lock for *session_id*, creating one if needed.

        Safe to call without external synchronisation because asyncio
        runs on a single thread — dict mutations are never interleaved.
        """
        if session_id not in self._locks:
            # Periodic cleanup to prevent memory leak of idle locks
            if len(self._locks) > 1000:
                idle_sessions = [
                    sid for sid, lk in self._locks.items() if not lk.locked() and not getattr(lk, "_waiters", None)
                ]
                for sid in idle_sessions:
                    self._locks.pop(sid, None)
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def close(self) -> None:
        """Cleanly close the underlying storage backend and clear locks."""
        self._locks.clear()
        await self._backend.close()

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Adds a message to the session's history."""
        async with self._get_lock(session_id):
            history = await self._backend.load(session_id)
            history.append({"role": role, "content": content})

            # Enforce max history to prevent unbounded memory growth
            if len(history) > self.max_history:
                history = history[-self.max_history :]

            await self._backend.save(session_id, history)

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieves the history for a given session."""
        async with self._get_lock(session_id):
            return await self._backend.load(session_id)

    async def clear(self, session_id: str) -> None:
        """Clears the history for a session."""
        async with self._get_lock(session_id):
            await self._backend.delete(session_id)
            # Clean up the lock entry to prevent unbounded growth
            self._locks.pop(session_id, None)

    async def get_state(self, session_id: str) -> dict[str, Any]:
        """Retrieves the shared state for a given session."""
        async with self._get_lock(session_id):
            return await self._backend.load_state(session_id)

    async def update_state(self, session_id: str, updates: dict[str, Any]) -> None:
        """Updates the shared state with new key-value pairs."""
        async with self._get_lock(session_id):
            state = await self._backend.load_state(session_id)
            state.update(updates)
            await self._backend.save_state(session_id, state)

    def __repr__(self) -> str:
        return f"MemoryManager(max_history={self.max_history}, backend={self._backend.__class__.__name__})"
