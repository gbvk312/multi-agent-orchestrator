import json
from typing import Any, cast

import redis.asyncio as redis

from .memory import MemoryBackend


class RedisMemoryBackend(MemoryBackend):
    """Redis-based memory backend for persistent storage."""

    def __init__(
        self,
        redis_url: str = "redis://localhost",
        prefix: str = "mao:session:",
        ttl_seconds: int | None = None,
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self.ttl = ttl_seconds
        self._redis = redis.from_url(self.redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        data = await self._redis.get(self._key(session_id))
        if data:
            return cast(list[dict[str, Any]], json.loads(data))
        return []

    async def save(self, session_id: str, history: list[dict[str, Any]]) -> None:
        kwargs: dict[str, Any] = {}
        if self.ttl is not None:
            kwargs["ex"] = self.ttl
        await self._redis.set(self._key(session_id), json.dumps(history), **kwargs)

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
        await self._redis.delete(self._key(session_id) + ":state")

    async def load_state(self, session_id: str) -> dict[str, Any]:
        data = await self._redis.get(self._key(session_id) + ":state")
        if data:
            return cast(dict[str, Any], json.loads(data))
        return {}

    async def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        kwargs: dict[str, Any] = {}
        if self.ttl is not None:
            kwargs["ex"] = self.ttl
        await self._redis.set(self._key(session_id) + ":state", json.dumps(state), **kwargs)
