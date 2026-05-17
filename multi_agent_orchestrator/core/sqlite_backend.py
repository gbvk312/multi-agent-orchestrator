import asyncio
import json
from typing import Any, cast

import aiosqlite

from .memory import MemoryBackend


class SQLiteMemoryBackend(MemoryBackend):
    """SQLite-based memory backend for persistent storage."""

    def __init__(self, db_path: str = "memory.db", table_name: str = "sessions"):
        if not table_name.isidentifier():
            raise ValueError(f"Invalid SQLite table name: '{table_name}'. Must be a valid identifier.")
        self.db_path = db_path
        self.table_name = table_name
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _get_db(self) -> aiosqlite.Connection:
        async with self._lock:
            if self._db is None:
                self._db = await aiosqlite.connect(self.db_path)
                await self._db.execute("PRAGMA journal_mode=WAL;")
                await self._db.execute(
                    f"CREATE TABLE IF NOT EXISTS {self.table_name} "
                    f"(session_id TEXT PRIMARY KEY, history TEXT, state TEXT)"
                )
                await self._db.commit()
            return self._db

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        async with db.execute(
            f"SELECT history FROM {self.table_name} WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return cast(list[dict[str, Any]], json.loads(row[0]))
        return []

    async def save(self, session_id: str, history: list[dict[str, Any]]) -> None:
        db = await self._get_db()
        history_json = json.dumps(history)
        await db.execute(
            f"INSERT INTO {self.table_name} (session_id, history, state) "
            f"VALUES (?, ?, '{{}}') "
            f"ON CONFLICT(session_id) DO UPDATE SET history=excluded.history",
            (session_id, history_json),
        )
        await db.commit()

    async def delete(self, session_id: str) -> None:
        db = await self._get_db()
        await db.execute(f"DELETE FROM {self.table_name} WHERE session_id = ?", (session_id,))
        await db.commit()

    async def load_state(self, session_id: str) -> dict[str, Any]:
        db = await self._get_db()
        async with db.execute(
            f"SELECT state FROM {self.table_name} WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return cast(dict[str, Any], json.loads(row[0]))
        return {}

    async def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        db = await self._get_db()
        state_json = json.dumps(state)
        await db.execute(
            f"INSERT INTO {self.table_name} (session_id, history, state) "
            f"VALUES (?, '[]', ?) "
            f"ON CONFLICT(session_id) DO UPDATE SET state=excluded.state",
            (session_id, state_json),
        )
        await db.commit()

    async def close(self) -> None:
        """Close the active connection cleanly."""
        async with self._lock:
            if self._db is not None:
                await self._db.close()
                self._db = None
