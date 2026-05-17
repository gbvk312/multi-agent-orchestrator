import json
from typing import Any, cast

import aiosqlite

from .memory import MemoryBackend


class SQLiteMemoryBackend(MemoryBackend):
    """SQLite-based memory backend for persistent storage."""

    def __init__(self, db_path: str = "memory.db", table_name: str = "sessions"):
        self.db_path = db_path
        self.table_name = table_name
        self._initialized = False

    async def _init_db(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table_name} (session_id TEXT PRIMARY KEY, history TEXT, state TEXT)"
            )
            await db.commit()
        self._initialized = True

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        await self._init_db()
        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(f"SELECT history FROM {self.table_name} WHERE session_id = ?", (session_id,)) as cursor,
        ):
            row = await cursor.fetchone()
            if row:
                return cast(list[dict[str, Any]], json.loads(row[0]))
        return []

    async def save(self, session_id: str, history: list[dict[str, Any]]) -> None:
        await self._init_db()
        history_json = json.dumps(history)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"INSERT INTO {self.table_name} (session_id, history, state) "
                f"VALUES (?, ?, '{{}}') "
                f"ON CONFLICT(session_id) DO UPDATE SET history=excluded.history",
                (session_id, history_json),
            )
            await db.commit()

    async def delete(self, session_id: str) -> None:
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"DELETE FROM {self.table_name} WHERE session_id = ?", (session_id,))
            await db.commit()

    async def load_state(self, session_id: str) -> dict[str, Any]:
        await self._init_db()
        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(f"SELECT state FROM {self.table_name} WHERE session_id = ?", (session_id,)) as cursor,
        ):
            row = await cursor.fetchone()
            if row and row[0]:
                return cast(dict[str, Any], json.loads(row[0]))
        return {}

    async def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        await self._init_db()
        state_json = json.dumps(state)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"INSERT INTO {self.table_name} (session_id, history, state) "
                f"VALUES (?, '[]', ?) "
                f"ON CONFLICT(session_id) DO UPDATE SET state=excluded.state",
                (session_id, state_json),
            )
            await db.commit()
