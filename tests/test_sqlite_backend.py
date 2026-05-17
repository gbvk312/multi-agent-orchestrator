import pytest

from multi_agent_orchestrator.core.sqlite_backend import SQLiteMemoryBackend


@pytest.fixture
def sqlite_backend(tmp_path):
    # Use a temporary file database for testing
    db_file = tmp_path / "test_memory.db"
    return SQLiteMemoryBackend(db_path=str(db_file))


@pytest.mark.asyncio
async def test_sqlite_backend_load_empty(sqlite_backend):
    history = await sqlite_backend.load("session1")
    assert history == []


@pytest.mark.asyncio
async def test_sqlite_backend_save_and_load(sqlite_backend):
    test_history = [{"role": "user", "content": "hello"}]

    # Save
    await sqlite_backend.save("session2", test_history)

    # Load
    history = await sqlite_backend.load("session2")
    assert history == test_history


@pytest.mark.asyncio
async def test_sqlite_backend_update(sqlite_backend):
    test_history_1 = [{"role": "user", "content": "hello"}]
    test_history_2 = [{"role": "user", "content": "hello"}, {"role": "model", "content": "world"}]

    # Save first
    await sqlite_backend.save("session3", test_history_1)

    # Update
    await sqlite_backend.save("session3", test_history_2)

    # Load and verify it was updated
    history = await sqlite_backend.load("session3")
    assert history == test_history_2


@pytest.mark.asyncio
async def test_sqlite_backend_delete(sqlite_backend):
    test_history = [{"role": "user", "content": "hello"}]

    # Save
    await sqlite_backend.save("session4", test_history)

    # Verify saved
    history = await sqlite_backend.load("session4")
    assert history == test_history

    # Delete
    await sqlite_backend.delete("session4")

    # Verify deleted
    history = await sqlite_backend.load("session4")
    assert history == []


@pytest.mark.asyncio
async def test_sqlite_backend_state(sqlite_backend):
    # Verify default state is empty
    state = await sqlite_backend.load_state("session_state_1")
    assert state == {}

    # Save state
    test_state = {"user_theme": "dark", "tokens": 120}
    await sqlite_backend.save_state("session_state_1", test_state)

    # Load and assert
    loaded_state = await sqlite_backend.load_state("session_state_1")
    assert loaded_state == test_state

    # Override state
    test_state_2 = {"user_theme": "light"}
    await sqlite_backend.save_state("session_state_1", test_state_2)
    loaded_state_2 = await sqlite_backend.load_state("session_state_1")
    assert loaded_state_2 == test_state_2


@pytest.mark.asyncio
async def test_sqlite_backend_wal_mode(sqlite_backend):
    # Initialize the database by calling save
    await sqlite_backend.save("session_wal", [])

    import aiosqlite

    async with aiosqlite.connect(sqlite_backend.db_path) as db, db.execute("PRAGMA journal_mode;") as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row[0].lower() == "wal"


@pytest.mark.asyncio
async def test_sqlite_backend_invalid_table_name():
    with pytest.raises(ValueError, match="Invalid SQLite table name"):
        SQLiteMemoryBackend(table_name="semi;colon")
    with pytest.raises(ValueError, match="Invalid SQLite table name"):
        SQLiteMemoryBackend(table_name="123_invalid_start")


@pytest.mark.asyncio
async def test_sqlite_backend_close(sqlite_backend):
    # Perform operations to initialize cached connection
    await sqlite_backend.save("session_close", [])
    assert sqlite_backend._db is not None

    # Close it
    await sqlite_backend.close()
    assert sqlite_backend._db is None
