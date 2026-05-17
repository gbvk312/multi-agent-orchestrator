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
