import pytest

from multi_agent_orchestrator.core.memory import InMemoryBackend, MemoryManager


@pytest.mark.asyncio
async def test_memory_manager_initialization():
    memory = MemoryManager()
    history = await memory.get_history("non_existent")
    assert history == []


@pytest.mark.asyncio
async def test_add_message():
    memory = MemoryManager()
    session_id = "test_session"
    await memory.add_message(session_id, "user", "hello")

    history = await memory.get_history(session_id)
    assert len(history) == 1
    assert history[0] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_get_history():
    memory = MemoryManager()
    session_id = "test_session"
    await memory.add_message(session_id, "user", "hello")
    await memory.add_message(session_id, "model", "hi there")

    history = await memory.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "model"


@pytest.mark.asyncio
async def test_get_history_empty():
    memory = MemoryManager()
    assert await memory.get_history("non_existent") == []


@pytest.mark.asyncio
async def test_clear_session():
    memory = MemoryManager()
    session_id = "test_session"
    await memory.add_message(session_id, "user", "hello")

    history = await memory.get_history(session_id)
    assert len(history) == 1

    await memory.clear(session_id)
    assert await memory.get_history(session_id) == []


@pytest.mark.asyncio
async def test_clear_non_existent_session():
    memory = MemoryManager()
    # Should not raise error
    await memory.clear("non_existent")


@pytest.mark.asyncio
async def test_max_history_enforcement():
    memory = MemoryManager(max_history=3)
    session_id = "bounded"

    for i in range(5):
        await memory.add_message(session_id, "user", f"msg-{i}")

    history = await memory.get_history(session_id)
    assert len(history) == 3
    # Should keep the most recent 3
    assert history[0]["content"] == "msg-2"
    assert history[2]["content"] == "msg-4"


@pytest.mark.asyncio
async def test_custom_backend():
    """Verify that a custom MemoryBackend is used when provided."""
    backend = InMemoryBackend()
    memory = MemoryManager(backend=backend)
    await memory.add_message("s1", "user", "hello")

    # Directly check the backend
    stored = await backend.load("s1")
    assert len(stored) == 1
    assert stored[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_per_session_lock_isolation():
    """Verify that per-session locks are created independently."""
    memory = MemoryManager()
    await memory.add_message("s1", "user", "msg1")
    await memory.add_message("s2", "user", "msg2")

    # Each session should have its own lock
    assert "s1" in memory._locks
    assert "s2" in memory._locks
    assert memory._locks["s1"] is not memory._locks["s2"]


@pytest.mark.asyncio
async def test_clear_removes_session_lock():
    """Clearing a session should also clean up its lock entry."""
    memory = MemoryManager()
    await memory.add_message("s1", "user", "msg")
    assert "s1" in memory._locks

    await memory.clear("s1")
    assert "s1" not in memory._locks


def test_memory_manager_repr():
    memory = MemoryManager(max_history=100)
    assert repr(memory) == "MemoryManager(max_history=100, backend=InMemoryBackend)"


@pytest.mark.asyncio
async def test_state_management():
    memory = MemoryManager()
    session_id = "state_session"

    # Initial state should be empty
    state = await memory.get_state(session_id)
    assert state == {}

    # Update state
    await memory.update_state(session_id, {"user_name": "Alice", "step": 1})
    state = await memory.get_state(session_id)
    assert state == {"user_name": "Alice", "step": 1}

    # Partial update
    await memory.update_state(session_id, {"step": 2, "complete": True})
    state = await memory.get_state(session_id)
    assert state == {"user_name": "Alice", "step": 2, "complete": True}


@pytest.mark.asyncio
async def test_state_deletion_on_clear():
    memory = MemoryManager()
    session_id = "state_session"
    await memory.update_state(session_id, {"foo": "bar"})

    await memory.clear(session_id)
    state = await memory.get_state(session_id)
    assert state == {}
