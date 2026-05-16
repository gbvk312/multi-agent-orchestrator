import pytest
from multi_agent_orchestrator.core.memory import MemoryManager, InMemoryBackend


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


def test_memory_manager_repr():
    memory = MemoryManager(max_history=100)
    assert repr(memory) == "MemoryManager(max_history=100, backend=InMemoryBackend)"
