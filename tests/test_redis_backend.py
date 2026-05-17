import json
from unittest.mock import AsyncMock, patch

import pytest

from multi_agent_orchestrator.core.redis_backend import RedisMemoryBackend


@pytest.fixture
def mock_redis():
    with patch("multi_agent_orchestrator.core.redis_backend.redis.from_url") as mock_from_url:
        mock_instance = AsyncMock()
        mock_from_url.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_redis_backend_load_empty(mock_redis):
    mock_redis.get.return_value = None
    backend = RedisMemoryBackend()

    history = await backend.load("session1")
    assert history == []
    mock_redis.get.assert_called_once_with("mao:session:session1")


@pytest.mark.asyncio
async def test_redis_backend_save_and_load(mock_redis):
    backend = RedisMemoryBackend(prefix="test:")
    test_history = [{"role": "user", "content": "hello"}]

    # Save
    await backend.save("session2", test_history)
    mock_redis.set.assert_called_once_with("test:session2", json.dumps(test_history))

    # Load
    mock_redis.get.return_value = json.dumps(test_history)
    history = await backend.load("session2")
    assert history == test_history


@pytest.mark.asyncio
async def test_redis_backend_delete(mock_redis):
    backend = RedisMemoryBackend()

    await backend.delete("session3")
    assert mock_redis.delete.call_count == 2
    mock_redis.delete.assert_any_call("mao:session:session3")
    mock_redis.delete.assert_any_call("mao:session:session3:state")


@pytest.mark.asyncio
async def test_redis_backend_state(mock_redis):
    backend = RedisMemoryBackend()
    test_state = {"user_theme": "dark", "tokens": 120}

    # Load empty state
    mock_redis.get.return_value = None
    state = await backend.load_state("session4")
    assert state == {}
    mock_redis.get.assert_called_with("mao:session:session4:state")

    # Save state (no TTL)
    await backend.save_state("session4", test_state)
    mock_redis.set.assert_called_with("mao:session:session4:state", json.dumps(test_state))

    # Load state
    mock_redis.get.return_value = json.dumps(test_state)
    state = await backend.load_state("session4")
    assert state == test_state


@pytest.mark.asyncio
async def test_redis_backend_ttl(mock_redis):
    backend = RedisMemoryBackend(ttl_seconds=3600)
    test_history = [{"role": "user", "content": "hello"}]
    test_state = {"user_theme": "dark"}

    # Save history with custom TTL
    await backend.save("session5", test_history)
    mock_redis.set.assert_called_with("mao:session:session5", json.dumps(test_history), ex=3600)

    # Save state with custom TTL
    await backend.save_state("session5", test_state)
    mock_redis.set.assert_called_with("mao:session:session5:state", json.dumps(test_state), ex=3600)


@pytest.mark.asyncio
async def test_redis_backend_close(mock_redis):
    backend = RedisMemoryBackend()
    await backend.close()
    mock_redis.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_redis_backend_custom_client():
    custom_client = AsyncMock()
    backend = RedisMemoryBackend(redis_client=custom_client)
    assert backend._redis == custom_client
    assert backend._owns_client is False

    await backend.close()
    custom_client.aclose.assert_not_called()
