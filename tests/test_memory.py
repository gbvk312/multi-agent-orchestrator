import pytest
from multi_agent_orchestrator.core.memory import MemoryManager

def test_memory_manager_initialization():
    memory = MemoryManager()
    assert memory.sessions == {}

def test_add_message():
    memory = MemoryManager()
    session_id = "test_session"
    memory.add_message(session_id, "user", "hello")
    
    assert session_id in memory.sessions
    assert len(memory.sessions[session_id]) == 1
    assert memory.sessions[session_id][0] == {"role": "user", "content": "hello"}

def test_get_history():
    memory = MemoryManager()
    session_id = "test_session"
    memory.add_message(session_id, "user", "hello")
    memory.add_message(session_id, "model", "hi there")
    
    history = memory.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "model"

def test_get_history_empty():
    memory = MemoryManager()
    assert memory.get_history("non_existent") == []

def test_clear_session():
    memory = MemoryManager()
    session_id = "test_session"
    memory.add_message(session_id, "user", "hello")
    assert session_id in memory.sessions
    
    memory.clear(session_id)
    assert session_id not in memory.sessions

def test_clear_non_existent_session():
    memory = MemoryManager()
    # Should not raise error
    memory.clear("non_existent")
