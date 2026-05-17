import pytest


def test_top_level_lazy_imports():
    """Verify that importing main components from multi_agent_orchestrator triggers lazy loading successfully."""
    import multi_agent_orchestrator

    # Fetching attributes should work
    assert multi_agent_orchestrator.BaseAgent is not None
    assert multi_agent_orchestrator.AgentHandoff is not None
    assert multi_agent_orchestrator.HumanApprovalRequired is not None
    assert multi_agent_orchestrator.Orchestrator is not None
    assert multi_agent_orchestrator.MemoryManager is not None
    assert multi_agent_orchestrator.RedisMemoryBackend is not None
    assert multi_agent_orchestrator.SQLiteMemoryBackend is not None
    assert multi_agent_orchestrator.EventHandler is not None
    assert multi_agent_orchestrator.AgentStartEvent is not None
    assert multi_agent_orchestrator.OrchestratorStartEvent is not None

    # Invalid attribute should raise AttributeError
    with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
        _ = multi_agent_orchestrator.NonExistent


def test_core_lazy_imports():
    """Verify that importing from multi_agent_orchestrator.core triggers lazy loading successfully."""
    import multi_agent_orchestrator.core as core

    # Fetching attributes should work
    assert core.BaseAgent is not None
    assert core.AgentHandoff is not None
    assert core.HumanApprovalRequired is not None
    assert core.Orchestrator is not None
    assert core.MemoryManager is not None
    assert core.RedisMemoryBackend is not None
    assert core.SQLiteMemoryBackend is not None
    assert core.EventHandler is not None
    assert core.AgentStartEvent is not None
    assert core.OrchestratorStartEvent is not None

    # Invalid attribute should raise AttributeError
    with pytest.raises(AttributeError, match="has no attribute 'InvalidAttr'"):
        _ = core.InvalidAttr
