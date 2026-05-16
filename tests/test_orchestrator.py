import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from multi_agent_orchestrator.core.orchestrator import Orchestrator
from multi_agent_orchestrator.core.agent import BaseAgent
from multi_agent_orchestrator.core.memory import MemoryManager


@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
def test_orchestrator_initialization(mock_client):
    orchestrator = Orchestrator()
    assert orchestrator.agents == {}
    assert isinstance(orchestrator.memory, MemoryManager)
    mock_client.assert_called_once()


@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
def test_register_agent(mock_client):
    orchestrator = Orchestrator()
    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.name = "TestAgent"

    orchestrator.register_agent(mock_agent)
    assert "TestAgent" in orchestrator.agents
    assert orchestrator.agents["TestAgent"] == mock_agent


@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
def test_unregister_agent(mock_client):
    orchestrator = Orchestrator()
    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.name = "TestAgent"

    orchestrator.register_agent(mock_agent)
    assert orchestrator.unregister_agent("TestAgent") is True
    assert "TestAgent" not in orchestrator.agents

    # Removing non-existent agent returns False
    assert orchestrator.unregister_agent("Ghost") is False


@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
def test_register_duplicate_agent_warns(mock_client, caplog):
    """Registering an agent with a duplicate name should log a warning."""
    orchestrator = Orchestrator()

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentA"  # Same name

    orchestrator.register_agent(agent_a)

    import logging

    with caplog.at_level(logging.WARNING):
        orchestrator.register_agent(agent_b)

    assert "Overwriting existing agent: AgentA" in caplog.text
    assert orchestrator.agents["AgentA"] == agent_b


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_route_request(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_response = MagicMock()
    mock_response.text = "AgentB"
    mock_client.models.generate_content.return_value = mock_response

    orchestrator = Orchestrator()

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.system_prompt = "Prompt A"

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.system_prompt = "Prompt B"

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    selected = await orchestrator._route_request("How to code?")
    assert selected == "AgentB"
    mock_client.models.generate_content.assert_called_once()


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_route_request_single_agent_skips_llm(mock_client_class):
    """With only one agent, routing should skip the LLM call entirely."""
    mock_client = mock_client_class.return_value

    orchestrator = Orchestrator()
    agent = MagicMock(spec=BaseAgent)
    agent.name = "OnlyAgent"
    agent.system_prompt = "Prompt"
    orchestrator.register_agent(agent)

    selected = await orchestrator._route_request("Any query")
    assert selected == "OnlyAgent"

    # LLM should NOT be called
    mock_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_route_request_fallback(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_response = MagicMock()
    mock_response.text = "UnknownAgent"  # Not registered
    mock_client.models.generate_content.return_value = mock_response

    orchestrator = Orchestrator()
    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.system_prompt = "Prompt A"
    orchestrator.register_agent(agent_a)

    # Register a second agent to force the routing LLM call
    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.system_prompt = "Prompt B"
    orchestrator.register_agent(agent_b)

    selected = await orchestrator._route_request("Query")
    # Should fallback to the first agent (AgentA)
    assert selected == "AgentA"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_process_request(mock_client_class):
    mock_client = mock_client_class.return_value

    # Mock routing response
    mock_route_response = MagicMock()
    mock_route_response.text = "AgentA"
    mock_client.models.generate_content.return_value = mock_route_response

    orchestrator = Orchestrator()

    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.name = "AgentA"
    mock_agent.system_prompt = "Test prompt"
    mock_agent.process = AsyncMock(return_value="Agent response")
    orchestrator.register_agent(mock_agent)

    response = await orchestrator.process_request("session_1", "User query")

    assert response == "Agent response"
    # Agent receives empty history (no double-query bug)
    mock_agent.process.assert_called_once_with("User query", [])

    # Verify memory was updated AFTER processing
    history = await orchestrator.memory.get_history("session_1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "User query"}
    assert history[1] == {"role": "model", "content": "Agent response"}


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_fan_out(mock_client_class):
    """Verify parallel execution returns results from all agents."""
    orchestrator = Orchestrator()

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process = AsyncMock(return_value="Response A")

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.process = AsyncMock(return_value="Response B")

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    results = await orchestrator.fan_out("session_1", "Multi query")

    assert results == {"AgentA": "Response A", "AgentB": "Response B"}
    agent_a.process.assert_called_once()
    agent_b.process.assert_called_once()


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_fan_out_unknown_agent_raises(mock_client_class):
    """fan_out should raise ValueError for unknown agent names."""
    orchestrator = Orchestrator()

    agent = MagicMock(spec=BaseAgent)
    agent.name = "AgentA"
    orchestrator.register_agent(agent)

    with pytest.raises(ValueError, match="Unknown agents"):
        await orchestrator.fan_out("s1", "query", agent_names=["NonExistent"])


@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
def test_orchestrator_repr(mock_client):
    orchestrator = Orchestrator()
    assert "Orchestrator" in repr(orchestrator)
    assert "gemini-2.5-flash" in repr(orchestrator)
