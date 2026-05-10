import pytest
from unittest.mock import MagicMock, patch
from multi_agent_orchestrator.core.orchestrator import Orchestrator
from multi_agent_orchestrator.core.agent import BaseAgent
from multi_agent_orchestrator.core.memory import MemoryManager

@patch('multi_agent_orchestrator.core.orchestrator.genai.Client')
def test_orchestrator_initialization(mock_client):
    orchestrator = Orchestrator()
    assert orchestrator.agents == {}
    assert isinstance(orchestrator.memory, MemoryManager)
    mock_client.assert_called_once()

@patch('multi_agent_orchestrator.core.orchestrator.genai.Client')
def test_register_agent(mock_client):
    orchestrator = Orchestrator()
    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.name = "TestAgent"
    
    orchestrator.register_agent(mock_agent)
    assert "TestAgent" in orchestrator.agents
    assert orchestrator.agents["TestAgent"] == mock_agent

@patch('multi_agent_orchestrator.core.orchestrator.genai.Client')
def test_route_request(mock_client_class):
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
    
    selected = orchestrator._route_request("How to code?")
    assert selected == "AgentB"
    mock_client.models.generate_content.assert_called_once()

@patch('multi_agent_orchestrator.core.orchestrator.genai.Client')
def test_route_request_fallback(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_response = MagicMock()
    mock_response.text = "UnknownAgent" # Not registered
    mock_client.models.generate_content.return_value = mock_response
    
    orchestrator = Orchestrator()
    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.system_prompt = "Prompt A"
    orchestrator.register_agent(agent_a)
    
    selected = orchestrator._route_request("Query")
    # Should fallback to the first agent (AgentA)
    assert selected == "AgentA"

@patch('multi_agent_orchestrator.core.orchestrator.genai.Client')
def test_process_request(mock_client_class):
    mock_client = mock_client_class.return_value
    
    # Mock routing response
    mock_route_response = MagicMock()
    mock_route_response.text = "AgentA"
    
    mock_client.models.generate_content.return_value = mock_route_response
    
    orchestrator = Orchestrator()
    
    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.name = "AgentA"
    mock_agent.system_prompt = "Test prompt"
    mock_agent.process.return_value = "Agent response"
    orchestrator.register_agent(mock_agent)
    
    response = orchestrator.process_request("session_1", "User query")
    
    assert response == "Agent response"
    mock_agent.process.assert_called_once_with("User query", [])
    
    # Verify memory was updated
    history = orchestrator.memory.get_history("session_1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "User query"}
    assert history[1] == {"role": "model", "content": "Agent response"}
