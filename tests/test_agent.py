import pytest
from unittest.mock import MagicMock, patch
from multi_agent_orchestrator.core.agent import BaseAgent
from google.genai import types

@patch('multi_agent_orchestrator.core.agent.genai.Client')
def test_base_agent_initialization(mock_client):
    agent = BaseAgent(name="TestAgent", system_prompt="You are a tester")
    assert agent.name == "TestAgent"
    assert agent.system_prompt == "You are a tester"
    assert agent.model == "gemini-2.5-flash"
    assert agent.tools == []
    mock_client.assert_called_once()

@patch('multi_agent_orchestrator.core.agent.genai.Client')
def test_base_agent_process(mock_client_class):
    # Setup mock client and response
    mock_client = mock_client_class.return_value
    mock_response = MagicMock()
    mock_response.text = "Mocked response"
    mock_response.function_calls = None
    mock_client.models.generate_content.return_value = mock_response
    
    agent = BaseAgent(name="TestAgent", system_prompt="System Prompt")
    history = [{"role": "user", "content": "previous query"}, {"role": "model", "content": "previous response"}]
    
    response = agent.process("current query", history)
    
    assert response == "Mocked response"
    
    # Verify generate_content was called with correct structure
    mock_client.models.generate_content.assert_called_once()
    args, kwargs = mock_client.models.generate_content.call_args
    
    assert kwargs['model'] == "gemini-2.5-flash"
    contents = kwargs['contents']
    assert len(contents) == 3 # 2 from history + 1 current
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "current query"
    
    assert kwargs['config'].system_instruction == "System Prompt"

@patch('multi_agent_orchestrator.core.agent.genai.Client')
def test_base_agent_tool_call(mock_client_class):
    # Setup mock client and response with function call
    mock_client = mock_client_class.return_value
    mock_response = MagicMock()
    mock_response.text = ""
    
    mock_function_call = MagicMock()
    mock_function_call.name = "get_weather"
    mock_function_call.args = {"location": "London"}
    mock_response.function_calls = [mock_function_call]
    
    mock_client.models.generate_content.return_value = mock_response
    
    agent = BaseAgent(name="WeatherAgent", system_prompt="Weather prompt")
    response = agent.process("What's the weather?", [])
    
    assert "Suggested tool call: get_weather" in response
    assert "London" in response
