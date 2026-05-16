import pytest
from unittest.mock import MagicMock, patch
from multi_agent_orchestrator.core.agent import BaseAgent, AgentError
from google.genai import types


@patch("multi_agent_orchestrator.core.agent.genai.Client")
def test_base_agent_initialization(mock_client):
    agent = BaseAgent(name="TestAgent", system_prompt="You are a tester")
    assert agent.name == "TestAgent"
    assert agent.system_prompt == "You are a tester"
    assert agent.model == "gemini-2.5-flash"
    assert agent.tools == []
    assert agent.timeout == 120.0
    mock_client.assert_called_once()


@patch("multi_agent_orchestrator.core.agent.genai.Client")
def test_base_agent_repr(mock_client):
    agent = BaseAgent(name="TestAgent", system_prompt="Prompt")
    assert repr(agent) == "BaseAgent(name='TestAgent', model='gemini-2.5-flash', tools=0)"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_process(mock_client_class):
    """Verify that process() awaits the model and returns text."""
    mock_client = mock_client_class.return_value
    mock_response = MagicMock()
    mock_response.text = "Mocked response"
    mock_response.function_calls = None
    mock_client.models.generate_content.return_value = mock_response

    agent = BaseAgent(name="TestAgent", system_prompt="System Prompt")
    history = [
        {"role": "user", "content": "previous query"},
        {"role": "model", "content": "previous response"},
    ]

    response = await agent.process("current query", history)

    assert response == "Mocked response"

    # Verify generate_content was called with correct structure
    mock_client.models.generate_content.assert_called_once()
    _, kwargs = mock_client.models.generate_content.call_args

    assert kwargs["model"] == "gemini-2.5-flash"
    contents = kwargs["contents"]
    assert len(contents) == 3  # 2 from history + 1 current
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "current query"
    assert kwargs["config"].system_instruction == "System Prompt"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_tool_call(mock_client_class):
    """Verify the full tool-call loop: model requests tool -> agent executes -> model returns text."""
    mock_client = mock_client_class.return_value

    # Define a real tool function for the agent
    def get_weather(location: str) -> str:
        return f"Sunny in {location}"

    # First call: model requests a tool
    mock_fn_call = MagicMock()
    mock_fn_call.name = "get_weather"
    mock_fn_call.args = {"location": "London"}

    mock_tool_response = MagicMock()
    mock_tool_response.function_calls = [mock_fn_call]
    mock_tool_response.candidates = [MagicMock()]
    mock_tool_response.candidates[0].content = types.Content(role="model", parts=[])

    # Second call: model returns final text
    mock_final_response = MagicMock()
    mock_final_response.function_calls = None
    mock_final_response.text = "It's sunny in London!"

    mock_client.models.generate_content.side_effect = [mock_tool_response, mock_final_response]

    agent = BaseAgent(name="WeatherAgent", system_prompt="Weather prompt", tools=[get_weather])
    response = await agent.process("What's the weather?", [])

    assert response == "It's sunny in London!"
    assert mock_client.models.generate_content.call_count == 2


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_unknown_tool(mock_client_class):
    """Agent should return an error string when model calls an unregistered tool."""
    mock_client = mock_client_class.return_value

    # Model requests a tool the agent doesn't have
    mock_fn_call = MagicMock()
    mock_fn_call.name = "unknown_tool"
    mock_fn_call.args = {}

    mock_tool_response = MagicMock()
    mock_tool_response.function_calls = [mock_fn_call]
    mock_tool_response.candidates = [MagicMock()]
    mock_tool_response.candidates[0].content = types.Content(role="model", parts=[])

    # After receiving the error, model returns text
    mock_final_response = MagicMock()
    mock_final_response.function_calls = None
    mock_final_response.text = "I couldn't find that tool."

    mock_client.models.generate_content.side_effect = [mock_tool_response, mock_final_response]

    agent = BaseAgent(name="TestAgent", system_prompt="Prompt")
    response = await agent.process("Do something", [])

    assert response == "I couldn't find that tool."


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_retry_on_429(mock_client_class):
    """Verify exponential backoff on rate limit errors."""
    mock_client = mock_client_class.return_value

    mock_response = MagicMock()
    mock_response.text = "Success after retry"
    mock_response.function_calls = None

    # First call raises 429, second succeeds
    mock_client.models.generate_content.side_effect = [
        Exception("429 Resource Exhausted"),
        mock_response,
    ]

    agent = BaseAgent(name="RetryAgent", system_prompt="Prompt")
    response = await agent.process("query", [])

    assert response == "Success after retry"
    assert mock_client.models.generate_content.call_count == 2


def test_base_agent_missing_api_key(monkeypatch):
    """Agent should fail fast if no API key is available."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(AgentError, match="GEMINI_API_KEY"):
        BaseAgent(name="NoKeyAgent", system_prompt="Prompt")
