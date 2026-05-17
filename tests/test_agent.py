from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from multi_agent_orchestrator.core.agent import AgentError, BaseAgent


@patch("multi_agent_orchestrator.core.agent.genai.Client")
def test_base_agent_initialization(mock_client):
    agent = BaseAgent(name="TestAgent", system_prompt="You are a tester")
    assert agent.name == "TestAgent"
    assert agent.system_prompt == "You are a tester"
    assert agent.model == "gemini-2.5-flash"
    assert agent.tools == []
    assert agent.timeout == 120.0
    assert agent.temperature == 0.2
    assert agent.max_tool_rounds == 5
    mock_client.assert_called_once()


@patch("multi_agent_orchestrator.core.agent.genai.Client")
def test_base_agent_explicit_params_override_config(mock_client):
    """Explicit constructor params should take precedence over config defaults."""
    agent = BaseAgent(
        name="Custom",
        system_prompt="Prompt",
        model="gemini-2.0-flash",
        max_retries=1,
        timeout=30.0,
        temperature=0.8,
        max_tool_rounds=2,
    )
    assert agent.model == "gemini-2.0-flash"
    assert agent.max_retries == 1
    assert agent.timeout == 30.0
    assert agent.temperature == 0.8
    assert agent.max_tool_rounds == 2


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
    assert kwargs["config"].temperature == 0.2


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


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_timeout(mock_client_class):
    """Verify that process() raises AgentError on TimeoutError."""
    agent = BaseAgent(name="TimeoutAgent", system_prompt="Prompt", timeout=0.01)

    async def slow_process(*args, **kwargs):
        import asyncio

        await asyncio.sleep(0.1)
        return "Done"

    with (
        patch.object(agent, "_process_inner", side_effect=slow_process),
        pytest.raises(AgentError, match="Processing timed out"),
    ):
        await agent.process("query", [])


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_max_tool_rounds(mock_client_class):
    """Verify it stops after max_tool_rounds."""
    mock_client = mock_client_class.return_value
    agent = BaseAgent(name="MaxToolAgent", system_prompt="Prompt", max_tool_rounds=1, tools=[lambda x: x])

    mock_fn_call = MagicMock()
    mock_fn_call.name = "<lambda>"
    mock_fn_call.args = {"x": "y"}

    mock_tool_response = MagicMock()
    mock_tool_response.function_calls = [mock_fn_call]
    mock_tool_response.candidates = [MagicMock()]
    mock_tool_response.candidates[0].content = types.Content(role="model", parts=[])

    mock_client.models.generate_content.return_value = mock_tool_response

    response = await agent.process("query", [])
    assert "Reached maximum tool execution rounds" in response


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_unrecoverable_model_error(mock_client_class):
    """Verify generic model errors are raised immediately."""
    mock_client = mock_client_class.return_value
    mock_client.models.generate_content.side_effect = Exception("Some weird error")

    agent = BaseAgent(name="ErrorAgent", system_prompt="Prompt")
    with pytest.raises(AgentError, match="Model call failed: Some weird error"):
        await agent.process("query", [])


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_tool_exception(mock_client_class):
    """Verify tool exceptions are caught and returned as string."""
    mock_client = mock_client_class.return_value

    def failing_tool():
        raise ValueError("Tool failed miserably")

    mock_fn_call = MagicMock()
    mock_fn_call.name = "failing_tool"
    mock_fn_call.args = {}

    mock_tool_response = MagicMock()
    mock_tool_response.function_calls = [mock_fn_call]
    mock_tool_response.candidates = [MagicMock()]
    mock_tool_response.candidates[0].content = types.Content(role="model", parts=[])

    mock_final_response = MagicMock()
    mock_final_response.function_calls = None
    mock_final_response.text = "Handled"

    mock_client.models.generate_content.side_effect = [mock_tool_response, mock_final_response]

    agent = BaseAgent(name="FailingToolAgent", system_prompt="Prompt", tools=[failing_tool])
    await agent.process("query", [])

    # We can check what was passed to the 2nd model call
    _, kwargs = mock_client.models.generate_content.call_args
    contents = kwargs["contents"]
    fn_response_part = contents[-1].parts[0]
    # Check if the string representation contains the error message
    assert "Error executing tool" in str(fn_response_part.function_response)
    assert "Tool failed miserably" in str(fn_response_part.function_response)


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_async_tool(mock_client_class):
    """Verify async tools are supported."""
    mock_client = mock_client_class.return_value

    async def async_tool():
        return "Async Result"

    mock_fn_call = MagicMock()
    mock_fn_call.name = "async_tool"
    mock_fn_call.args = {}

    mock_tool_response = MagicMock()
    mock_tool_response.function_calls = [mock_fn_call]
    mock_tool_response.candidates = [MagicMock()]
    mock_tool_response.candidates[0].content = types.Content(role="model", parts=[])

    mock_final_response = MagicMock()
    mock_final_response.function_calls = None
    mock_final_response.text = "Handled Async"

    mock_client.models.generate_content.side_effect = [mock_tool_response, mock_final_response]

    agent = BaseAgent(name="AsyncToolAgent", system_prompt="Prompt", tools=[async_tool])
    response = await agent.process("query", [])

    assert response == "Handled Async"
    _, kwargs = mock_client.models.generate_content.call_args
    contents = kwargs["contents"]
    fn_response_part = contents[-1].parts[0]
    assert "Async Result" in str(fn_response_part.function_response)


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_retries_exhausted(mock_client_class):
    """Verify that after max_retries of 429 errors, it raises AgentError."""
    mock_client = mock_client_class.return_value
    mock_client.models.generate_content.side_effect = Exception("429 Too Many Requests")

    agent = BaseAgent(name="ExhaustedAgent", system_prompt="Prompt", max_retries=2)
    # Patch asyncio.sleep so we don't actually wait
    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(AgentError, match="All 2 retries exhausted"),
    ):
        await agent.process("query", [])
    
    assert mock_client.models.generate_content.call_count == 2
