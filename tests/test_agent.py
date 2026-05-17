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
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "Mocked response"
    mock_response.function_calls = None
    mock_client.aio.models.generate_content.return_value = mock_response

    agent = BaseAgent(name="TestAgent", system_prompt="System Prompt")
    history = [
        {"role": "user", "content": "previous query"},
        {"role": "model", "content": "previous response"},
    ]

    response = await agent.process("current query", history)

    assert response == "Mocked response"

    # Verify generate_content was called with correct structure
    mock_client.aio.models.generate_content.assert_called_once()
    _, kwargs = mock_client.aio.models.generate_content.call_args

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
    mock_client.aio.models.generate_content = AsyncMock()

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

    mock_client.aio.models.generate_content.side_effect = [mock_tool_response, mock_final_response]

    agent = BaseAgent(name="WeatherAgent", system_prompt="Weather prompt", tools=[get_weather])
    response = await agent.process("What's the weather?", [])

    assert response == "It's sunny in London!"
    assert mock_client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_unknown_tool(mock_client_class):
    """Agent should return an error string when model calls an unregistered tool."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()

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

    mock_client.aio.models.generate_content.side_effect = [mock_tool_response, mock_final_response]

    agent = BaseAgent(name="TestAgent", system_prompt="Prompt")
    response = await agent.process("Do something", [])

    assert response == "I couldn't find that tool."


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_retry_on_429(mock_client_class):
    """Verify exponential backoff on rate limit errors."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()

    mock_response = MagicMock()
    mock_response.text = "Success after retry"
    mock_response.function_calls = None

    # First call raises 429, second succeeds
    mock_client.aio.models.generate_content.side_effect = [
        Exception("429 Resource Exhausted"),
        mock_response,
    ]

    agent = BaseAgent(name="RetryAgent", system_prompt="Prompt")
    response = await agent.process("query", [])

    assert response == "Success after retry"
    assert mock_client.aio.models.generate_content.call_count == 2


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
    mock_client.aio.models.generate_content = AsyncMock()
    agent = BaseAgent(name="MaxToolAgent", system_prompt="Prompt", max_tool_rounds=1, tools=[lambda x: x])

    mock_fn_call = MagicMock()
    mock_fn_call.name = "<lambda>"
    mock_fn_call.args = {"x": "y"}

    mock_tool_response = MagicMock()
    mock_tool_response.function_calls = [mock_fn_call]
    mock_tool_response.candidates = [MagicMock()]
    mock_tool_response.candidates[0].content = types.Content(role="model", parts=[])

    mock_client.aio.models.generate_content.return_value = mock_tool_response

    response = await agent.process("query", [])
    assert "Reached maximum tool execution rounds" in response


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_unrecoverable_model_error(mock_client_class):
    """Verify generic model errors are raised immediately."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_client.aio.models.generate_content.side_effect = Exception("Some weird error")

    agent = BaseAgent(name="ErrorAgent", system_prompt="Prompt")
    with pytest.raises(AgentError, match="Model call failed: Some weird error"):
        await agent.process("query", [])


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_tool_exception(mock_client_class):
    """Verify tool exceptions are caught and returned as string."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()

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

    mock_client.aio.models.generate_content.side_effect = [mock_tool_response, mock_final_response]

    agent = BaseAgent(name="FailingToolAgent", system_prompt="Prompt", tools=[failing_tool])
    await agent.process("query", [])

    # We can check what was passed to the 2nd model call
    _, kwargs = mock_client.aio.models.generate_content.call_args
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
    mock_client.aio.models.generate_content = AsyncMock()

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

    mock_client.aio.models.generate_content.side_effect = [mock_tool_response, mock_final_response]

    agent = BaseAgent(name="AsyncToolAgent", system_prompt="Prompt", tools=[async_tool])
    response = await agent.process("query", [])

    assert response == "Handled Async"
    _, kwargs = mock_client.aio.models.generate_content.call_args
    contents = kwargs["contents"]
    fn_response_part = contents[-1].parts[0]
    assert "Async Result" in str(fn_response_part.function_response)


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_retries_exhausted(mock_client_class):
    """Verify that after max_retries of 429 errors, it raises AgentError."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_client.aio.models.generate_content.side_effect = Exception("429 Too Many Requests")

    agent = BaseAgent(name="ExhaustedAgent", system_prompt="Prompt", max_retries=2)
    # Patch asyncio.sleep so we don't actually wait
    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(AgentError, match="All 2 retries exhausted"),
    ):
        await agent.process("query", [])

    assert mock_client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_no_candidates(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.candidates = []
    mock_client.aio.models.generate_content.return_value = mock_response

    agent = BaseAgent(name="TestAgent", system_prompt="Prompt")
    response = await agent.process("query", [])
    assert "Generation failed or blocked by safety settings" in response


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_response_schema(mock_client_class):
    """Verify that response_schema is properly passed to GenerateContentConfig."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = '{"some": "json"}'
    mock_response.function_calls = None
    mock_client.aio.models.generate_content.return_value = mock_response

    schema = {"type": "object", "properties": {"some": {"type": "string"}}}
    agent = BaseAgent(name="SchemaAgent", system_prompt="Prompt", response_schema=schema)
    response = await agent.process("query", [])

    assert response == '{"some": "json"}'
    mock_client.aio.models.generate_content.assert_called_once()
    _, kwargs = mock_client.aio.models.generate_content.call_args
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_schema == schema


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_process_stream(mock_client_class):
    mock_client = mock_client_class.return_value

    # Mock stream chunks
    class MockChunk:
        def __init__(self, text=None, function_calls=None, candidates=None):
            self.text = text
            self.function_calls = function_calls
            self.candidates = candidates

    # Define an async generator to mock generate_content_stream
    async def mock_stream_generator():
        yield MockChunk(text="Hello ")
        yield MockChunk(text="world!")

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream_generator())

    agent = BaseAgent(name="StreamAgent", system_prompt="Prompt")
    chunks = []
    async for chunk in agent.process_stream("query", []):
        chunks.append(chunk)

    assert chunks == ["Hello ", "world!"]


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_tool_handoff_and_hitl(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()

    from multi_agent_orchestrator.core.agent import AgentHandoff, HumanApprovalRequired

    def trigger_handoff():
        raise AgentHandoff("TargetAgent", "transfer message")

    def trigger_approval():
        raise HumanApprovalRequired("some_tool", {"param": 1}, "approve please")

    mock_fn_call_handoff = MagicMock()
    mock_fn_call_handoff.name = "trigger_handoff"
    mock_fn_call_handoff.args = {}

    mock_tool_response_handoff = MagicMock()
    mock_tool_response_handoff.function_calls = [mock_fn_call_handoff]
    mock_tool_response_handoff.candidates = [MagicMock()]
    mock_tool_response_handoff.candidates[0].content = types.Content(role="model", parts=[])

    mock_client.aio.models.generate_content.return_value = mock_tool_response_handoff

    agent = BaseAgent(name="TestAgent", system_prompt="Prompt", tools=[trigger_handoff, trigger_approval])

    with pytest.raises(AgentHandoff) as exc_info:
        await agent.process("query", [])
    assert exc_info.value.target_agent == "TargetAgent"
    assert exc_info.value.message == "transfer message"

    # Now test HumanApprovalRequired
    mock_fn_call_approval = MagicMock()
    mock_fn_call_approval.name = "trigger_approval"
    mock_fn_call_approval.args = {}

    mock_tool_response_approval = MagicMock()
    mock_tool_response_approval.function_calls = [mock_fn_call_approval]
    mock_tool_response_approval.candidates = [MagicMock()]
    mock_tool_response_approval.candidates[0].content = types.Content(role="model", parts=[])

    mock_client.aio.models.generate_content.return_value = mock_tool_response_approval

    with pytest.raises(HumanApprovalRequired) as exc_info:
        await agent.process("query", [])
    assert exc_info.value.tool_name == "some_tool"
    assert exc_info.value.tool_args == {"param": 1}
    assert exc_info.value.message == "approve please"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_with_event_handler(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "Handled"
    mock_response.function_calls = None
    mock_client.aio.models.generate_content.return_value = mock_response

    from multi_agent_orchestrator.core.events import EventHandler
    handler = MagicMock(spec=EventHandler)
    handler.on_event = AsyncMock()

    agent = BaseAgent(name="EventAgent", system_prompt="Prompt")
    await agent.process("query", [], event_handler=handler)

    assert handler.on_event.call_count == 2  # Start and finish events


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_process_stream_with_tool_calls(mock_client_class):
    mock_client = mock_client_class.return_value

    class MockChunk:
        def __init__(self, text=None, function_calls=None, candidates=None):
            self.text = text
            self.function_calls = function_calls
            self.candidates = candidates

    # Round 1: Yields a tool call
    mock_fn_call = MagicMock()
    mock_fn_call.name = "my_tool"
    mock_fn_call.args = {"val": 123}

    async def mock_stream_1():
        yield MockChunk(function_calls=[mock_fn_call])

    # Round 2: Yields final text
    async def mock_stream_2():
        yield MockChunk(text="Finished")

    mock_client.aio.models.generate_content_stream = AsyncMock(side_effect=[mock_stream_1(), mock_stream_2()])

    def my_tool(val: int) -> str:
        return f"Tool got {val}"

    from multi_agent_orchestrator.core.events import EventHandler
    handler = MagicMock(spec=EventHandler)
    handler.on_event = AsyncMock()

    agent = BaseAgent(name="StreamAgent", system_prompt="Prompt", tools=[my_tool])

    # We also pass some history and response schema to cover those lines!
    agent.response_schema = {"type": "string"}
    history = [{"role": "user", "content": "prev"}, {"role": "model", "content": "resp"}]

    chunks = []
    async for chunk in agent.process_stream("query", history, event_handler=handler):
        chunks.append(chunk)

    assert chunks == ["Finished"]
    assert handler.on_event.call_count > 0
