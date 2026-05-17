from collections.abc import AsyncGenerator, Callable
from typing import Any
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
    from collections.abc import AsyncGenerator
    from typing import Any

    mock_client = mock_client_class.return_value

    # Mock stream chunks
    class MockChunk:
        def __init__(
            self,
            text: str | None = None,
            function_calls: list[Any] | None = None,
            candidates: list[Any] | None = None,
        ) -> None:
            self.text = text
            self.function_calls = function_calls
            self.candidates = candidates

    # Define an async generator to mock generate_content_stream
    async def mock_stream_generator() -> AsyncGenerator[MockChunk, None]:
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

    with pytest.raises(HumanApprovalRequired) as exc_info_hitl:
        await agent.process("query", [])
    assert exc_info_hitl.value.tool_name == "some_tool"
    assert exc_info_hitl.value.tool_args == {"param": 1}
    assert exc_info_hitl.value.message == "approve please"


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
    from collections.abc import AsyncGenerator
    from typing import Any

    mock_client = mock_client_class.return_value

    class MockChunk:
        def __init__(
            self,
            text: str | None = None,
            function_calls: list[Any] | None = None,
            candidates: list[Any] | None = None,
        ) -> None:
            self.text = text
            self.function_calls = function_calls
            self.candidates = candidates

    # Round 1: Yields a tool call
    mock_fn_call = MagicMock()
    mock_fn_call.name = "my_tool"
    mock_fn_call.args = {"val": 123}

    async def mock_stream_1() -> AsyncGenerator[MockChunk, None]:
        yield MockChunk(function_calls=[mock_fn_call])

    # Round 2: Yields final text
    async def mock_stream_2() -> AsyncGenerator[MockChunk, None]:
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


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_tool_event_handler(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()

    mock_fn_call = MagicMock()
    mock_fn_call.name = "my_tool"
    mock_fn_call.args = None  # test args_dict = {}

    mock_response = MagicMock()
    mock_response.function_calls = [mock_fn_call]
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = types.Content(role="model", parts=[])

    mock_final_response = MagicMock()
    mock_final_response.function_calls = None
    mock_final_response.text = "Final Output"

    mock_client.aio.models.generate_content.side_effect = [mock_response, mock_final_response]

    def my_tool() -> str:
        return "result"

    from multi_agent_orchestrator.core.events import EventHandler

    handler = MagicMock(spec=EventHandler)
    handler.on_event = AsyncMock()

    agent = BaseAgent(name="ToolAgent", system_prompt="Prompt", tools=[my_tool])
    res = await agent.process("query", [], event_handler=handler)

    assert res == "Final Output"
    # Verify handler received tool call and tool result events
    assert handler.on_event.call_count == 4  # Start, ToolCall, ToolResult, Finish


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_process_stream_candidates_and_max_rounds(mock_client_class):
    from collections.abc import AsyncGenerator
    from typing import Any

    mock_client = mock_client_class.return_value

    class MockChunk:
        def __init__(
            self,
            text: str | None = None,
            function_calls: list[Any] | None = None,
            candidates: list[Any] | None = None,
        ) -> None:
            self.text = text
            self.function_calls = function_calls
            self.candidates = candidates

    # Yields a tool call to keep executing tool loops
    mock_fn_call = MagicMock()
    mock_fn_call.name = "my_tool"
    mock_fn_call.args = {}

    async def mock_infinite_stream() -> AsyncGenerator[MockChunk, None]:
        # Yield candidate content to cover parts is None and parts extend branches!
        mock_cand_1 = MagicMock()
        mock_cand_1.content = types.Content(role="model", parts=None)

        mock_cand_2 = MagicMock()
        mock_cand_2.content = types.Content(role="model", parts=[types.Part.from_text(text="part2")])

        yield MockChunk(function_calls=[mock_fn_call], candidates=[mock_cand_1])
        yield MockChunk(function_calls=[], candidates=[mock_cand_2])

    async def get_infinite_stream(*args, **kwargs):
        return mock_infinite_stream()

    # Let's set side effect to return infinite stream for each round
    mock_client.aio.models.generate_content_stream.side_effect = get_infinite_stream

    def my_tool() -> str:
        return "result"

    agent = BaseAgent(name="MaxRoundAgent", system_prompt="Prompt", tools=[my_tool])
    agent.max_tool_rounds = 2  # Keep it small to trigger limit fast

    chunks = []
    async for chunk in agent.process_stream("query", []):
        chunks.append(chunk)

    assert any("Reached maximum tool execution rounds" in c for c in chunks)


@pytest.mark.asyncio
async def test_base_agent_execute_tool_edge_cases():
    agent = BaseAgent(name="EdgeAgent", system_prompt="Prompt")

    # 1. Unknown tool call without a name
    mock_call_no_name = MagicMock()
    mock_call_no_name.name = None
    res = await agent._execute_tool(mock_call_no_name)
    assert res == "Error: Unknown tool call without a name."

    # 2. Tool throws exception
    def broken_tool() -> str:
        raise ValueError("Tool failed")

    agent_broken = BaseAgent(name="BrokenAgent", system_prompt="Prompt", tools=[broken_tool])
    mock_call = MagicMock()
    mock_call.name = "broken_tool"
    mock_call.args = {}

    res2 = await agent_broken._execute_tool(mock_call)
    assert "Error executing tool 'broken_tool': Tool failed" in res2


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_is_async_callable_types(mock_client):
    """Test _is_async_callable resolves complex callables (partial, custom classes, wraps)."""
    from functools import partial, wraps

    from multi_agent_orchestrator.core.agent import _is_async_callable

    # 1. Custom callable class
    class CustomCallableClass:
        async def __call__(self, arg: str) -> str:
            return f"Async {arg}"

    callable_instance = CustomCallableClass()
    assert _is_async_callable(callable_instance) is True

    # 2. Bad callable that raises AttributeError on __call__ access
    class BadCallable:
        @property
        def __call__(self) -> Any:
            raise AttributeError("Access denied")

    assert _is_async_callable(BadCallable()) is False

    # 3. Synchronous decorator wrapping async fn
    async def async_fn(val: int) -> str:
        return str(val)

    def sync_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    wrapped_fn = sync_decorator(async_fn)
    assert _is_async_callable(wrapped_fn) is True

    # 4. functools.partial wrapping a custom callable class instance (not a coroutine directly)
    partial_callable = partial(callable_instance, arg="hello")
    assert _is_async_callable(partial_callable) is True

    # 5. Standard sync function
    def sync_fn() -> str:
        return "Sync"

    assert _is_async_callable(sync_fn) is False


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_process_stream_timeout(mock_client_class):
    """Verify that process_stream() raises AgentError on TimeoutError."""
    agent = BaseAgent(name="TimeoutStreamAgent", system_prompt="Prompt", timeout=0.01)

    async def slow_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async def inner_gen() -> AsyncGenerator[Any, None]:
            import asyncio

            await asyncio.sleep(0.1)
            yield MagicMock()

        return inner_gen()

    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content_stream = slow_stream

    with pytest.raises(AgentError, match="Streaming timed out after"):
        async for _ in agent.process_stream("query", []):
            pass


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_callable_system_prompt(mock_client_class):
    """Verify system_prompt accepts and evaluates callables dynamically."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "Mocked Response"
    mock_response.function_calls = None
    mock_client.aio.models.generate_content.return_value = mock_response

    # Test query-arg callable
    def prompt_fn(q: str) -> str:
        return f"Prompt for: {q}"

    agent = BaseAgent(name="CallableAgent", system_prompt=prompt_fn)
    await agent.process("hello world", [])

    _, kwargs = mock_client.aio.models.generate_content.call_args
    assert kwargs["config"].system_instruction == "Prompt for: hello world"

    # Test zero-arg callable
    def prompt_zero() -> str:
        return "Zero prompt"

    agent_zero = BaseAgent(name="ZeroCallableAgent", system_prompt=prompt_zero)
    await agent_zero.process("hello", [])

    _, kwargs_zero = mock_client.aio.models.generate_content.call_args
    assert kwargs_zero["config"].system_instruction == "Zero prompt"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_custom_executor(mock_client_class):
    """Verify that synchronous tools are executed inside the configured custom executor."""
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def my_sync_tool(x: int) -> int:
        return x + 1

    agent = BaseAgent(name="ExecAgent", system_prompt="Prompt", tools=[my_sync_tool], executor=executor)

    mock_fn_call = MagicMock()
    mock_fn_call.name = "my_sync_tool"
    mock_fn_call.args = {"x": 5}

    # Execute it
    res = await agent._execute_tool(mock_fn_call)
    assert res == "6"

    executor.shutdown()


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_retry_jitter(mock_client_class):
    """Verify that exponential retry calculation adds fractional randomized jitter."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_client.aio.models.generate_content.side_effect = Exception("429 Too Many Requests")

    agent = BaseAgent(name="JitterAgent", system_prompt="Prompt", max_retries=2)

    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        pytest.raises(AgentError, match="All 2 retries exhausted"),
    ):
        await agent.process("query", [])

    # The sleep call should have fractional jitter (2**0 + jitter and 2**1 + jitter)
    assert mock_sleep.call_count == 2
    sleep_arg_1 = mock_sleep.call_args_list[0][0][0]
    sleep_arg_2 = mock_sleep.call_args_list[1][0][0]
    assert 1.1 <= sleep_arg_1 <= 2.0
    assert 2.1 <= sleep_arg_2 <= 3.0


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_process_stream_callable_system_prompt(mock_client_class):
    """Verify that process_stream evaluates dynamic callable system prompts correctly."""
    from collections.abc import AsyncGenerator

    mock_client = mock_client_class.return_value

    class MockChunk:
        def __init__(self, text: str) -> None:
            self.text = text
            self.function_calls = None
            self.candidates = None

    async def mock_stream_generator() -> AsyncGenerator[MockChunk, None]:
        yield MockChunk(text="Stream ")
        yield MockChunk(text="Result")

    mock_client.aio.models.generate_content_stream = AsyncMock(
        side_effect=lambda *args, **kwargs: mock_stream_generator()
    )

    def prompt_stream_fn(q: str) -> str:
        return f"Prompt for streaming: {q}"

    agent = BaseAgent(name="StreamCallableAgent", system_prompt=prompt_stream_fn)

    chunks = []
    async for chunk in agent.process_stream("stream query", []):
        chunks.append(chunk)

    assert chunks == ["Stream ", "Result"]
    _, kwargs = mock_client.aio.models.generate_content_stream.call_args
    assert kwargs["config"].system_instruction == "Prompt for streaming: stream query"

    # Test zero-arg callable in streaming
    def prompt_stream_zero() -> str:
        return "Zero stream prompt"

    agent_zero = BaseAgent(name="StreamZeroAgent", system_prompt=prompt_stream_zero)
    chunks_zero = []
    async for chunk in agent_zero.process_stream("query", []):
        chunks_zero.append(chunk)
    assert chunks_zero == ["Stream ", "Result"]
    _, kwargs_zero = mock_client.aio.models.generate_content_stream.call_args
    assert kwargs_zero["config"].system_instruction == "Zero stream prompt"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_async_system_prompt(mock_client_class):
    """Verify async callable system prompts are awaited correctly."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "Mocked"
    mock_response.function_calls = None
    mock_client.aio.models.generate_content.return_value = mock_response

    # Async prompt taking query arg
    async def async_prompt(q: str) -> str:
        return f"Async prompt for: {q}"

    agent = BaseAgent(name="AsyncPromptAgent", system_prompt=async_prompt)
    await agent.process("hello", [])

    _, kwargs = mock_client.aio.models.generate_content.call_args
    assert kwargs["config"].system_instruction == "Async prompt for: hello"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_async_system_prompt_zero_arg(mock_client_class):
    """Verify async callable system prompts with zero args work correctly."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "Mocked"
    mock_response.function_calls = None
    mock_client.aio.models.generate_content.return_value = mock_response

    async def async_prompt_zero() -> str:
        return "Async zero prompt"

    agent = BaseAgent(name="AsyncZeroAgent", system_prompt=async_prompt_zero)
    await agent.process("hello", [])

    _, kwargs = mock_client.aio.models.generate_content.call_args
    assert kwargs["config"].system_instruction == "Async zero prompt"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_lifecycle_hooks(mock_client_class):
    """Verify pre_process and post_process hooks are called during process()."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "Raw response"
    mock_response.function_calls = None
    mock_client.aio.models.generate_content.return_value = mock_response

    class HookedAgent(BaseAgent):
        async def pre_process(self, query, history):
            return f"PREPROCESSED: {query}"

        async def post_process(self, response):
            return f"POSTPROCESSED: {response}"

    agent = HookedAgent(name="HookedAgent", system_prompt="Prompt")
    response = await agent.process("original query", [])

    assert response == "POSTPROCESSED: Raw response"
    # Verify pre_process modified the query sent to model
    _, kwargs = mock_client.aio.models.generate_content.call_args
    contents = kwargs["contents"]
    assert contents[-1].parts[0].text == "PREPROCESSED: original query"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_lifecycle_hooks_streaming(mock_client_class):
    """Verify pre_process is called during process_stream()."""
    from collections.abc import AsyncGenerator

    mock_client = mock_client_class.return_value

    class MockChunk:
        def __init__(self, text: str) -> None:
            self.text = text
            self.function_calls = None
            self.candidates = None

    async def mock_stream_generator() -> AsyncGenerator[MockChunk, None]:
        yield MockChunk(text="Streamed")

    mock_client.aio.models.generate_content_stream = AsyncMock(
        side_effect=lambda *args, **kwargs: mock_stream_generator()
    )

    class HookedStreamAgent(BaseAgent):
        async def pre_process(self, query, history):
            return f"STREAM_PRE: {query}"

    agent = HookedStreamAgent(name="HookedStreamAgent", system_prompt="Prompt")
    chunks = []
    async for chunk in agent.process_stream("original", []):
        chunks.append(chunk)

    assert chunks == ["Streamed"]
    # Verify pre_process modified the query
    _, kwargs = mock_client.aio.models.generate_content_stream.call_args
    contents = kwargs["contents"]
    assert contents[-1].parts[0].text == "STREAM_PRE: original"


@pytest.mark.asyncio
async def test_resolve_prompt_utility():
    """Verify _resolve_prompt handles all prompt types correctly."""
    from multi_agent_orchestrator.core.agent import _resolve_prompt

    # 1. Static string
    assert await _resolve_prompt("Static", "query") == "Static"

    # 2. Sync callable with query arg
    def sync_fn(q: str) -> str:
        return f"Sync: {q}"

    assert await _resolve_prompt(sync_fn, "test") == "Sync: test"

    # 3. Sync callable with zero args
    def sync_zero() -> str:
        return "Sync zero"

    assert await _resolve_prompt(sync_zero, "test") == "Sync zero"

    # 4. Async callable with query arg
    async def async_fn(q: str) -> str:
        return f"Async: {q}"

    assert await _resolve_prompt(async_fn, "test") == "Async: test"

    # 5. Async callable with zero args
    async def async_zero() -> str:
        return "Async zero"

    assert await _resolve_prompt(async_zero, "test") == "Async zero"


def test_base_agent_invalid_response_schema():
    """Verify that BaseAgent raises AgentError for invalid response_schema type."""
    from multi_agent_orchestrator.core.agent import AgentError, BaseAgent

    with pytest.raises(AgentError, match="response_schema must be a dict or type"):
        BaseAgent(name="InvalidSchemaAgent", system_prompt="Prompt", response_schema="not-a-dict-or-type")


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.agent.genai.Client")
async def test_base_agent_post_process_streaming(mock_client_class):
    """Verify post_process is called on the final collected text during streaming."""
    from collections.abc import AsyncGenerator

    from multi_agent_orchestrator.core.agent import BaseAgent
    from multi_agent_orchestrator.core.events import AgentFinishEvent

    mock_client = mock_client_class.return_value

    class MockChunk:
        def __init__(self, text: str) -> None:
            self.text = text
            self.function_calls = None
            self.candidates = None

    async def mock_stream_generator() -> AsyncGenerator[MockChunk, None]:
        yield MockChunk(text="Streamed chunk")

    mock_client.aio.models.generate_content_stream = AsyncMock(
        side_effect=lambda *args, **kwargs: mock_stream_generator()
    )

    class CustomPostProcessAgent(BaseAgent):
        async def post_process(self, response: str) -> str:
            return f"PROCESSED: {response}"

    agent = CustomPostProcessAgent(name="CustomAgent", system_prompt="Prompt")

    mock_handler = MagicMock()
    mock_handler.on_event = AsyncMock()

    chunks = []
    async for chunk in agent.process_stream("query", [], event_handler=mock_handler):
        chunks.append(chunk)

    # In process_stream, the yielded chunks are raw generated chunks,
    # but the finish event should carry the post-processed collected text.
    assert chunks == ["Streamed chunk"]

    # Verify AgentFinishEvent carries the post-processed value
    assert mock_handler.on_event.call_count == 2
    event = mock_handler.on_event.call_args_list[1][0][0]
    assert isinstance(event, AgentFinishEvent)
    assert event.response == "PROCESSED: Streamed chunk"
