from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from multi_agent_orchestrator import (
    EventHandler,
    OrchestratorErrorEvent,
    OrchestratorFinishEvent,
    OrchestratorHandoffEvent,
    OrchestratorRouteEvent,
    OrchestratorStartEvent,
)
from multi_agent_orchestrator.core.agent import BaseAgent
from multi_agent_orchestrator.core.memory import MemoryManager
from multi_agent_orchestrator.core.orchestrator import Orchestrator, OrchestratorError


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
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "AgentB"
    mock_client.aio.models.generate_content.return_value = mock_response

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
    mock_client.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_route_request_single_agent_skips_llm(mock_client_class):
    """With only one agent, routing should skip the LLM call entirely."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()

    orchestrator = Orchestrator()
    agent = MagicMock(spec=BaseAgent)
    agent.name = "OnlyAgent"
    agent.system_prompt = "Prompt"
    orchestrator.register_agent(agent)

    selected = await orchestrator._route_request("Any query")
    assert selected == "OnlyAgent"

    # LLM should NOT be called
    mock_client.aio.models.generate_content.assert_not_called()


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_route_request_fallback(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "UnknownAgent"  # Not registered
    mock_client.aio.models.generate_content.return_value = mock_response

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
async def test_route_request_no_agents_raises(mock_client_class):
    """Routing with no agents should raise OrchestratorError."""
    orchestrator = Orchestrator()
    with pytest.raises(OrchestratorError, match="No agents registered"):
        await orchestrator._route_request("Query")


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_process_request(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()

    # Mock routing response
    mock_route_response = MagicMock()
    mock_route_response.text = "AgentA"
    mock_client.aio.models.generate_content.return_value = mock_route_response

    orchestrator = Orchestrator()

    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.name = "AgentA"
    mock_agent.system_prompt = "Test prompt"
    mock_agent.process = AsyncMock(return_value="Agent response")
    orchestrator.register_agent(mock_agent)

    response = await orchestrator.process_request("session_1", "User query")

    assert response == "Agent response"
    # Agent receives empty history (no double-query bug)
    mock_agent.process.assert_called_once_with("User query", [], session_id="session_1", event_handler=None)

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

    # Verify fan_out saves to memory
    history = await orchestrator.memory.get_history("session_1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Multi query"}
    assert "[AgentA]: Response A" in history[1]["content"]
    assert "[AgentB]: Response B" in history[1]["content"]


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_fan_out_unknown_agent_raises(mock_client_class):
    """fan_out should raise OrchestratorError for unknown agent names."""
    orchestrator = Orchestrator()

    agent = MagicMock(spec=BaseAgent)
    agent.name = "AgentA"
    orchestrator.register_agent(agent)

    with pytest.raises(OrchestratorError, match="Unknown agents"):
        await orchestrator.fan_out("s1", "query", agent_names=["NonExistent"])


@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
def test_orchestrator_repr(mock_client):
    orchestrator = Orchestrator()
    assert "Orchestrator" in repr(orchestrator)
    assert "gemini-2.5-flash" in repr(orchestrator)


@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
def test_orchestrator_missing_api_key(mock_client, monkeypatch):
    """Orchestrator should fail fast if no API key is available."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(OrchestratorError, match="GEMINI_API_KEY"):
        Orchestrator()


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_route_request_llm_exception_fallback(mock_client_class):
    """Verify routing falls back to the first agent if the LLM raises an exception."""
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    mock_client.aio.models.generate_content.side_effect = Exception("API Error")

    orchestrator = Orchestrator()
    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.system_prompt = "Prompt A"

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.system_prompt = "Prompt B"

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    selected = await orchestrator._route_request("Query")
    assert selected == "AgentA"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_process_request_agent_error(mock_client_class):
    """Verify process_request handles AgentError and returns a formatted string."""
    from multi_agent_orchestrator.core.agent import AgentError

    orchestrator = Orchestrator()

    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.name = "AgentA"
    mock_agent.system_prompt = "Test prompt"
    mock_agent.process = AsyncMock(side_effect=AgentError("Agent failed hard"))
    orchestrator.register_agent(mock_agent)

    # Route request mocked to return AgentA
    with patch.object(orchestrator, "_route_request", return_value="AgentA"):
        response = await orchestrator.process_request("session_1", "User query")

    assert "Error from AgentA: Agent failed hard" in response


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_fan_out_agent_error(mock_client_class):
    """Verify fan_out catches AgentError and includes it in results."""
    from multi_agent_orchestrator.core.agent import AgentError

    orchestrator = Orchestrator()

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process = AsyncMock(return_value="Response A")

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.process = AsyncMock(side_effect=AgentError("AgentB blew up"))

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    results = await orchestrator.fan_out("session_1", "Multi query")

    assert results["AgentA"] == "Response A"
    assert "Error from AgentB: AgentB blew up" in results["AgentB"]


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_handoff_in_processing(mock_client_class):
    """Verify that agent handoff exceptions are caught and correctly routed in process_request."""
    from multi_agent_orchestrator.core.agent import AgentHandoff

    orchestrator = Orchestrator()

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process = AsyncMock(side_effect=AgentHandoff("AgentB", "transfer payload"))

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.process = AsyncMock(return_value="Final Answer from B")

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    with patch.object(orchestrator, "_route_request", return_value="AgentA"):
        response = await orchestrator.process_request("s1", "Initial request")

    assert response == "Final Answer from B"
    agent_a.process.assert_called_once()
    # Agent B should be called with the handoff message
    agent_b.process.assert_called_once_with("transfer payload", ANY, session_id="s1", event_handler=None)


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_human_approval_pausing(mock_client_class):
    """Verify that human approval exceptions pause processing in process_request."""
    from multi_agent_orchestrator.core.agent import HumanApprovalRequired

    orchestrator = Orchestrator()

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process = AsyncMock(side_effect=HumanApprovalRequired("tool_x", {"foo": "bar"}, "please approve"))

    orchestrator.register_agent(agent_a)

    with patch.object(orchestrator, "_route_request", return_value="AgentA"):
        response = await orchestrator.process_request("s1", "Run tool")

    assert "Execution paused. Human approval required for tool 'tool_x'" in response
    assert "please approve" in response


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_sequential_chain(mock_client_class):
    """Verify orchestrator chain executes agents in a pipeline."""
    orchestrator = Orchestrator()

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process = AsyncMock(return_value="Output A")

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.process = AsyncMock(return_value="Output B")

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    response = await orchestrator.chain("s1", "Start", ["AgentA", "AgentB"])
    assert response == "Output B"
    agent_a.process.assert_called_once()
    agent_b.process.assert_called_once()


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_process_request_stream(mock_client_class):
    """Verify process_request_stream handles streaming responses and dynamic handoffs."""
    from multi_agent_orchestrator.core.agent import AgentHandoff

    orchestrator = Orchestrator()

    async def mock_stream_a(*args, **kwargs):
        raise AgentHandoff("AgentB", "transfer stream")
        # To make it an async generator
        yield "never"

    async def mock_stream_b(*args, **kwargs):
        yield "Chunk 1"
        yield " Chunk 2"

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process_stream = mock_stream_a

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.process_stream = mock_stream_b

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    with patch.object(orchestrator, "_route_request", return_value="AgentA"):
        chunks = []
        async for chunk in orchestrator.process_request_stream("s1", "Stream me"):
            chunks.append(chunk)

    assert chunks == ["Chunk 1", " Chunk 2"]


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_chain_edge_cases(mock_client_class):
    orchestrator = Orchestrator()
    # Empty sequence
    with pytest.raises(OrchestratorError, match="Sequence of agents cannot be empty"):
        await orchestrator.chain("s1", "Query", [])

    # Unknown agent in sequence
    with pytest.raises(OrchestratorError, match="Unknown agent in sequence"):
        await orchestrator.chain("s1", "Query", ["Ghost"])

    # Agent throws exception inside chain
    agent = MagicMock(spec=BaseAgent)
    agent.name = "AgentA"
    agent.process = AsyncMock(side_effect=Exception("Hard crash"))
    orchestrator.register_agent(agent)

    response = await orchestrator.chain("s1", "Query", ["AgentA"])
    assert "Chain failed at AgentA: Hard crash" in response


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_process_request_agent_not_found(mock_client_class):
    orchestrator = Orchestrator()
    # Mock route to returns agent that doesn't exist
    with patch.object(orchestrator, "_route_request", return_value="Ghost"):
        response = await orchestrator.process_request("s1", "Query")
    assert "Error: Agent 'Ghost' not found" in response

    # Also for stream
    with patch.object(orchestrator, "_route_request", return_value="Ghost"):
        chunks = []
        async for chunk in orchestrator.process_request_stream("s1", "Query"):
            chunks.append(chunk)
    assert "Error: Agent 'Ghost' not found" in "".join(chunks)


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_process_request_stream_exceptions(mock_client_class):
    from multi_agent_orchestrator.core.agent import AgentError, HumanApprovalRequired

    orchestrator = Orchestrator()

    # 1. Approval Required
    async def mock_stream_approval(*args, **kwargs):
        raise HumanApprovalRequired("some_tool", {}, "paused stream")
        yield "never"

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process_stream = mock_stream_approval
    orchestrator.register_agent(agent_a)

    with patch.object(orchestrator, "_route_request", return_value="AgentA"):
        chunks = []
        async for chunk in orchestrator.process_request_stream("s1", "Query"):
            chunks.append(chunk)
    assert "Execution paused. Human approval required for tool 'some_tool'" in "".join(chunks)

    # 2. Agent Error
    async def mock_stream_error(*args, **kwargs):
        raise AgentError("Agent is down")
        yield "never"

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.process_stream = mock_stream_error
    orchestrator.register_agent(agent_b)

    with patch.object(orchestrator, "_route_request", return_value="AgentB"):
        chunks = []
        async for chunk in orchestrator.process_request_stream("s1", "Query"):
            chunks.append(chunk)
    assert "Error from AgentB: Agent is down" in "".join(chunks)


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_custom_fallback_agent(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.aio.models.generate_content = AsyncMock()
    # Mock LLM to throw an exception to trigger the fallback logic
    mock_client.aio.models.generate_content.side_effect = Exception("Routing failed")

    orchestrator = Orchestrator(default_fallback_agent="AgentB")
    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.system_prompt = "Prompt A"

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.system_prompt = "Prompt B"

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    selected = await orchestrator._route_request("Query")
    # Should fallback to our custom default fallback agent
    assert selected == "AgentB"

    # Test unregistered fallback agent falls back to first registered agent
    orchestrator.default_fallback_agent = "GhostAgent"
    selected2 = await orchestrator._route_request("Query")
    assert selected2 == "AgentA"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_fan_out_concurrency_semaphore(mock_client_class):
    orchestrator = Orchestrator()

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process = AsyncMock(return_value="Response A")

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.process = AsyncMock(return_value="Response B")

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    # Test parallel fan-out with max_concurrency=1
    results = await orchestrator.fan_out("s1", "Query", max_concurrency=1)
    assert results["AgentA"] == "Response A"
    assert results["AgentB"] == "Response B"


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_events_normal_flow(mock_client_class):
    """Verify standard orchestrator events are dispatched during normal process_request."""
    orchestrator = Orchestrator()
    mock_handler = AsyncMock(spec=EventHandler)
    orchestrator.event_handler = mock_handler

    agent = MagicMock(spec=BaseAgent)
    agent.name = "AgentA"
    agent.process = AsyncMock(return_value="Ok response")
    orchestrator.register_agent(agent)

    with patch.object(orchestrator, "_route_request", return_value="AgentA"):
        await orchestrator.process_request("session_event_1", "Hello there")

    # Assert mock_handler.on_event was called with Start, Route, Finish events
    assert mock_handler.on_event.call_count == 3
    events = [call.args[0] for call in mock_handler.on_event.call_args_list]

    assert any(
        isinstance(e, OrchestratorStartEvent) and e.session_id == "session_event_1" and e.query == "Hello there"
        for e in events
    )
    assert any(
        isinstance(e, OrchestratorRouteEvent) and e.session_id == "session_event_1" and e.agent_name == "AgentA"
        for e in events
    )
    assert any(
        isinstance(e, OrchestratorFinishEvent) and e.session_id == "session_event_1" and e.response == "Ok response"
        for e in events
    )


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_events_handoff(mock_client_class):
    """Verify OrchestratorHandoffEvent is dispatched during agent handoff."""
    from multi_agent_orchestrator.core.agent import AgentHandoff

    orchestrator = Orchestrator()
    mock_handler = AsyncMock(spec=EventHandler)
    orchestrator.event_handler = mock_handler

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process = AsyncMock(side_effect=AgentHandoff("AgentB", "transfer info"))

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.process = AsyncMock(return_value="Final Answer")

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    with patch.object(orchestrator, "_route_request", return_value="AgentA"):
        await orchestrator.process_request("session_event_2", "Go to B")

    events = [call.args[0] for call in mock_handler.on_event.call_args_list]
    assert any(
        isinstance(e, OrchestratorHandoffEvent)
        and e.source_agent == "AgentA"
        and e.target_agent == "AgentB"
        and e.message == "transfer info"
        for e in events
    )


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_events_error(mock_client_class):
    """Verify OrchestratorErrorEvent is dispatched during a process_request exception."""
    orchestrator = Orchestrator()
    mock_handler = AsyncMock(spec=EventHandler)
    orchestrator.event_handler = mock_handler

    agent = MagicMock(spec=BaseAgent)
    agent.name = "AgentA"
    # Cause a standard exception to trigger OrchestratorErrorEvent
    agent.process = AsyncMock(side_effect=ValueError("Unexpected backend failure"))
    orchestrator.register_agent(agent)

    with (
        patch.object(orchestrator, "_route_request", return_value="AgentA"),
        pytest.raises(ValueError, match="Unexpected backend failure"),
    ):
        await orchestrator.process_request("session_event_3", "Break things")

    events = [call.args[0] for call in mock_handler.on_event.call_args_list]
    assert any(isinstance(e, OrchestratorErrorEvent) and isinstance(e.error, ValueError) for e in events)


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_events_stream_flow(mock_client_class):
    """Verify OrchestratorStartEvent, RouteEvent, FinishEvent, and HandoffEvent work in streaming mode."""
    from multi_agent_orchestrator.core.agent import AgentHandoff

    orchestrator = Orchestrator()
    mock_handler = AsyncMock(spec=EventHandler)
    orchestrator.event_handler = mock_handler

    async def mock_stream_a(*args, **kwargs):
        raise AgentHandoff("AgentB", "handoff from stream")
        yield "never"

    async def mock_stream_b(*args, **kwargs):
        yield "Chunk X"

    agent_a = MagicMock(spec=BaseAgent)
    agent_a.name = "AgentA"
    agent_a.process_stream = mock_stream_a

    agent_b = MagicMock(spec=BaseAgent)
    agent_b.name = "AgentB"
    agent_b.process_stream = mock_stream_b

    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)

    with patch.object(orchestrator, "_route_request", return_value="AgentA"):
        chunks = []
        async for chunk in orchestrator.process_request_stream("session_event_4", "Stream handoff"):
            chunks.append(chunk)

    assert "".join(chunks) == "Chunk X"
    events = [call.args[0] for call in mock_handler.on_event.call_args_list]

    assert any(isinstance(e, OrchestratorStartEvent) for e in events)
    assert any(isinstance(e, OrchestratorRouteEvent) for e in events)
    assert any(
        isinstance(e, OrchestratorHandoffEvent)
        and e.source_agent == "AgentA"
        and e.target_agent == "AgentB"
        and e.message == "handoff from stream"
        for e in events
    )
    assert any(isinstance(e, OrchestratorFinishEvent) and e.response == "Chunk X" for e in events)


@pytest.mark.asyncio
@patch("multi_agent_orchestrator.core.orchestrator.genai.Client")
async def test_orchestrator_events_stream_error(mock_client_class):
    """Verify OrchestratorErrorEvent is dispatched during a streaming exception."""
    orchestrator = Orchestrator()
    mock_handler = AsyncMock(spec=EventHandler)
    orchestrator.event_handler = mock_handler

    async def mock_stream_fail(*args, **kwargs):
        raise RuntimeError("Streaming crashed")
        yield "never"

    agent = MagicMock(spec=BaseAgent)
    agent.name = "AgentA"
    agent.process_stream = mock_stream_fail
    orchestrator.register_agent(agent)

    with (
        patch.object(orchestrator, "_route_request", return_value="AgentA"),
        pytest.raises(RuntimeError, match="Streaming crashed"),
    ):
        async for _ in orchestrator.process_request_stream("session_event_5", "Stream crash"):
            pass

    events = [call.args[0] for call in mock_handler.on_event.call_args_list]
    assert any(isinstance(e, OrchestratorErrorEvent) and isinstance(e.error, RuntimeError) for e in events)
