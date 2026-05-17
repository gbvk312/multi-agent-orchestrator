# Multi-Agent Orchestrator (Nexus)

[![CI](https://github.com/gbvk312/multi-agent-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/gbvk312/multi-agent-orchestrator/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An asynchronous, from-scratch Python framework for building complex multi-agent workflows, powered by Google's Gemini.

## Features
- **Dynamic Routing**: A smart `Orchestrator` that evaluates user intent and routes prompts to the most capable agent.
- **Automatic Tool Execution**: Agents automatically execute function calls and feed results back to the model in a loop (up to 5 rounds).
- **Parallel Execution**: Use the `fan_out` method to execute multiple independent agents concurrently for complex tasks.
- **Error Handling & Retries**: Built-in exponential backoff for rate limits (429) and server errors, with configurable retry counts.
- **Context Management & Pluggable Memory**: A bounded `MemoryManager` passes context across agents seamlessly. Swap in Redis, SQLite, or custom backends via the `MemoryBackend` interface.
- **Security Enhancements**: Fail-fast mechanisms for API keys prevent silent failures.
- **Gemini Powered**: Uses the `google-genai` SDK to run `gemini-2.5-flash` natively, with full async support.
- **Production-Ready**: Adheres to high hygiene standards with CI/CD matrix testing and comprehensive code formatting checks.

## Architecture

1. **User Request** -> `await Orchestrator.process_request()`
2. **Orchestrator** -> Analyzes the intent using an async LLM router.
3. **Delegation** -> Hands off the context to the `BaseAgent` (e.g., `ResearchAgent` or `CodingAgent`).
4. **Execution** -> Agent executes tools automatically in a loop until the model produces a text response.
5. **Memory** -> Response is saved to the bounded session history.

## Installation

```bash
git clone https://github.com/gbvk312/multi-agent-orchestrator.git
cd multi-agent-orchestrator

# Using uv (Recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,api,storage]"

# Or using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,storage]"
```

## Quick Start

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
2. Add your Gemini API key to `.env`.
3. Run the basic workflow example:
   ```bash
   python examples/basic_workflow.py
   ```
4. Run the FastAPI wrapper example:
   ```bash
   python examples/fastapi_wrapper.py
   ```

## Documentation

This project uses [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for documentation.
To build and serve the documentation locally:
```bash
uv run mkdocs serve
```
Then open `http://127.0.0.1:8000` in your browser.

## Creating Custom Agents

```python
import asyncio
from multi_agent_orchestrator.core import BaseAgent, Orchestrator

# Define a custom tool
def fetch_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

# Initialize the Agent
weather_agent = BaseAgent(
    name="WeatherBot",
    system_prompt="You check the weather.",
    tools=[fetch_weather]
)

# Register with Orchestrator
orchestrator = Orchestrator()
orchestrator.register_agent(weather_agent)

# Process request (async)
async def main():
    response = await orchestrator.process_request(
        session_id="session_123",
        query="What's the weather in London?"
    )
    print(response)

asyncio.run(main())
```

## Advanced Storage and Pluggable Memory

Swap memory storage backends seamlessly to scale.

### Redis Memory Backend (with Custom Client support)
You can inject a pre-configured or shared `redis.Redis` client to safely integrate with FastAPI or serverless application pools. This ensures that the orchestrator uses your active connection pool without closing it prematurely when `.close()` is called:

```python
import redis.asyncio as redis
from multi_agent_orchestrator.core import RedisMemoryBackend

# Pre-configured shared pool client
custom_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

backend = RedisMemoryBackend(redis_client=custom_client)
```

## Observability & Event System

The framework features a built-in event tracing system. You can inherit from `EventHandler` to intercept orchestrator-level and agent-level transitions.

```python
from multi_agent_orchestrator import (
    EventHandler,
    OrchestratorStartEvent,
    OrchestratorFinishEvent,
    AgentStartEvent,
    Orchestrator
)

class MyTraceHandler(EventHandler):
    async def on_orchestrator_start(self, event: OrchestratorStartEvent) -> None:
        print(f"🚀 Processing started for session: {event.session_id}")

    async def on_agent_start(self, event: AgentStartEvent) -> None:
        print(f"🤖 Delegate agent activated: {event.agent_name}")

    async def on_orchestrator_finish(self, event: OrchestratorFinishEvent) -> None:
        print(f"✅ Finished! Response length: {len(event.response)}")

# Register handler
orchestrator = Orchestrator()
orchestrator.event_handler = MyTraceHandler()
```

### Supported Event Types
*   **Orchestrator Lifecycle**: `OrchestratorStartEvent`, `OrchestratorRouteEvent`, `OrchestratorHandoffEvent`, `OrchestratorFinishEvent`, `OrchestratorErrorEvent`
*   **Agent Lifecycle**: `AgentStartEvent`, `AgentFinishEvent`
*   **Tool Lifecycle**: `ToolCallEvent`, `ToolResultEvent`

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by [gbvk312](https://github.com/gbvk312).*

