# Multi-Agent Orchestrator (Nexus)

[![CI](https://github.com/gbvk312/multi-agent-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/gbvk312/multi-agent-orchestrator/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An asynchronous, from-scratch Python framework for building complex multi-agent workflows, powered by Google's Gemini.

## Features

- **Dynamic Routing**: A smart `Orchestrator` that evaluates user intent and routes prompts to the most capable agent.
- **Automatic Tool Execution**: Agents automatically execute function calls and feed results back to the model in a loop (up to configurable rounds), supporting both sync and async tools.
- **Sequential Chaining & Parallel Fan-Out**: Pipeline agents sequentially (`chain`) or fan out concurrently (`fan_out`) with configurable concurrency throttle.
- **Agent Handoffs & HITL**: Agents can transfer control to other agents via `AgentHandoff`, or pause for human approval via `HumanApprovalRequired`.
- **Lifecycle Hooks**: Subclass `BaseAgent` and override `pre_process` / `post_process` for custom query/response transformation.
- **Pluggable Memory**: Bounded `MemoryManager` with `InMemoryBackend`, `SQLiteMemoryBackend`, `RedisMemoryBackend`, or custom backends.
- **Observability**: Full event system via `EventHandler` for tracing agent starts, finishes, tool calls, routing, handoffs, and errors.
- **Production Safety**: Handoff loop protection, exponential backoff with jitter, configurable error propagation, and 100% test coverage enforced in CI.
- **Gemini Powered**: Uses the `google-genai` SDK to run `gemini-2.5-flash` natively with full async support.

## Architecture

1. **User Request** -> `await Orchestrator.process_request()`
2. **Orchestrator** -> Analyzes the intent using an async LLM router.
3. **Delegation** -> Hands off to the `BaseAgent` (e.g., `ResearchAgent` or `CodingAgent`).
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

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by [gbvk312](https://github.com/gbvk312).*
