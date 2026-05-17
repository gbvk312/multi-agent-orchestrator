# Multi-Agent Orchestrator (Nexus)

[![CI](https://github.com/gbvk312/multi-agent-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/gbvk312/multi-agent-orchestrator/actions/workflows/ci.yml)
[![Coverage Status](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)](#)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An asynchronous, from-scratch Python framework for building complex multi-agent workflows, powered natively by Google's Gemini.

---

## 🌟 Key Features

*   **Dynamic Intent Routing**: A smart routing supervisor that evaluates user input and dynamically delegates prompts to the most capable specialized agent.
*   **Automatic Async Tool Execution**: Agents automatically intercept, execute, and feed back function-call results in a loop (up to configurable rounds), supporting both synchronous (via thread pools) and native asynchronous tools.
*   **Sequential Chaining & Parallel Execution**: Easily pipeline multiple agents sequentially (`chain`), or fan out query processing to multiple agents concurrently (`fan_out`) with configurable concurrency throttle control.
*   **Context & Pluggable Memory**: Bounded state management passed seamlessly between agents. Easily swap in `InMemoryBackend`, `SQLiteMemoryBackend`, `RedisMemoryBackend`, or a custom persistent storage.
*   **High Performance & Lazy-Loaded Imports**: Clean module isolation. Deferring the loading of heavy optional dependencies like `redis`, `aiosqlite`, and `fastapi` until they are explicitly needed.
*   **Robust Error Handling & Backoff**: Standardized API key fail-fast guards, per-operation timeouts, and automatic exponential backoff retries for rate limits (429) or transient server errors (500).
*   **100% Test Coverage**: Fully verified package suite checking every async stream, event state transition, and tool executor path with zero regressions.

---

## 📐 Architecture Overview

```
                        +----------------------------+
                        |        User Request        |
                        +--------------+-------------+
                                       |
                                       v
                    +------------------+------------------+
                    |  Orchestrator.process_request(...)  |
                    +------------------+------------------+
                                       |
                                       v
                    +------------------+------------------+
                    |      Routing Supervisor (LLM)       |
                    +------------------+------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
     +-------------+-------------+           +-------------+-------------+
     |   Research Agent (Base)   |           |    Coding Agent (Base)    |
     +-------------+-------------+           +-------------+-------------+
                   |                                       |
                   | (Optional tool execution loop)        | (Optional tool loop)
                   v                                       v
     +-------------+-------------+           +-------------+-------------+
     |      Google Gemini API     |           |      Google Gemini API     |
     +-------------+-------------+           +-------------+-------------+
                   |                                       |
                   v                                       v
     +-------------+-------------+           +-------------+-------------+
     |     Memory/State Backend  |           |     Memory/State Backend  |
     |  (In-Memory/SQLite/Redis) |           |  (In-Memory/SQLite/Redis) |
     +---------------------------+           +---------------------------+
```

---

## 📦 Installation

Get started quickly using [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`:

```bash
git clone https://github.com/gbvk312/multi-agent-orchestrator.git
cd multi-agent-orchestrator

# Option A: Using uv (Fastest)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,api,storage]"

# Option B: Using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,storage]"
```

---

## 🚀 Quick Start

1. Copy the environment variables template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your **`GEMINI_API_KEY`**.
3. Run the basic orchestrated workflow:
   ```bash
   python examples/basic_workflow.py
   ```

---

## 💡 Usage Examples

### 1. Simple Custom Agent & Orchestrator Setup

```python
import asyncio
from multi_agent_orchestrator import BaseAgent, Orchestrator

# Define a custom tool for the agent
def fetch_weather(location: str) -> str:
    """Fetch the current weather for a location."""
    return f"The weather in {location} is 22°C and sunny."

async def main():
    # 1. Initialize the Orchestrator
    orchestrator = Orchestrator()

    # 2. Setup the Weather Agent
    weather_agent = BaseAgent(
        name="WeatherBot",
        system_prompt="You are a friendly weather assistant. Use your tools to check weather conditions.",
        tools=[fetch_weather]
    )

    # 3. Register the agent
    orchestrator.register_agent(weather_agent)

    # 4. Route and process query asynchronously
    response = await orchestrator.process_request(
        session_id="session_user_99",
        query="Can you check what the weather is like in San Francisco?"
    )
    
    print(f"🤖 Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Multi-Agent Sequential Chaining

```python
# Sequentially pipeline queries across multiple agents
final_result = await orchestrator.chain(
    session_id="pipeline_1",
    query="Draft a specification for an email-sender service",
    sequence=["Research_Agent", "Coding_Agent"]
)
```

### 3. Parallel Fan-Out Execution

```python
# Query multiple agents concurrently with a concurrency throttle
results = await orchestrator.fan_out(
    session_id="parallel_1",
    query="Review this code block from multiple perspectives",
    agent_names=["Security_Agent", "Coding_Agent"],
    max_concurrency=2
)
```

---

## ⚡ FastAPI Production Integration

The framework is highly optimized for deployment behind modern asynchronous APIs like **FastAPI**. By passing standard request models, you can run orchestrator workflows or execute parallel fan-outs instantly.

A complete API server example is included in the project under [fastapi_wrapper.py](file:///Users/gbvk/Downloads/repo/github/personal/multi-agent-orchestrator/examples/fastapi_wrapper.py).

### How to Run:
```bash
python examples/fastapi_wrapper.py
```

### Exposing Endpoints:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from multi_agent_orchestrator import Orchestrator

app = FastAPI(title="Multi-Agent Orchestrator API")
orchestrator = Orchestrator()

class ChatRequest(BaseModel):
    session_id: str
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await orchestrator.process_request(
            session_id=request.session_id,
            query=request.query
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 💾 Advanced Storage & Pluggable Memory

Swap backends seamlessly by passing them directly into the `MemoryManager`.

### 1. Redis Memory Backend (High Scale)
Pass a custom pre-configured client pool to prevent early socket teardowns in serverless environments:
```python
import redis.asyncio as redis
from multi_agent_orchestrator import RedisMemoryBackend, MemoryManager, Orchestrator

custom_redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
backend = RedisMemoryBackend(redis_client=custom_redis_client)
memory = MemoryManager(backend=backend)

orchestrator = Orchestrator(memory_manager=memory)
```

### 2. SQLite Memory Backend (Local Persistence)
```python
from multi_agent_orchestrator import SQLiteMemoryBackend, MemoryManager, Orchestrator

sqlite_backend = SQLiteMemoryBackend(db_path="orchestrator_sessions.db")
memory = MemoryManager(backend=sqlite_backend)

orchestrator = Orchestrator(memory_manager=memory)
```

---

## 🛡️ Observability & Event System

Track and trace every transition in the orchestrator workflow by subclassing `EventHandler`:

```python
from multi_agent_orchestrator import (
    EventHandler,
    OrchestratorStartEvent,
    AgentStartEvent,
    OrchestratorFinishEvent
)

class MyTraceHandler(EventHandler):
    async def on_orchestrator_start(self, event: OrchestratorStartEvent) -> None:
        print(f"🚀 Session {event.session_id} process started.")

    async def on_agent_start(self, event: AgentStartEvent) -> None:
        print(f"🤖 Activated specialized agent: {event.agent_name}")

    async def on_orchestrator_finish(self, event: OrchestratorFinishEvent) -> None:
        print(f"✅ Finished! Response size: {len(event.response)} chars")
```

### Lifecycle Event Types Available:
*   **Orchestrator Transitions**: `OrchestratorStartEvent`, `OrchestratorRouteEvent`, `OrchestratorHandoffEvent`, `OrchestratorFinishEvent`, `OrchestratorErrorEvent`
*   **Agent Transitions**: `AgentStartEvent`, `AgentFinishEvent`
*   **Tool Executions**: `ToolCallEvent`, `ToolResultEvent`

---

## ⚡ Lazy-Loading Module Architecture

To ensure minimum startup overhead and resource isolation, Nexus utilizes lazy imports for optional components. 

When you do:
```python
from multi_agent_orchestrator import Orchestrator, BaseAgent
```
Only the absolute minimum dependencies are initialized. Optional libraries (such as `redis` for `RedisMemoryBackend`, `aiosqlite` for `SQLiteMemoryBackend`, or `fastapi` in wrappers) are dynamically resolved and loaded **only when the class is actually accessed or constructed**. This prevents unnecessary module imports and keeps microservices extremely lightweight.

---

## 📖 Local Documentation

Detailed API documentation is built using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

To run the local documentation server:
```bash
uv run mkdocs serve
```
Then open `http://127.0.0.1:8000` in your web browser.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by [gbvk312](https://github.com/gbvk312).*
