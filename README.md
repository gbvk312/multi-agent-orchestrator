# Multi-Agent Orchestrator (Nexus)

An asynchronous, from-scratch Python framework for building complex multi-agent workflows, powered by Google's Gemini.

## Features
- **Dynamic Routing**: A smart `Orchestrator` that evaluates user intent and routes prompts to the most capable agent.
- **Context Management**: A centralized `MemoryManager` that passes context seamlessly during agent handoffs.
- **Gemini Powered**: Uses the new `google-genai` SDK to run `gemini-2.5-flash` natively, with support for advanced function calling.

## Architecture

1. **User Request** -> `Orchestrator.process_request()`
2. **Orchestrator** -> Analyzes the intent using an LLM router.
3. **Delegation** -> Hands off the context to the `BaseAgent` (e.g., `ResearchAgent` or `CodingAgent`).
4. **Execution** -> Agent executes tools or generates text.
5. **Memory** -> Response is saved to the session history.

## Installation

```bash
git clone https://github.com/gbvk312/multi-agent-orchestrator.git
cd multi-agent-orchestrator

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"
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

response = orchestrator.process_request(
    session_id="session_123", 
    query="What's the weather in London?"
)
print(response)
```