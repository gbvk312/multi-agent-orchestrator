import asyncio
import logging
import os

from multi_agent_orchestrator import (
    BaseAgent,
    MemoryManager,
    Orchestrator,
    SQLiteMemoryBackend,
)

# Set up logging to see the routing and fan-out process
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 1. Define tools that an agent can use
async def fetch_stock_price(symbol: str) -> float:
    """Mock function to fetch the current stock price of a company.

    Args:
        symbol: The stock ticker symbol (e.g. AAPL, MSFT, GOOGL)
    """
    await asyncio.sleep(0.5)  # Simulate network latency
    prices = {"AAPL": 150.0, "MSFT": 310.0, "GOOGL": 2800.0}
    return prices.get(symbol.upper(), 100.0)


# 2. Define our multi-agent workflow
async def run_complex_workflow() -> None:
    # We will use the new SQLite backend for persistent memory.
    # A temporary test database will be created.
    backend = SQLiteMemoryBackend(db_path="workflow_memory.db")
    memory_manager = MemoryManager(max_history=10, backend=backend)

    orchestrator = Orchestrator(memory_manager=memory_manager)

    # Agent 1: Financial Analyst with tools
    financial_agent = BaseAgent(
        name="FinancialAnalyst",
        system_prompt=(
            "You are a financial expert. Use the fetch_stock_price tool to answer questions about stock prices."
        ),
        tools=[fetch_stock_price],
    )

    # Agent 2: Tech News Summarizer
    tech_agent = BaseAgent(
        name="TechSummarizer",
        system_prompt="You are a tech journalist. Provide concise, 1-2 sentence summaries of tech companies.",
    )

    # Register agents
    orchestrator.register_agent(financial_agent)
    orchestrator.register_agent(tech_agent)

    session_id = "complex_session_1"

    print("--- 1. Orchestrator Routing & Tool Execution ---")
    query = "What is the current stock price of MSFT?"
    print(f"User: {query}")

    # The orchestrator will route to FinancialAnalyst, which will use the tool
    response = await orchestrator.process_request(session_id, query)
    print(f"Response: {response}\n")

    print("--- 2. Parallel Execution (Fan-out) ---")
    fan_out_query = "What are the core products of Apple?"
    print(f"User: {fan_out_query}")

    # fan_out routes the query to all registered agents simultaneously
    results = await orchestrator.fan_out(session_id, fan_out_query)
    for agent_name, result in results.items():
        print(f"[{agent_name}]: {result}")


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set the GEMINI_API_KEY environment variable.")
    else:
        asyncio.run(run_complex_workflow())
