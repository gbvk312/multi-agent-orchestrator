import os
import asyncio

from multi_agent_orchestrator.core import BaseAgent, Orchestrator


# Define a sample tool for the Coding Agent
def write_python_function(function_name: str, purpose: str) -> str:
    """Generates a skeleton python function based on a purpose."""
    return f"def {function_name}():\n    # TODO: {purpose}\n    pass\n"


async def main():
    # Load environment variables from .env
    from dotenv import load_dotenv

    load_dotenv()

    # Make sure GEMINI_API_KEY is set in your environment
    if not os.getenv("GEMINI_API_KEY"):
        print("Please set GEMINI_API_KEY environment variable.")
        return

    # 1. Initialize Agents
    research_agent = BaseAgent(
        name="Research_Agent",
        system_prompt="You are a researcher. Your job is to gather and summarize information concisely.",
    )

    coding_agent = BaseAgent(
        name="Coding_Agent",
        system_prompt="You are an expert Python developer. Provide high-quality code snippets.",
        tools=[write_python_function],
    )

    # 2. Initialize Orchestrator
    orchestrator = Orchestrator()
    orchestrator.register_agent(research_agent)
    orchestrator.register_agent(coding_agent)

    session_id = "demo_session_1"

    print("Multi-Agent Orchestrator initialized. Two agents available: Research_Agent, Coding_Agent.")
    print("--------------------------------------------------")

    # 3. Simulate multi-turn conversation
    queries = ["What is the capital of France?", "Can you write a python script to print the name of that city?"]

    for query in queries:
        print(f"\nUser: {query}")
        response = await orchestrator.process_request(session_id=session_id, query=query)
        print(f"Response:\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
