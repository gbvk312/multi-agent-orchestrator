import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from multi_agent_orchestrator.core import BaseAgent, Orchestrator

# Ensure .env is loaded
load_dotenv()

# Initialize orchestrator globally
orchestrator = Orchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Setup: Initialize agents on startup
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY environment variable is required.")

    # 1. Define Agents
    research_agent = BaseAgent(
        name="Research_Agent",
        system_prompt="You are a researcher. Provide concise, accurate information.",
    )

    coding_agent = BaseAgent(
        name="Coding_Agent",
        system_prompt="You are an expert developer. Provide clean, well-commented code.",
    )

    # 2. Register Agents
    orchestrator.register_agent(research_agent)
    orchestrator.register_agent(coding_agent)

    yield
    # Teardown (if needed)


app = FastAPI(
    title="Multi-Agent Orchestrator API",
    description="A REST API wrapper for the Multi-Agent Orchestrator.",
    version="1.0.0",
    lifespan=lifespan,
)


class RequestModel(BaseModel):
    session_id: str
    query: str


class ResponseModel(BaseModel):
    agent_response: str


@app.post("/chat", response_model=ResponseModel)
async def chat_endpoint(request: RequestModel) -> ResponseModel:
    try:
        response = await orchestrator.process_request(session_id=request.session_id, query=request.query)
        return ResponseModel(agent_response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/fan_out")
async def fan_out_endpoint(request: RequestModel) -> dict[str, Any]:
    """Executes the query across all available agents in parallel."""
    try:
        results = await orchestrator.fan_out(session_id=request.session_id, query=request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    # Run the server with: python examples/fastapi_wrapper.py
    uvicorn.run(app, host="127.0.0.1", port=8000)
