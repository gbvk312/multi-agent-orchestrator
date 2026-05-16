"""Centralized configuration for the Multi-Agent Orchestrator."""

import os

from pydantic import BaseModel, Field


class OrchestratorConfig(BaseModel):
    """Unified configuration loaded from environment variables or explicit values.

    Usage:
        config = OrchestratorConfig.from_env()
        config = OrchestratorConfig(gemini_api_key="...", max_history=100)

    Note: Call ``dotenv.load_dotenv()`` at your application entry-point
    before using ``from_env()`` if you need ``.env`` file support.
    """

    gemini_api_key: str = Field(default="")
    default_model: str = Field(default="gemini-2.5-flash")
    max_tool_rounds: int = Field(default=5, ge=1)
    max_retries: int = Field(default=3, ge=1)
    max_history: int = Field(default=50, ge=1)
    agent_timeout: float = Field(default=120.0, gt=0)
    temperature: float = Field(default=0.2, ge=0, le=2)
    routing_temperature: float = Field(default=0.0, ge=0, le=2)

    @classmethod
    def from_env(cls, **overrides) -> "OrchestratorConfig":
        """Create config from environment variables with optional overrides."""
        env_map = {
            "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
            "default_model": os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"),
            "max_tool_rounds": int(os.getenv("MAX_TOOL_ROUNDS", "5")),
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "max_history": int(os.getenv("MAX_HISTORY", "50")),
            "agent_timeout": float(os.getenv("AGENT_TIMEOUT", "120.0")),
            "temperature": float(os.getenv("TEMPERATURE", "0.2")),
            "routing_temperature": float(os.getenv("ROUTING_TEMPERATURE", "0.0")),
        }
        env_map.update(overrides)
        return cls(**env_map)
