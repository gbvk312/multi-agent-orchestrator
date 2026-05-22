"""Centralized configuration for the Multi-Agent Orchestrator."""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OrchestratorConfig(BaseModel):
    """Unified configuration loaded from environment variables or explicit values.


    Usage:
        config = OrchestratorConfig.from_env()
        config = OrchestratorConfig(gemini_api_key="...", max_history=100)

    Note: Call ``dotenv.load_dotenv()`` at your application entry-point
    before using ``from_env()`` if you need ``.env`` file support.
    """

    model_config = ConfigDict(frozen=True)

    gemini_api_key: str = Field(default="")
    default_model: str = Field(default="gemini-2.5-flash")
    max_tool_rounds: int = Field(default=5, ge=1)
    max_retries: int = Field(default=3, ge=1)
    max_history: int = Field(default=50, ge=1)
    agent_timeout: float = Field(default=120.0, gt=0)
    temperature: float = Field(default=0.2, ge=0, le=2)
    routing_temperature: float = Field(default=0.0, ge=0, le=2)
    propagate_errors: bool = Field(default=False)
    max_handoffs: int = Field(default=5, ge=1)
    routing_system_instruction: str = Field(
        default=(
            "You are a routing supervisor. Based on the user's query, "
            "you must decide which agent is best suited to handle the request."
        )
    )

    @classmethod
    def from_env(cls, **overrides: Any) -> "OrchestratorConfig":
        """Create config from environment variables with optional overrides."""
        env_map: dict[str, Any] = {
            "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
            "default_model": os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"),
            "routing_system_instruction": os.getenv(
                "ROUTING_SYSTEM_INSTRUCTION",
                (
                    "You are a routing supervisor. Based on the user's query, "
                    "you must decide which agent is best suited to handle the request."
                ),
            ),
        }

        def _get_int(key: str, default: int) -> int:
            val = os.getenv(key)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError as e:
                raise ValueError(
                    f"Invalid environment variable value for {key}: '{val}' is not a valid integer."
                ) from e

        def _get_float(key: str, default: float) -> float:
            val = os.getenv(key)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError as e:
                raise ValueError(
                    f"Invalid environment variable value for {key}: '{val}' is not a valid float."
                ) from e

        env_map.update({
            "max_tool_rounds": _get_int("MAX_TOOL_ROUNDS", 5),
            "max_retries": _get_int("MAX_RETRIES", 3),
            "max_history": _get_int("MAX_HISTORY", 50),
            "agent_timeout": _get_float("AGENT_TIMEOUT", 120.0),
            "temperature": _get_float("TEMPERATURE", 0.2),
            "routing_temperature": _get_float("ROUTING_TEMPERATURE", 0.0),
            "propagate_errors": os.getenv("PROPAGATE_ERRORS", "false").lower() in ("true", "1", "yes"),
            "max_handoffs": _get_int("MAX_HANDOFFS", 5),
        })
        env_map.update(overrides)
        return cls(**env_map)
