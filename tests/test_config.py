import pytest
from pydantic import ValidationError

from multi_agent_orchestrator.core.config import OrchestratorConfig


def test_default_config():
    """Config with all defaults should have sensible values."""
    config = OrchestratorConfig()
    assert config.gemini_api_key == ""
    assert config.default_model == "gemini-2.5-flash"
    assert config.max_tool_rounds == 5
    assert config.max_retries == 3
    assert config.max_history == 50
    assert config.agent_timeout == 120.0
    assert config.temperature == 0.2
    assert config.routing_temperature == 0.0


def test_config_explicit_values():
    """Explicit constructor values should override defaults."""
    config = OrchestratorConfig(
        gemini_api_key="test-key",
        default_model="gemini-2.0-flash",
        max_tool_rounds=10,
        max_retries=5,
        max_history=200,
        agent_timeout=60.0,
        temperature=1.0,
        routing_temperature=0.5,
    )
    assert config.gemini_api_key == "test-key"
    assert config.default_model == "gemini-2.0-flash"
    assert config.max_tool_rounds == 10
    assert config.max_retries == 5
    assert config.max_history == 200
    assert config.agent_timeout == 60.0
    assert config.temperature == 1.0
    assert config.routing_temperature == 0.5


def test_config_from_env(monkeypatch):
    """from_env() should read from environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "env-key-123")
    monkeypatch.setenv("DEFAULT_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("MAX_TOOL_ROUNDS", "10")
    monkeypatch.setenv("TEMPERATURE", "0.9")

    config = OrchestratorConfig.from_env()
    assert config.gemini_api_key == "env-key-123"
    assert config.default_model == "gemini-2.0-flash"
    assert config.max_tool_rounds == 10
    assert config.temperature == 0.9


def test_config_from_env_with_overrides(monkeypatch):
    """Overrides passed to from_env() should take precedence over env vars."""
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")

    config = OrchestratorConfig.from_env(gemini_api_key="override-key", max_history=100)
    assert config.gemini_api_key == "override-key"
    assert config.max_history == 100


def test_config_validation_max_tool_rounds():
    """max_tool_rounds must be >= 1."""
    with pytest.raises(ValidationError):
        OrchestratorConfig(max_tool_rounds=0)


def test_config_validation_max_retries():
    """max_retries must be >= 1."""
    with pytest.raises(ValidationError):
        OrchestratorConfig(max_retries=0)


def test_config_validation_max_history():
    """max_history must be >= 1."""
    with pytest.raises(ValidationError):
        OrchestratorConfig(max_history=0)


def test_config_validation_agent_timeout():
    """agent_timeout must be > 0."""
    with pytest.raises(ValidationError):
        OrchestratorConfig(agent_timeout=0)


def test_config_validation_temperature_range():
    """temperature must be in [0, 2]."""
    with pytest.raises(ValidationError):
        OrchestratorConfig(temperature=-0.1)
    with pytest.raises(ValidationError):
        OrchestratorConfig(temperature=2.1)


def test_config_validation_routing_temperature_range():
    """routing_temperature must be in [0, 2]."""
    with pytest.raises(ValidationError):
        OrchestratorConfig(routing_temperature=-0.1)
    with pytest.raises(ValidationError):
        OrchestratorConfig(routing_temperature=2.1)


def test_config_propagate_errors_defaults_and_env(monkeypatch):
    """Verify default value and environment reading for propagate_errors."""
    config = OrchestratorConfig()
    assert config.propagate_errors is False

    monkeypatch.setenv("PROPAGATE_ERRORS", "true")
    config_env = OrchestratorConfig.from_env()
    assert config_env.propagate_errors is True

    monkeypatch.setenv("PROPAGATE_ERRORS", "1")
    assert OrchestratorConfig.from_env().propagate_errors is True

    monkeypatch.setenv("PROPAGATE_ERRORS", "false")
    assert OrchestratorConfig.from_env().propagate_errors is False


def test_config_max_handoffs_defaults_and_env(monkeypatch):
    """Verify default value and environment reading for max_handoffs."""
    config = OrchestratorConfig()
    assert config.max_handoffs == 5

    monkeypatch.setenv("MAX_HANDOFFS", "10")
    config_env = OrchestratorConfig.from_env()
    assert config_env.max_handoffs == 10


def test_config_validation_max_handoffs():
    """max_handoffs must be >= 1."""
    with pytest.raises(ValidationError):
        OrchestratorConfig(max_handoffs=0)


def test_config_routing_system_instruction_defaults_and_env(monkeypatch):
    """Verify default value and environment reading for routing_system_instruction."""
    config = OrchestratorConfig()
    assert "You are a routing supervisor" in config.routing_system_instruction

    monkeypatch.setenv("ROUTING_SYSTEM_INSTRUCTION", "Custom Supervisor Instructions")
    config_env = OrchestratorConfig.from_env()
    assert config_env.routing_system_instruction == "Custom Supervisor Instructions"


def test_config_from_env_invalid_int(monkeypatch):
    """from_env() should raise ValueError on invalid integers in env."""
    monkeypatch.setenv("MAX_TOOL_ROUNDS", "not-an-int")
    with pytest.raises(ValueError) as excinfo:
        OrchestratorConfig.from_env()
    assert "Invalid environment variable value for MAX_TOOL_ROUNDS" in str(excinfo.value)


def test_config_from_env_invalid_float(monkeypatch):
    """from_env() should raise ValueError on invalid floats in env."""
    monkeypatch.setenv("TEMPERATURE", "not-a-float")
    with pytest.raises(ValueError) as excinfo:
        OrchestratorConfig.from_env()
    assert "Invalid environment variable value for TEMPERATURE" in str(excinfo.value)
