import pytest


@pytest.fixture(autouse=True)
def mock_api_key(monkeypatch):
    """Ensure GEMINI_API_KEY is set for all tests."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-tests")
