"""
Tests for configuration loading, API key validation, and tool initialization.
"""

import os
import pytest
from src.config import load_env_config, validate_environment, get_llm, get_tavily_tool


def test_load_env_config():
    """Verify load_env_config executes without throwing errors."""
    loaded = load_env_config("config.env")
    assert isinstance(loaded, bool)


def test_validate_environment_success(monkeypatch):
    """Verify validate_environment passes when keys are present."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-valid-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-valid-key")
    assert validate_environment(raise_error=True) is True


def test_validate_environment_missing_keys(monkeypatch):
    """Verify validate_environment raises ValueError when keys are absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert validate_environment(raise_error=False) is False

    with pytest.raises(ValueError) as exc_info:
        validate_environment(raise_error=True)
    assert "Missing required environment variables" in str(exc_info.value)


def test_get_llm_initialization(monkeypatch):
    """Verify ChatOpenAI initializes with proper model and temperature."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    llm = get_llm(model_name="gpt-4o-mini", temperature=0.1)
    assert llm.model_name == "gpt-4o-mini"
    assert llm.temperature == 0.1


def test_get_tavily_tool_initialization(monkeypatch):
    """Verify TavilySearchResults initializes with correct max_results."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-mock-key")
    tool = get_tavily_tool(max_results=3)
    assert tool.max_results == 3
