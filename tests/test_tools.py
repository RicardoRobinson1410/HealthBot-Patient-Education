"""
Tests for Tavily search tool integration and result formatting.
"""

from unittest.mock import MagicMock
from langchain_community.tools.tavily_search import TavilySearchResults
from src.tools import create_tavily_tool, format_search_results, execute_tavily_search


def test_create_tavily_tool():
    """Verify tool initialization parameters."""
    tool = create_tavily_tool(max_results=4)
    assert isinstance(tool, TavilySearchResults)
    assert tool.max_results == 4


def test_format_search_results_with_list():
    """Verify list of dictionaries is formatted into readable context with titles and URLs."""
    raw = [
        {"title": "CDC Diabetes", "url": "https://cdc.gov/diabetes", "content": "Diabetes causes high glucose."},
        {"title": "NIH Care", "url": "https://nih.gov/care", "content": "Diet helps manage blood sugar."},
    ]
    formatted = format_search_results(raw)
    assert "[Source 1]: CDC Diabetes" in formatted
    assert "URL: https://cdc.gov/diabetes" in formatted
    assert "Excerpt: Diabetes causes high glucose." in formatted
    assert "[Source 2]: NIH Care" in formatted


def test_execute_tavily_search_with_mock_tool():
    """Verify execution of Tavily tool returns structured results and formatted context."""
    mock_tool = MagicMock(spec=TavilySearchResults)
    mock_tool.invoke.return_value = [
        {"title": "Test Title", "url": "https://test.org", "content": "Test content snippet"}
    ]

    res = execute_tavily_search("asthma", max_results=2, tavily_tool=mock_tool)
    assert "raw_results" in res
    assert "formatted_context" in res
    assert len(res["raw_results"]) == 1
    assert "Test Title" in res["formatted_context"]


def test_execute_tavily_search_fallback():
    """Verify fallback medical knowledge works when placeholder API key is active."""
    res = execute_tavily_search("hypertension", max_results=5)
    assert len(res["raw_results"]) > 0
    assert "hypertension" in res["formatted_context"].lower()
