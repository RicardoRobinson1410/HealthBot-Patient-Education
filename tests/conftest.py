"""
Pytest fixtures and configuration for HealthBot test suite.
"""

import os
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from src.state import HealthBotState, create_initial_state


@pytest.fixture(autouse=True)
def mock_environment(monkeypatch):
    """
    Ensures safe dummy API keys are present during tests so validation passes.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-openai-key-for-unit-tests")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-mock-tavily-key-for-unit-tests")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")


@pytest.fixture
def sample_search_results():
    return [
        {
            "title": "Hypertension Overview - Mayo Clinic",
            "url": "https://www.mayoclinic.org/hypertension",
            "content": "Hypertension is high blood pressure exceeding 130/80 mm Hg. Symptoms are rare, making regular checks vital. Management includes low sodium diet and exercise.",
        },
        {
            "title": "High Blood Pressure Risks - CDC",
            "url": "https://www.cdc.gov/bloodpressure",
            "content": "Uncontrolled hypertension increases the risk of stroke and heart attack. Medications include ACE inhibitors.",
        },
    ]


@pytest.fixture
def sample_summary():
    return (
        "Hypertension, commonly known as high blood pressure, occurs when the force of blood against the arterial walls is consistently elevated above 130/80 mm Hg.\n\n"
        "Often called a 'silent killer', high blood pressure rarely presents overt warning symptoms in its initial stages, though sustained elevation damages blood vessels and organs.\n\n"
        "Effective clinical management requires adopting the DASH diet with restricted sodium, engaging in routine aerobic exercise, and taking prescribed antihypertensive medications.\n\n"
        "Patients experiencing acute blood pressure surges accompanied by chest discomfort or severe headache must seek emergency medical intervention immediately."
    )


@pytest.fixture
def base_state(sample_summary, sample_search_results):
    state = create_initial_state(topic="Hypertension", difficulty="Intermediate")
    state["summary"] = sample_summary
    state["raw_search_results"] = sample_search_results
    state["search_context"] = "Sample context from Tavily."
    state["quiz_question"] = "What reading signifies high blood pressure?"
    state["user_answer"] = "130/80 mm Hg"
    return state

