"""
Tests for LangGraph State definition, message history, and privacy-preserving reset.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import create_initial_state, reset_topic_state, HealthBotState


def test_initial_state_structure():
    """Verify initial state contains all required properties with proper defaults."""
    state = create_initial_state(topic="Asthma", difficulty="Beginner")

    assert state["topic"] == "Asthma"
    assert state["difficulty"] == "Beginner"
    assert state["search_query"] == ""
    assert state["raw_search_results"] == []
    assert state["summary"] == ""
    assert state["ready_for_quiz"] is False
    assert state["quiz_question"] == ""
    assert state["user_answer"] == ""
    assert state["quiz_grade"] == ""
    assert state["citations"] == []
    assert state["related_topics"] == []
    assert state["session_active"] is True
    assert isinstance(state["messages"], list)


def test_initial_state_with_system_prompt():
    """Verify system message is placed in messages when provided."""
    state = create_initial_state(topic="Diabetes", system_prompt="System instructions")
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], SystemMessage)
    assert state["messages"][0].content == "System instructions"


def test_reset_topic_state_preserves_privacy():
    """
    CRITICAL RUBRIC REQUIREMENT:
    Ensure state is reset when starting a new topic to maintain patient privacy and accuracy.
    """
    prior_state: HealthBotState = {
        "messages": [HumanMessage(content="My blood pressure is 150/95"), AIMessage(content="Here is advice")],
        "topic": "Hypertension",
        "difficulty": "Advanced",
        "search_query": "Hypertension crisis",
        "raw_search_results": [{"title": "Source 1", "content": "Sensitive medical data"}],
        "search_context": "Sensitive clinical context",
        "summary": "Confidential patient summary text...",
        "ready_for_quiz": True,
        "quiz_question": "What is hypertension?",
        "quiz_options": ["A) High BP", "B) Low BP"],
        "quiz_history": [{"question": "Q1", "grade": "A"}],
        "user_answer": "A",
        "quiz_grade": "A",
        "quiz_explanation": "Detailed explanation with citations",
        "citations": ["Exact quote from prior summary"],
        "related_topics": ["Stroke", "Aneurysm"],
        "next_action": "new_topic",
        "session_active": True,
    }

    reset_state = reset_topic_state(prior_state)

    # Topic and summaries must be completely cleared
    assert reset_state["topic"] == ""
    assert reset_state["search_query"] == ""
    assert reset_state["raw_search_results"] == []
    assert reset_state["search_context"] == ""
    assert reset_state["summary"] == ""
    assert reset_state["ready_for_quiz"] is False
    assert reset_state["quiz_question"] == ""
    assert reset_state["quiz_options"] == []
    assert reset_state["quiz_history"] == []
    assert reset_state["user_answer"] == ""
    assert reset_state["quiz_grade"] == ""
    assert reset_state["quiz_explanation"] == ""
    assert reset_state["citations"] == []
    assert reset_state["related_topics"] == []
    assert reset_state["session_active"] is True
