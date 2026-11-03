"""
Unit tests for individual LangGraph nodes verifying single responsibility and state updates.
"""

from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from src.state import create_initial_state, reset_topic_state
from src.nodes import (
    collect_topic_node,
    generate_query_node,
    execute_search_node,
    summarize_results_node,
    present_summary_node,
    generate_quiz_node,
    present_quiz_node,
    evaluate_answer_node,
    present_grade_node,
    recommend_topics_node,
    ask_next_steps_node,
    reset_state_node,
)


def test_collect_topic_node():
    """Verify topic and difficulty are captured and message appended."""
    state = create_initial_state(topic="Asthma", difficulty="Beginner")
    output = collect_topic_node(state)

    assert output["topic"] == "Asthma"
    assert output["difficulty"] == "Beginner"
    assert output["session_active"] is True
    assert len(output["messages"]) == 1
    assert "Asthma" in output["messages"][0].content


def test_generate_query_node():
    """Verify query generation node produces search_query."""
    state = create_initial_state(topic="Hypertension", difficulty="Intermediate")
    output = generate_query_node(state)

    assert "search_query" in output
    assert len(output["search_query"]) > 0


def test_execute_search_node(monkeypatch):
    """Verify search node executes Tavily tool and updates search context."""
    state = create_initial_state(topic="Hypertension")
    state["search_query"] = "Hypertension overview"

    output = execute_search_node(state)
    assert "raw_search_results" in output
    assert "search_context" in output
    assert len(output["search_context"]) > 0


def test_summarize_results_node():
    """Verify 3-4 paragraph summary is produced and stored in state."""
    state = create_initial_state(topic="Hypertension")
    state["search_context"] = "Hypertension occurs when blood pressure is consistently high."

    output = summarize_results_node(state)
    assert "summary" in output
    assert len(output["summary"]) > 50
    assert "messages" in output
    paragraphs = [p for p in output["summary"].split("\n\n") if p.strip()]
    assert 3 <= len(paragraphs) <= 4


def test_present_summary_node():
    """Verify readiness flag is set."""
    state = create_initial_state(topic="Hypertension")
    output = present_summary_node(state)
    assert output["ready_for_quiz"] is True


def test_generate_quiz_node(sample_summary):
    """Verify quiz question and options are extracted."""
    state = create_initial_state(topic="Hypertension")
    state["summary"] = sample_summary

    output = generate_quiz_node(state)
    assert "quiz_question" in output
    assert len(output["quiz_question"]) > 0


def test_present_quiz_node():
    """Verify user answer is captured and added to messages."""
    state = create_initial_state(topic="Hypertension")
    state["user_answer"] = "Option B"

    output = present_quiz_node(state)
    assert output["user_answer"] == "Option B"
    assert len(output["messages"]) == 1
    assert "Option B" in output["messages"][0].content


def test_evaluate_answer_node(sample_summary):
    """Verify evaluation generates letter grade, explanation, citations, and history."""
    state = create_initial_state(topic="Hypertension")
    state["summary"] = sample_summary
    state["quiz_question"] = "What reading indicates hypertension?"
    state["user_answer"] = "Consistently above 130/80 mm Hg"

    output = evaluate_answer_node(state)
    assert "quiz_grade" in output
    assert output["quiz_grade"] in ["A", "B", "C", "D", "F"]
    assert "quiz_explanation" in output
    assert "citations" in output
    assert len(output["quiz_history"]) == 1
    assert len(output["messages"]) == 1


def test_present_grade_node():
    """Verify grade presentation passes through grade and citations."""
    state = create_initial_state(topic="Hypertension")
    state["quiz_grade"] = "A"
    state["quiz_explanation"] = "Well done!"
    state["citations"] = ["DASH diet is recommended."]

    output = present_grade_node(state)
    assert output["quiz_grade"] == "A"
    assert output["quiz_explanation"] == "Well done!"
    assert output["citations"] == ["DASH diet is recommended."]


def test_recommend_topics_node(sample_summary):
    """Verify 3-4 related medical topics are recommended."""
    state = create_initial_state(topic="Hypertension")
    state["summary"] = sample_summary

    output = recommend_topics_node(state)
    assert "related_topics" in output
    assert isinstance(output["related_topics"], list)
    assert len(output["related_topics"]) >= 1


def test_ask_next_steps_node():
    """Verify next action is captured."""
    state = create_initial_state(topic="Hypertension")
    state["next_action"] = "new_topic"

    output = ask_next_steps_node(state)
    assert output["next_action"] == "new_topic"


def test_reset_state_node(sample_summary):
    """Verify reset state cleans topic and summary data."""
    state = create_initial_state(topic="Hypertension")
    state["summary"] = sample_summary
    output = reset_state_node(state)
    assert output["topic"] == ""
    assert output["summary"] == ""

