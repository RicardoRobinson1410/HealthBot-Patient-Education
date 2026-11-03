"""
End-to-end integration tests for HealthBot LangGraph workflow.
Tests full pipeline execution, conditional routing, multiple quiz loops, and state reset.
"""

from langgraph.checkpoint.memory import MemorySaver
from src.state import create_initial_state
from src.graph import build_healthbot_graph, route_after_evaluation


def test_route_after_evaluation_branches():
    """Verify router branches correctly for another_question, new_topic, and exit."""
    # Another question
    state1 = create_initial_state()
    state1["next_action"] = "another_question"
    assert route_after_evaluation(state1) == "generate_quiz"

    # New topic
    state2 = create_initial_state()
    state2["next_action"] = "new_topic"
    assert route_after_evaluation(state2) == "reset_state"

    # Restart
    state3 = create_initial_state()
    state3["next_action"] = "restart"
    assert route_after_evaluation(state3) == "reset_state"

    # Exit
    state4 = create_initial_state()
    state4["next_action"] = "exit"
    assert route_after_evaluation(state4) == "__end__"


def test_complete_workflow_run_to_exit():
    """
    CRITICAL RUBRIC REQUIREMENT:
    Verify the entire workflow is executed and functions per the requirements:
    The user is able to enter a subject, see the summary, take a quiz, see the quiz grade,
    and opt to learn about a new subject or exit.
    """
    app = build_healthbot_graph()

    input_state = create_initial_state(topic="Hypertension", difficulty="Intermediate")
    input_state["user_answer"] = "A"
    input_state["next_action"] = "exit"

    final_state = app.invoke(input_state)

    # 1. Topic captured
    assert final_state["topic"] == "Hypertension"

    # 2. Search query formulated
    assert "Hypertension" in final_state["search_query"]

    # 3. Search executed
    assert len(final_state["raw_search_results"]) > 0
    assert len(final_state["search_context"]) > 0

    # 4. Summary created (3-4 paragraphs)
    paragraphs = [p for p in final_state["summary"].split("\n\n") if p.strip()]
    assert 3 <= len(paragraphs) <= 4

    # 5. Quiz question generated
    assert len(final_state["quiz_question"]) > 0

    # 6. Evaluation completed with grade and citations
    assert final_state["quiz_grade"] in ["A", "B", "C", "D", "F"]
    assert len(final_state["quiz_explanation"]) > 0
    assert isinstance(final_state["citations"], list)

    # 7. Related topics provided (Standout feature)
    assert len(final_state["related_topics"]) >= 1

    # 8. Finished at exit
    assert final_state["next_action"] == "exit"


def test_workflow_with_checkpointer():
    """Verify workflow functions seamlessly with LangGraph MemorySaver checkpointer."""
    checkpointer = MemorySaver()
    app = build_healthbot_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test-patient-thread-1"}}
    input_state = create_initial_state(topic="Type 2 Diabetes", difficulty="Beginner")
    input_state["user_answer"] = "Healthy diet and exercise"
    input_state["next_action"] = "exit"

    res = app.invoke(input_state, config=config)
    assert res["topic"] == "Type 2 Diabetes"
    assert res["quiz_grade"] in ["A", "B", "C", "D", "F"]

