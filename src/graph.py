"""
LangGraph StateGraph construction for HealthBot.
Builds the complete workflow connecting all single-responsibility nodes,
sequential edges, and conditional routing for restarts, multiple quizzes, and exit.
"""

from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import HealthBotState
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


def route_after_evaluation(state: HealthBotState) -> Literal["generate_quiz", "reset_state", "__end__"]:
    """
    Conditional edge router for workflow branching after the grade is presented:
    - 'another_question': Loops back to generate_quiz for another question on the same topic.
    - 'new_topic' or 'choose_related': Routes to reset_state, clearing prior data to maintain
      privacy and prevent cross-contamination, then returns to collect_topic.
    - 'exit': Concludes the patient education session.
    """
    action = (state.get("next_action") or "exit").strip().lower()

    if action in ["another_question", "quiz_again", "more_questions"]:
        return "generate_quiz"
    elif action in ["new_topic", "choose_related", "restart", "another_topic"]:
        return "reset_state"
    else:
        return "__end__"


def build_healthbot_graph(checkpointer: Optional[MemorySaver] = None):
    """
    Constructs and compiles the complete HealthBot LangGraph workflow.
    
    Nodes:
    1. collect_topic: Obtains or confirms the patient's medical topic & difficulty.
    2. generate_query: Formulates targeted medical search query.
    3. execute_search: Calls Tavily search via LangChain community tool.
    4. summarize_results: Synthesizes 3-4 paragraph explanation strictly using Tavily data.
    5. present_summary: Displays summary and checks patient readiness.
    6. generate_quiz: Generates a comprehension question strictly from the summary.
    7. present_quiz: Displays question and captures patient answer.
    8. evaluate_answer: Grades answer (A/B/C/D/F) and extracts citations from summary.
    9. present_grade: Formats and communicates grade and rationale.
    10. recommend_topics: Standout feature suggesting related health topics.
    11. ask_next_steps: Captures patient intent (more questions, new topic, or exit).
    12. reset_state: Resets state to protect privacy between topics.
    """
    workflow = StateGraph(HealthBotState)

    # Register all nodes
    workflow.add_node("collect_topic", collect_topic_node)
    workflow.add_node("generate_query", generate_query_node)
    workflow.add_node("execute_search", execute_search_node)
    workflow.add_node("summarize_results", summarize_results_node)
    workflow.add_node("present_summary", present_summary_node)
    workflow.add_node("generate_quiz", generate_quiz_node)
    workflow.add_node("present_quiz", present_quiz_node)
    workflow.add_node("evaluate_answer", evaluate_answer_node)
    workflow.add_node("present_grade", present_grade_node)
    workflow.add_node("recommend_topics", recommend_topics_node)
    workflow.add_node("ask_next_steps", ask_next_steps_node)
    workflow.add_node("reset_state", reset_state_node)

    # Sequential edges
    workflow.add_edge(START, "collect_topic")
    workflow.add_edge("collect_topic", "generate_query")
    workflow.add_edge("generate_query", "execute_search")
    workflow.add_edge("execute_search", "summarize_results")
    workflow.add_edge("summarize_results", "present_summary")
    workflow.add_edge("present_summary", "generate_quiz")
    workflow.add_edge("generate_quiz", "present_quiz")
    workflow.add_edge("present_quiz", "evaluate_answer")
    workflow.add_edge("evaluate_answer", "present_grade")
    workflow.add_edge("present_grade", "recommend_topics")
    workflow.add_edge("recommend_topics", "ask_next_steps")

    # Conditional branching after next steps
    workflow.add_conditional_edges(
        "ask_next_steps",
        route_after_evaluation,
        {
            "generate_quiz": "generate_quiz",
            "reset_state": "reset_state",
            "__end__": END,
        },
    )

    # Loop back from reset_state to collect_topic
    workflow.add_edge("reset_state", "collect_topic")

    # Compile the graph (with optional checkpointer)
    compiled_app = workflow.compile(checkpointer=checkpointer)
    return compiled_app


def get_graph_mermaid_diagram() -> str:
    """
    Returns Mermaid markdown diagram depicting the complete LangGraph workflow.
    """
    return """```mermaid
flowchart TD
    START([Start Session]) --> collect_topic["1. Collect Topic & Difficulty"]
    collect_topic --> generate_query["2. Formulate Search Query"]
    generate_query --> execute_search["3. Execute Tavily Search Tool"]
    execute_search --> summarize_results["4. Summarize Search Results (3-4 paragraphs)"]
    summarize_results --> present_summary["5. Present Summary & Await Ready"]
    present_summary --> generate_quiz["6. Generate Quiz Question (Strictly from Summary)"]
    generate_quiz --> present_quiz["7. Present Quiz & Collect Patient Answer"]
    present_quiz --> evaluate_answer["8. Evaluate Answer (Grade + Citations)"]
    evaluate_answer --> present_grade["9. Present Grade & Citations"]
    present_grade --> recommend_topics["10. Recommend Related Topics (Standout)"]
    recommend_topics --> ask_next_steps["11. Prompt Next Steps"]
    
    ask_next_steps -->|Another Question| generate_quiz
    ask_next_steps -->|New Topic / Related| reset_state["12. Reset State (Privacy & Accuracy)"]
    reset_state --> collect_topic
    ask_next_steps -->|Exit| END([End Session])
```"""

