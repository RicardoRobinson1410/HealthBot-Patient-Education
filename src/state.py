"""
State definition for HealthBot LangGraph workflow.
Defines the HealthBotState TypedDict, reducers, and helper functions for state
initialization and privacy-preserving resets between patient topics.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages


class HealthBotState(TypedDict, total=False):
    """
    State object referenced and updated by all nodes across the HealthBot workflow.
    
    Attributes:
        messages: Conversation message history with LangGraph's add_messages reducer.
        topic: The health topic or medical condition chosen by the patient.
        difficulty: Difficulty level ('Beginner', 'Intermediate', 'Advanced').
        search_query: The specific search query sent to Tavily.
        raw_search_results: Raw result list returned by TavilySearchResults.
        search_context: Formatted string of Tavily search results.
        summary: 3-4 paragraph patient-friendly explanation strictly grounded in search results.
        ready_for_quiz: Boolean flag indicating patient is ready for comprehension check.
        quiz_question: Single quiz question based strictly on the summary.
        quiz_options: Optional multiple-choice options or open-ended prompt guidance.
        quiz_history: Record of all quiz questions and evaluations taken in this topic.
        user_answer: Patient's submitted answer to the quiz question.
        quiz_grade: Letter grade ('A', 'B', 'C', 'D', 'F') evaluated by the model.
        quiz_explanation: Pedagogical explanation justifying the grade with citations.
        citations: List of exact or cited passages from the summary reinforcing learning.
        related_topics: Curated suggestions for related medical topics (standout feature).
        next_action: User choice for next step ('another_question', 'new_topic', 'exit').
        session_active: Whether the education session is ongoing.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    difficulty: str
    search_query: str
    raw_search_results: List[Dict[str, Any]]
    search_context: str
    summary: str
    ready_for_quiz: bool
    quiz_question: str
    quiz_options: List[str]
    quiz_history: List[Dict[str, Any]]
    user_answer: str
    quiz_grade: str
    quiz_explanation: str
    citations: List[str]
    related_topics: List[str]
    next_action: str
    session_active: bool


def create_initial_state(
    topic: str = "",
    difficulty: str = "Intermediate",
    system_prompt: Optional[str] = None,
) -> HealthBotState:
    """
    Initializes a clean HealthBotState.
    """
    initial_messages: List[BaseMessage] = []
    if system_prompt:
        initial_messages.append(SystemMessage(content=system_prompt))

    return {
        "messages": initial_messages,
        "topic": topic,
        "difficulty": difficulty,
        "search_query": "",
        "raw_search_results": [],
        "search_context": "",
        "summary": "",
        "ready_for_quiz": False,
        "quiz_question": "",
        "quiz_options": [],
        "quiz_history": [],
        "user_answer": "",
        "quiz_grade": "",
        "quiz_explanation": "",
        "citations": [],
        "related_topics": [],
        "next_action": "",
        "session_active": True,
    }


def reset_topic_state(current_state: HealthBotState) -> Dict[str, Any]:
    """
    Resets topic-specific data when a patient chooses to explore a new medical topic.
    Ensures patient privacy, clears prior medical context, and prevents hallucinations
    or cross-contamination between conditions.
    """
    return {
        "topic": "",
        "search_query": "",
        "raw_search_results": [],
        "search_context": "",
        "summary": "",
        "ready_for_quiz": False,
        "quiz_question": "",
        "quiz_options": [],
        "quiz_history": [],
        "user_answer": "",
        "quiz_grade": "",
        "quiz_explanation": "",
        "citations": [],
        "related_topics": [],
        "next_action": "",
        "session_active": True,
    }

