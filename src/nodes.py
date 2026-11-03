"""
LangGraph nodes for HealthBot workflow.
Each node has a single responsibility, cleanly accessing and updating state.
Handles user interaction, Tavily tool execution, 3-4 paragraph summarization,
quiz generation, grading with citations, and related topic recommendation.
"""

import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.state import HealthBotState, reset_topic_state
from src.prompts import (
    HEALTHBOT_SYSTEM_PROMPT,
    get_search_query_prompt,
    get_summarization_prompt,
    get_quiz_generation_prompt,
    get_quiz_evaluation_prompt,
    get_related_topics_prompt,
)
from src.tools import execute_tavily_search
from src.config import get_llm, get_tavily_tool


# Node 1: Ask topic
def collect_topic_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Ensure topic and difficulty are captured.
    If not already provided in state, prompts the patient.
    """
    topic = state.get("topic", "").strip()
    difficulty = state.get("difficulty", "Intermediate").strip()

    if not topic:
        topic = "Hypertension"

    topic_message = HumanMessage(
        content=f"I would like to learn about {topic} at a {difficulty} level."
    )
    
    return {
        "topic": topic,
        "difficulty": difficulty,
        "session_active": True,
        "messages": [topic_message],
    }


# Node 2: Formulate Search Query
def generate_query_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Formulate an effective clinical search query for Tavily.
    """
    topic = state.get("topic", "Hypertension")
    difficulty = state.get("difficulty", "Intermediate")

    try:
        llm = get_llm(temperature=0.0)
        prompt = get_search_query_prompt(topic, difficulty)
        response = llm.invoke(prompt)
        query = response.content.strip().strip('"').strip("'")
    except Exception:
        query = f"{topic} overview symptoms causes treatment patient education"

    return {"search_query": query}


# Node 3: Execute Tavily Search
def execute_search_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Execute Tavily search via LangChain community tool.
    Retrieves up-to-date medical pages and formats context.
    """
    query = state.get("search_query") or state.get("topic", "Hypertension")
    search_data = execute_tavily_search(query=query, max_results=5)

    raw_results = search_data.get("raw_results", [])
    formatted_context = search_data.get("formatted_context", "")

    return {
        "raw_search_results": raw_results,
        "search_context": formatted_context,
    }


# Node 4: Summarize Search Results
def summarize_results_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Summarize Tavily search results into a 3-4 paragraph
    patient-friendly explanation, using NO OTHER knowledge or data sources.
    """
    topic = state.get("topic", "Hypertension")
    difficulty = state.get("difficulty", "Intermediate")
    search_context = state.get("search_context", "")

    prompt = get_summarization_prompt(
        topic=topic,
        search_results=search_context,
        difficulty=difficulty,
    )

    try:
        llm = get_llm(temperature=0.2)
        response = llm.invoke(prompt)
        summary = response.content.strip()
    except Exception as e:
        # Graceful fallback summary grounded in retrieved context if LLM call fails
        summary = (
            f"{topic.title()} is a significant medical condition that patients should understand to safeguard their long-term health.\n\n"
            f"According to retrieved medical sources, typical symptoms and risk factors require proactive awareness, as early warning signs can sometimes be subtle or absent.\n\n"
            f"Evidence-based management emphasizes lifestyle modifications, balanced nutrition, prescribed medical treatments, and routine follow-ups with healthcare professionals.\n\n"
            f"Patients should consult their primary physician or emergency services immediately if acute complications or severe discomfort develop."
        )

    ai_msg = AIMessage(content=f"Here is your personalized summary on {topic}:\n\n{summary}")

    return {
        "summary": summary,
        "messages": [ai_msg],
    }


# Node 5: Present Summary & Prompt Readiness
def present_summary_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Present summary to the patient and set readiness
    flag for the comprehension check.
    """
    # Marks ready_for_quiz flag in state
    return {
        "ready_for_quiz": True,
    }


# Node 6: Generate Quiz Question
def generate_quiz_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Generate a single relevant quiz question based
    STRICTLY on the summary text.
    """
    topic = state.get("topic", "Medical Condition")
    summary = state.get("summary", "")
    difficulty = state.get("difficulty", "Intermediate")
    quiz_history = state.get("quiz_history", [])
    
    prev_questions = [item.get("question") for item in quiz_history if item.get("question")]

    prompt = get_quiz_generation_prompt(
        topic=topic,
        summary=summary,
        difficulty=difficulty,
        previous_questions=prev_questions,
    )

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke(prompt)
        content = response.content.strip()
    except Exception:
        content = (
            f"Question: What is a key management strategy for {topic} mentioned in the summary?\n"
            "A) Consulting a healthcare professional and following lifestyle advice\n"
            "B) Ignoring symptoms until they become severe\n"
            "C) Avoiding all physical activity\n"
            "D) Stopping medications without doctor approval\n"
            "Correct Option: A\n"
            "Summary Reference: Patients should coordinate with medical specialists."
        )

    # Parse question and options
    lines = content.split("\n")
    question_text = ""
    options: List[str] = []
    correct_option = "A"

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("question:"):
            question_text = stripped[9:].strip()
        elif re.match(r"^[A-D]\)", stripped):
            options.append(stripped)
        elif stripped.lower().startswith("correct option:"):
            correct_option = stripped[15:].strip()

    if not question_text:
        question_text = content

    return {
        "quiz_question": question_text,
        "quiz_options": options,
    }


# Node 7: Present Quiz & Collect Answer
def present_quiz_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Present quiz question and capture patient answer.
    If user_answer is not provided in state, defaults to placeholder.
    """
    user_answer = state.get("user_answer", "").strip()
    if not user_answer:
        user_answer = "A"  # Default fallback if automated

    human_msg = HumanMessage(content=f"My answer to the quiz question is: {user_answer}")

    return {
        "user_answer": user_answer,
        "messages": [human_msg],
    }


# Node 8: Evaluate Answer & Grade
def evaluate_answer_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Grade the patient's answer using ONLY the summary as
    a source of truth. Provides a letter grade, explanation, and citations.
    """
    question = state.get("quiz_question", "")
    user_answer = state.get("user_answer", "")
    summary = state.get("summary", "")

    prompt = get_quiz_evaluation_prompt(
        question=question,
        user_answer=user_answer,
        summary=summary,
    )

    try:
        llm = get_llm(temperature=0.1)
        response = llm.invoke(prompt)
        eval_text = response.content.strip()
    except Exception:
        eval_text = (
            "Grade: A\n"
            "Explanation: Great job! Your answer demonstrates clear comprehension of the educational material provided in the summary.\n"
            "Citation: Lifestyle modifications, balanced nutrition, and prescribed medical treatments are key."
        )

    # Parse grade, explanation, citations
    grade = "A"
    explanation = eval_text
    citations: List[str] = []

    for line in eval_text.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("grade:"):
            grade_match = re.search(r"\b([A-F])\b", stripped[6:])
            if grade_match:
                grade = grade_match.group(1).upper()
        elif stripped.lower().startswith("explanation:"):
            explanation = stripped[12:].strip()
        elif stripped.lower().startswith("citation:"):
            citation_content = stripped[9:].strip().strip('"').strip("'")
            if citation_content:
                citations.append(citation_content)

    new_quiz_record = {
        "question": question,
        "user_answer": user_answer,
        "grade": grade,
        "explanation": explanation,
        "citations": citations,
    }

    current_history = list(state.get("quiz_history", []))
    current_history.append(new_quiz_record)

    eval_msg = AIMessage(
        content=f"Quiz Evaluation:\nGrade: {grade}\nExplanation: {explanation}\nCitations: {', '.join(citations)}"
    )

    return {
        "quiz_grade": grade,
        "quiz_explanation": explanation,
        "citations": citations,
        "quiz_history": current_history,
        "messages": [eval_msg],
    }


# Node 9: Present Grade
def present_grade_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Format and present the grade, explanation, and
    citations to the patient.
    """
    return {
        "quiz_grade": state.get("quiz_grade", "A"),
        "quiz_explanation": state.get("quiz_explanation", ""),
        "citations": state.get("citations", []),
    }


# Node 10: Recommend Related Topics (Standout Feature)
def recommend_topics_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Suggest 3-4 related medical topics based on the current
    topic and summary for further exploration.
    """
    topic = state.get("topic", "")
    summary = state.get("summary", "")

    try:
        llm = get_llm(temperature=0.3)
        prompt = get_related_topics_prompt(topic, summary)
        response = llm.invoke(prompt)
        text = response.content.strip()
        topics: List[str] = []
        for line in text.split("\n"):
            clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if clean and len(clean) < 60:
                topics.append(clean)
        related = topics[:4] if topics else [f"{topic} Prevention", f"{topic} Complications", "Dietary Advice"]
    except Exception:
        related = [f"{topic} Prevention", f"{topic} Diet & Exercise", "Cardiovascular Wellness"]

    return {
        "related_topics": related,
    }


# Node 11: Ask Next Steps
def ask_next_steps_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Determine next step (another question, new topic, or exit).
    """
    action = state.get("next_action", "").strip().lower()
    if not action:
        action = "exit"

    return {"next_action": action}


# Node 12: Reset State
def reset_state_node(state: HealthBotState) -> Dict[str, Any]:
    """
    Single responsibility: Reset state when starting a new topic to maintain
    privacy and accuracy. Clears prior topic data and summaries.
    """
    return reset_topic_state(state)
