"""
Tests for prompt generation, strict grounding constraints, and rubric criteria compliance.
"""

from src.prompts import (
    HEALTHBOT_SYSTEM_PROMPT,
    get_search_query_prompt,
    get_summarization_prompt,
    get_quiz_generation_prompt,
    get_quiz_evaluation_prompt,
    get_related_topics_prompt,
)


def test_summarization_prompt_strict_grounding_and_paragraph_count():
    """
    CRITICAL RUBRIC REQUIREMENT:
    - Summarization prompt enforces using NO OTHER knowledge or data sources.
    - Specifies 3-4 paragraphs.
    """
    prompt = get_summarization_prompt(
        topic="Type 2 Diabetes",
        search_results="Sample search excerpt",
        difficulty="Intermediate",
    )

    # Verify strict grounding directives
    assert "ONLY the provided Tavily search results" in prompt
    assert "DO NOT use any other knowledge, external databases, assumptions" in prompt
    assert "3 TO 4 PARAGRAPHS" in prompt
    assert "Paragraph 1: Overview" in prompt
    assert "Paragraph 2: Common symptoms" in prompt
    assert "Paragraph 3: Treatments" in prompt
    assert "Paragraph 4: When to contact a doctor" in prompt
    assert "TAVILY SEARCH RESULTS (YOUR EXCLUSIVE SOURCE OF TRUTH)" in prompt


def test_summarization_prompt_difficulty_levels():
    """Verify different pedagogical guidance is included for each difficulty level."""
    p_beg = get_summarization_prompt("Asthma", "res", "Beginner")
    assert "6th-grade reading level" in p_beg

    p_int = get_summarization_prompt("Asthma", "res", "Intermediate")
    assert "essential medical terminology" in p_int

    p_adv = get_summarization_prompt("Asthma", "res", "Advanced")
    assert "physiological mechanisms" in p_adv


def test_quiz_generation_prompt_grounding():
    """
    CRITICAL RUBRIC REQUIREMENT:
    Quiz prompt must specify question is answerable using the summary ALONE.
    """
    prompt = get_quiz_generation_prompt(
        topic="Hypertension",
        summary="Patient summary text...",
        difficulty="Intermediate",
    )

    assert "SINGLE relevant quiz question based EXCLUSIVELY on the provided summary" in prompt
    assert "100% answerable using ONLY the information written in the summary" in prompt
    assert "PATIENT SUMMARY (YOUR EXCLUSIVE SOURCE OF TRUTH)" in prompt
    assert "Correct Option:" in prompt
    assert "Summary Reference:" in prompt


def test_quiz_evaluation_prompt_grade_and_citations():
    """
    CRITICAL RUBRIC REQUIREMENT:
    Evaluation prompt must require a letter grade (A, B, C, etc.), justification, and citations.
    """
    prompt = get_quiz_evaluation_prompt(
        question="What reading indicates hypertension?",
        user_answer="Over 130/80 mm Hg",
        summary="Hypertension is defined as over 130/80 mm Hg.",
    )

    assert "using ONLY the provided summary as the source of truth" in prompt
    assert "Assign a letter grade according to these criteria" in prompt
    assert "'A': Completely accurate" in prompt
    assert "Quote the exact sentence or excerpt from the summary as a CITATION" in prompt
    assert "Grade: <A, B, C, D, or F>" in prompt
    assert "Explanation:" in prompt
    assert "Citation:" in prompt


def test_related_topics_prompt():
    """Verify prompt generates 3-4 related medical subjects."""
    prompt = get_related_topics_prompt("Hypertension", "Summary text...")
    assert "recommend 3 to 4 related medical subjects" in prompt

