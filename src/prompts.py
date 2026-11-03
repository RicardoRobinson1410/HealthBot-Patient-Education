"""
Prompt templates for HealthBot patient education system.
Enforces strict grounding:
1. Summarization strictly uses ONLY Tavily search results (3-4 paragraphs).
2. Quiz generation strictly uses ONLY the summary.
3. Grading strictly uses ONLY the summary and provides letter grades + citations.
"""

from typing import Optional


HEALTHBOT_SYSTEM_PROMPT = """You are HealthBot, an empathetic, highly reliable AI Patient Education Specialist at MediTech Solutions.
Your mission is to empower patients with clear, accessible, and medically grounded education to improve health outcomes and treatment adherence.

Strict Operational Guidelines:
1. Grounding Integrity: Always base summaries strictly on the provided Tavily search results. Never hallucinate or add facts from outside sources.
2. Patient-Centered Communication: Present information in a compassionate, respectful, and easy-to-understand manner.
3. Educational Reinforcement: Ensure comprehension quizzes and evaluations directly test the information provided, reinforcing accurate knowledge with direct citations.
4. Medical Disclaimer: Always remind patients that HealthBot is an educational assistant and does not replace professional medical diagnosis, advice, or emergency care.
"""


def get_search_query_prompt(topic: str, difficulty: str = "Intermediate") -> str:
    """
    Constructs prompt to generate optimal medical search query for Tavily.
    """
    return f"""A patient has requested education on the following medical topic:
Topic: "{topic}"
Patient Education Level: {difficulty}

Formulate a targeted, clinical search query to retrieve trustworthy, up-to-date patient education information on this topic (overview, symptoms, causes, management, when to seek care).
Return only the concise search query string.
"""


def get_summarization_prompt(
    topic: str,
    search_results: str,
    difficulty: str = "Intermediate",
) -> str:
    """
    Constructs prompt for 3-4 paragraph summarization strictly grounded in Tavily search results.
    """
    difficulty_instructions = {
        "Beginner": (
            "- Use simple, everyday vocabulary (around a 6th-grade reading level).\n"
            "- Explain medical terms immediately with gentle analogies.\n"
            "- Keep sentence structures clear and straightforward."
        ),
        "Intermediate": (
            "- Use accessible patient-education language while preserving essential medical terminology.\n"
            "- Clearly define medical terms upon first mention.\n"
            "- Balance practical patient guidance with clear clinical explanations."
        ),
        "Advanced": (
            "- Include comprehensive clinical detail, physiological mechanisms, and treatment nuances.\n"
            "- Maintain professional clinical clarity suitable for an engaged, health-literate patient.\n"
            "- Discuss diagnostic pathways and pharmacological or surgical considerations mentioned in the search data."
        ),
    }

    difficulty_guide = difficulty_instructions.get(difficulty, difficulty_instructions["Intermediate"])

    return f"""You are summarizing medical search results for a patient learning about: "{topic}".

CRITICAL INSTRUCTIONS - STRICT GROUNDING REQUIREMENT:
1. You MUST summarize the information using ONLY the provided Tavily search results below.
2. DO NOT use any other knowledge, external databases, assumptions, or unverified facts. If the search results do not mention a detail, DO NOT include it.
3. You MUST format your explanation into EXACTLY 3 TO 4 PARAGRAPHS:
   - Paragraph 1: Overview and clear definition of {topic}.
   - Paragraph 2: Common symptoms, causes, or risk factors described in the search results.
   - Paragraph 3: Treatments, management strategies, or lifestyle actions described in the search results.
   - Paragraph 4: When to contact a doctor / key educational takeaway for the patient.
4. Difficulty Level: {difficulty}
{difficulty_guide}

TAVILY SEARCH RESULTS (YOUR EXCLUSIVE SOURCE OF TRUTH):
---
{search_results}
---

Provide the 3-4 paragraph patient-friendly explanation now:
"""


def get_quiz_generation_prompt(
    topic: str,
    summary: str,
    difficulty: str = "Intermediate",
    previous_questions: Optional[list] = None,
) -> str:
    """
    Constructs prompt to generate a comprehension quiz question strictly from the summary.
    """
    prev_q_text = ""
    if previous_questions:
        prev_q_text = "\nPreviously asked questions (do NOT repeat these concepts):\n" + "\n".join(
            f"- {q}" for q in previous_questions
        )

    difficulty_guidance = {
        "Beginner": "Focus on the primary definition or a key basic symptom explicitly stated in the summary.",
        "Intermediate": "Focus on the relationship between causes, symptoms, or recommended lifestyle management stated in the summary.",
        "Advanced": "Focus on nuanced clinical distinctions, specific management protocols, or specific precautions stated in the summary.",
    }

    guidance = difficulty_guidance.get(difficulty, difficulty_guidance["Intermediate"])

    return f"""You are creating a patient comprehension check for the topic: "{topic}".

CRITICAL INSTRUCTIONS - STRICT GROUNDING REQUIREMENT:
1. Formulate a SINGLE relevant quiz question based EXCLUSIVELY on the provided summary below.
2. The question MUST be 100% answerable using ONLY the information written in the summary.
3. Provide four distinct options labeled A), B), C), and D), where exactly ONE is the correct answer based on the summary.
4. Difficulty level: {difficulty}. {guidance}
{prev_q_text}

PATIENT SUMMARY (YOUR EXCLUSIVE SOURCE OF TRUTH):
---
{summary}
---

Output your response in the following exact format:
Question: <Clear, engaging question text>
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>
Correct Option: <A, B, C, or D>
Summary Reference: <Exact phrase or sentence from the summary that proves the answer>
"""


def get_quiz_evaluation_prompt(
    question: str,
    user_answer: str,
    summary: str,
) -> str:
    """
    Constructs prompt to evaluate the patient's answer strictly against the summary.
    Outputs letter grade, detailed explanation, and citations.
    """
    return f"""You are an educational evaluator assessing a patient's response to a comprehension check.

CRITICAL INSTRUCTIONS - STRICT GROUNDING REQUIREMENT:
1. Grade the patient's answer using ONLY the provided summary as the source of truth.
2. Assign a letter grade according to these criteria:
   - 'A': Completely accurate and demonstrates clear understanding of the fact in the summary.
   - 'B': Mostly accurate, captured the main concept but missed a minor nuance.
   - 'C': Partially correct, but contains notable misunderstanding or confusion.
   - 'D': Mostly incorrect, contradicts the summary.
   - 'F': Completely incorrect or unrelated.
3. Provide an empathetic, constructive explanation justifying why this grade was assigned.
4. Quote the exact sentence or excerpt from the summary as a CITATION to reinforce learning.

PATIENT SUMMARY (SOURCE OF TRUTH):
---
{summary}
---

QUIZ QUESTION:
{question}

PATIENT'S SUBMITTED ANSWER:
"{user_answer}"

Output your assessment in the following exact format:
Grade: <A, B, C, D, or F>
Explanation: <Empathetic, clear justification of the grade and clarification of the medical fact>
Citation: "<Exact quote from the summary supporting the correct answer>"
"""


def get_related_topics_prompt(topic: str, summary: str) -> str:
    """
    Constructs prompt to generate related health topics for patient exploration (standout feature).
    """
    return f"""Based on the medical topic "{topic}" and the patient summary below, recommend 3 to 4 related medical subjects or follow-up health topics that the patient might benefit from exploring next.

SUMMARY:
---
{summary}
---

Format your response as a numbered list of exactly 3-4 topics (topic names only, max 5 words each):
1. <Related Topic 1>
2. <Related Topic 2>
3. <Related Topic 3>
4. <Related Topic 4>
"""
