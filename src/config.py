"""
Configuration module for HealthBot.
Handles loading environment variables from config.env, API keys validation,
and initialization of LLM and search tools.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


def load_env_config(env_file: str = "config.env") -> bool:
    """
    Loads configuration from specified env file.
    Tries specified file, falls back to .env or process environment.
    """
    # Look in current working directory or same directory as this module
    if os.path.exists(env_file):
        load_dotenv(env_file, override=False)
        return True
    
    # Try searching parent or repo directory
    alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), env_file)
    if os.path.exists(alt_path):
        load_dotenv(alt_path, override=False)
        return True

    # Fallback to standard .env
    load_dotenv()
    return False


def validate_environment(raise_error: bool = True) -> bool:
    """
    Validates required API keys per project specifications:
    - OPENAI_API_KEY
    - TAVILY_API_KEY
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    is_valid = bool(openai_key and tavily_key)
    
    if not is_valid and raise_error:
        missing = []
        if not openai_key:
            missing.append("OPENAI_API_KEY")
        if not tavily_key:
            missing.append("TAVILY_API_KEY")
        raise ValueError(
            f"Missing required environment variables in config.env: {', '.join(missing)}. "
            "Please configure config.env with valid OpenAI and Tavily API keys."
        )
    return is_valid


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """
    Returns an initialized ChatOpenAI instance.
    Defaults to gpt-4o-mini for efficient, high-accuracy clinical explanations.
    """
    resolved_model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")

    return ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        api_key=resolved_key,
    )


def get_tavily_tool(
    max_results: Optional[int] = None,
    api_key: Optional[str] = None,
) -> TavilySearchResults:
    """
    Returns an initialized TavilySearchResults tool.
    Configured with max_results (default 5) for comprehensive medical retrieval.
    """
    results_count = max_results or int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    resolved_key = api_key or os.getenv("TAVILY_API_KEY")

    if resolved_key:
        os.environ["TAVILY_API_KEY"] = resolved_key

    return TavilySearchResults(max_results=results_count)

