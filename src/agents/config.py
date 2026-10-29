"""Groq LLM Configuration and Factory for the Agentic Investigation Layer."""

import logging
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_AGENT_MODEL = os.getenv("AGENT_MODEL_NAME", "openai/gpt-oss-120b")
FALLBACK_AGENT_MODEL = "openai/gpt-oss-20b"
SUPPORTED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

# Distributed Multi-LLM Role Specialization for Pattern Analysis:
# - qwen/qwen3.8-27b: Cohort comparison, temporal trends & verified error rate dynamics
# - openai/gpt-oss-20b: Evidence-backed structured finding generation & follow-up actions
# - openai/gpt-oss-120b: High-level executive synthesis & leadership brief assembly
PATTERN_COHORT_MODEL = os.getenv("PATTERN_COHORT_MODEL", "qwen/qwen3.8-27b")
PATTERN_FINDINGS_MODEL = os.getenv("PATTERN_FINDINGS_MODEL", "openai/gpt-oss-20b")
PATTERN_EXECUTIVE_MODEL = os.getenv("PATTERN_EXECUTIVE_MODEL", "openai/gpt-oss-120b")
DISTRIBUTED_PIPELINE_OPTION = "Distributed Multi-LLM (120b + 20b + 27b)"
PATTERN_MODELS = [
    DISTRIBUTED_PIPELINE_OPTION,
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]


def get_groq_api_key() -> str:
    """Retrieve and validate the Groq API key from environment."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add GROQ_API_KEY to your .env file or environment variables."
        )
    return key


def get_groq_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 2,
    timeout: float = 30.0,
) -> ChatGroq:
    """Initialize and return a configured ChatGroq instance.

    Defaults to openai/gpt-oss-120b with fallback to openai/gpt-oss-20b.
    """
    selected_model = model_name or DEFAULT_AGENT_MODEL
    if selected_model == DISTRIBUTED_PIPELINE_OPTION or selected_model not in SUPPORTED_MODELS:
        selected_model = DEFAULT_AGENT_MODEL

    api_key = get_groq_api_key()

    logger.info("Initializing ChatGroq with model='%s', temperature=%.2f", selected_model, temperature)
    return ChatGroq(
        model=selected_model,
        groq_api_key=api_key,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
    )

