"""Configuration module for Alethia.

Handles environment variable loading and validation without exposing secrets.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

PLACEHOLDER_KEY = "your_groq_api_key_here"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def get_groq_model() -> str:
    """Retrieve the configured Groq model name from environment.

    Defaults to 'openai/gpt-oss-20b'.
    """
    model = os.getenv("GROQ_MODEL", "").strip()
    return model if model else DEFAULT_GROQ_MODEL


def get_groq_api_key() -> str | None:
    """Retrieve the raw GROQ_API_KEY from environment.

    Returns:
        The GROQ_API_KEY string if set and not empty, otherwise None.
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key if key else None


def is_groq_configured() -> bool:
    """Check whether GROQ_API_KEY is set to a non-placeholder value.

    Returns:
        True if GROQ_API_KEY is configured with a valid non-empty,
        non-placeholder value; False otherwise.
    """
    key = get_groq_api_key()
    if not key or key == PLACEHOLDER_KEY:
        return False
    return True


def validate_groq_config() -> Dict[str, Any]:
    """Validate Groq configuration without exposing the actual API key or key fragments.

    Returns:
        Dict containing validation status and a safe status message.
    """
    key = get_groq_api_key()
    if not key:
        return {
            "is_valid": False,
            "message": "GROQ_API_KEY is missing or empty in environment.",
        }
    if key == PLACEHOLDER_KEY:
        return {
            "is_valid": False,
            "message": "GROQ_API_KEY is still set to default placeholder.",
        }

    return {
        "is_valid": True,
        "message": "GROQ_API_KEY is configured.",
    }

