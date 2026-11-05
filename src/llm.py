"""Groq LLM client wrapper for Alethia.

Handles secure client initialization, model configuration (llama-3.1-8b-instant),
reusable inference calls, error classification, and strict secret protection.
"""

import time
from typing import Dict, List, Optional
import groq
from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from src.config import get_groq_api_key, is_groq_configured, get_groq_model, DEFAULT_GROQ_MODEL

DEFAULT_MODEL = DEFAULT_GROQ_MODEL


class LLMError(Exception):
    """Base exception for LLM inference failures."""
    pass


class ConfigurationError(LLMError):
    """Raised when the LLM client is misconfigured or missing credentials."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM service exceeds API rate limits."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM call exceeds network timeouts."""
    pass


def get_groq_client() -> Groq:
    """Initialize and return an authenticated Groq client.

    Raises:
        ConfigurationError: If GROQ_API_KEY is not set or contains the default placeholder.
    """
    if not is_groq_configured():
        raise ConfigurationError(
            "GROQ_API_KEY is not configured. Please supply a valid Groq API key in your .env file."
        )

    api_key = get_groq_api_key()
    return Groq(api_key=api_key)


def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    model: Optional[str] = None,
    client: Optional[Groq] = None,
    max_rate_limit_retries: int = 3,
) -> str:
    """Execute a completion request to Groq with automatic rate-limit backoff retry.

    Args:
        messages: List of message dictionaries containing 'role' and 'content'.
        temperature: Sampling temperature for generation (e.g. 0.0 for deterministic).
        max_tokens: Maximum tokens in response completion.
        model: Model name. Configurable via GROQ_MODEL environment variable.
        client: Optional pre-initialized Groq client.
        max_rate_limit_retries: Number of retries on 429 rate limit before failing.

    Returns:
        Generated response text string.

    Raises:
        ConfigurationError: If API credentials are missing.
        LLMRateLimitError: If rate limit is exceeded after retries.
        LLMTimeoutError: If request times out.
        LLMError: If any other upstream API or network error occurs.
    """
    if not messages:
        raise ValueError("Messages list cannot be empty.")

    active_client = client or get_groq_client()
    selected_model = model or get_groq_model()

    for attempt in range(max_rate_limit_retries + 1):
        try:
            response = active_client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""

        except AuthenticationError as e:
            raise ConfigurationError("Groq authentication failed. Please check your API key.") from e
        except RateLimitError as e:
            if attempt < max_rate_limit_retries:
                # Exponential backoff: 2.0s, 4.0s, 8.0s
                sleep_seconds = 2.0 * (2 ** attempt)
                time.sleep(sleep_seconds)
                continue
            raise LLMRateLimitError(f"Groq API rate limit exceeded: {e.message}") from e
        except APITimeoutError as e:
            raise LLMTimeoutError("Groq API request timed out.") from e
        except APIConnectionError as e:
            raise LLMError("Failed to connect to Groq API. Please verify network connectivity.") from e
        except APIError as e:
            status = getattr(e, "status_code", getattr(e, "code", "unknown"))
            msg = getattr(e, "message", str(e))
            raise LLMError(f"Groq API error (status {status}): {msg}") from e
        except Exception as e:
            raise LLMError(f"Unexpected error during LLM invocation: {e}") from e

    raise LLMRateLimitError("Groq API rate limit exceeded.")

