"""Injectable answer providers.

Production uses the OpenAI Responses API (``client.responses.create``) with one
reused client. Tests use a deterministic fake needing no key or network. API
errors become safe application errors that never leak keys, prompts, or traces.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class AnswerGenerationError(Exception):
    """Safe, user-facing generation error (no keys, prompts, or raw traces).

    ``category`` is a safe code (auth/quota/rate_limit/timeout/...) for reporting.
    """

    def __init__(self, message: str, category: str = "other") -> None:
        super().__init__(message)
        self.category = category


@dataclass
class GeneratedAnswer:
    """Raw text plus optional token usage from a provider."""

    text: str
    usage: Optional[dict] = None


class AnswerProvider(ABC):
    """Minimal generation interface used by the tutor service."""

    model_name: str

    @abstractmethod
    def generate(self, instructions: str, user_prompt: str) -> GeneratedAnswer:
        """Generate an answer given developer instructions and user input."""


class OpenAIAnswerProvider(AnswerProvider):
    """OpenAI Responses API provider. Model comes from configuration."""

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise AnswerGenerationError("OpenAI API key is not configured.")
        if not model or not model.strip():
            raise AnswerGenerationError("No chat model is configured.")
        from openai import OpenAI

        self.model_name = model
        # One client per provider instance, reused for every request.
        self._client = OpenAI(api_key=api_key)

    def generate(self, instructions: str, user_prompt: str) -> GeneratedAnswer:
        try:
            response = self._client.responses.create(
                model=self.model_name,
                instructions=instructions,
                input=user_prompt,
            )
        except Exception as exc:  # noqa: BLE001 - mapped to safe errors below
            raise self._safe_error(exc) from exc

        # output_text is the SDK's aggregated text convenience accessor.
        text = (getattr(response, "output_text", "") or "").strip()
        usage = self._safe_usage(response)
        return GeneratedAnswer(text=text, usage=usage)

    @staticmethod
    def _safe_usage(response) -> Optional[dict]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        try:
            return {
                "input_tokens": int(getattr(usage, "input_tokens", 0)),
                "output_tokens": int(getattr(usage, "output_tokens", 0)),
            }
        except Exception:  # usage shape varies; never fail the answer over it
            return None

    @staticmethod
    def _safe_error(exc: Exception) -> AnswerGenerationError:
        """Map SDK exceptions to safe, user-friendly messages."""
        # Import lazily so the module loads even if the SDK layout shifts.
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                NotFoundError,
                RateLimitError,
            )
        except Exception:  # pragma: no cover - defensive
            APIConnectionError = APITimeoutError = AuthenticationError = ()
            BadRequestError = NotFoundError = RateLimitError = ()

        from src.openai_safe import classify_openai_error

        name = type(exc).__name__
        logger.error("OpenAI generation failed: %s", name)  # type only, no content
        category = classify_openai_error(exc)

        if AuthenticationError and isinstance(exc, AuthenticationError):
            msg = "Authentication with OpenAI failed. Check your API key."
        elif RateLimitError and isinstance(exc, RateLimitError):
            msg = "OpenAI rate limit or quota reached. Please try again later."
        elif (APITimeoutError and isinstance(exc, APITimeoutError)) or (
            APIConnectionError and isinstance(exc, APIConnectionError)
        ):
            msg = "Could not reach OpenAI (timeout or connection error). Try again."
        elif NotFoundError and isinstance(exc, NotFoundError):
            msg = "The configured chat model is unavailable to this account."
        elif BadRequestError and isinstance(exc, BadRequestError):
            msg = "The request to OpenAI was rejected. Check the model configuration."
        else:
            msg = "Answer generation failed due to a service error. Please try again."
        return AnswerGenerationError(msg, category=category)


class FakeAnswerProvider(AnswerProvider):
    """Deterministic provider for offline tests.

    Returns a scripted answer (or raises a scripted error) and records the last
    instructions/prompt it received so tests can assert what reached the model.
    """

    def __init__(
        self,
        response_text: str = "This is a grounded answer [S1].",
        usage: Optional[dict] = None,
        model_name: str = "fake-chat",
        raise_error: Optional[Exception] = None,
    ) -> None:
        self.model_name = model_name
        self._response_text = response_text
        self._usage = usage
        self._raise_error = raise_error
        self.call_count = 0
        self.last_instructions: Optional[str] = None
        self.last_user_prompt: Optional[str] = None

    def generate(self, instructions: str, user_prompt: str) -> GeneratedAnswer:
        self.call_count += 1
        self.last_instructions = instructions
        self.last_user_prompt = user_prompt
        if self._raise_error is not None:
            raise self._raise_error
        return GeneratedAnswer(text=self._response_text, usage=self._usage)
