"""
Abstract base class for LLM clients in FAMA.

This module defines the base interface that all LLM clients must implement
to ensure compatibility with the FAMA error analysis and agent systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class Role(Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """
    A message in a conversation.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The content of the message.
    """

    role: Role
    content: str


@dataclass
class LLMResponse:
    """
    Response from an LLM client.

    Attributes:
        content: The text content of the response.
        raw_response: The raw response from the LLM provider (provider-specific).
        model: The model that generated the response.
        usage: Token usage information (if available).
        finish_reason: Why the generation stopped (if available).
    """

    content: str
    raw_response: Optional[Any] = None
    model: Optional[str] = None
    usage: Optional[dict[str, int]] = None
    finish_reason: Optional[str] = None


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All LLM clients used in FAMA must inherit from this class and implement
    its abstract methods. This ensures compatibility with the error analysis
    and agent systems.

    Example:
        ```python
        class MyCustomClient(BaseLLMClient):
            def generate(
                self,
                messages: list[Message],
                **kwargs
            ) -> LLMResponse:
                # Implementation here
                pass
        ```
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            model: The model identifier (e.g., "gpt-4", "claude-3-opus").
            api_key: API key for authentication (if required).
            base_url: Base URL for the API endpoint (if using a proxy).
            timeout: Request timeout in seconds.
            **kwargs: Additional provider-specific arguments.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.
            stop_sequences: Sequences that stop generation.
            **kwargs: Additional provider-specific arguments.

        Returns:
            LLMResponse containing the generated content and metadata.

        Raises:
            LLMError: If the generation fails.
        """
        pass

    def generate_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response using a system prompt and user prompt.

        This is a convenience method that constructs messages from
        system and user prompts. It splits a system prompt if needed
        to work with providers that don't support system messages.

        Args:
            system_prompt: The system prompt/instructions.
            user_prompt: The user message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments passed to generate().

        Returns:
            LLMResponse containing the generated content.
        """
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_prompt),
        ]
        return self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the name of the LLM provider.

        Returns:
            Provider name (e.g., "openai", "anthropic").
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the client."""
        return f"{self.__class__.__name__}(model={self.model!r})"


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        raw_error: Optional[Any] = None,
    ) -> None:
        """
        Initialize the LLM error.

        Args:
            message: Human-readable error message.
            code: Error code (provider-specific).
            raw_error: The original exception or error object.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.raw_error = raw_error


class AuthenticationError(LLMError):
    """Raised when authentication fails."""

    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""

    pass


class InvalidRequestError(LLMError):
    """Raised when the request is invalid."""

    pass


class APIConnectionError(LLMError):
    """Raised when connection to the API fails."""

    pass
