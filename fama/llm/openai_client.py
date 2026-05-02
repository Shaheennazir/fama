"""
OpenAI LLM client implementation for FAMA.

This module provides the OpenAIClient class for interfacing with
OpenAI's GPT models through the chat completions API.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fama.llm.base import (
    APIConnectionError,
    AuthenticationError,
    BaseLLMClient,
    InvalidRequestError,
    LLMResponse,
    LLMError,
    Message,
    RateLimitError,
    Role,
)

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """
    OpenAI LLM client using the chat completions API.

    Supports GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, and other OpenAI models
    through the official OpenAI Python library.

    Attributes:
        model: The OpenAI model identifier (e.g., "gpt-4", "gpt-3.5-turbo").
        api_key: OpenAI API key for authentication.
        base_url: Optional custom base URL for API requests.
        timeout: Request timeout in seconds.

    Example:
        ```python
        from fama.llm.openai_client import OpenAIClient

        client = OpenAIClient(model="gpt-4", api_key="sk-...")
        response = client.generate(
            messages=[
                Message(role=Role.SYSTEM, content="You are a helpful assistant."),
                Message(role=Role.USER, content="Hello!"),
            ],
            temperature=0.7,
        )
        print(response.content)
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the OpenAI client.

        Args:
            model: OpenAI model identifier (default: "gpt-4").
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env.
            base_url: Optional custom base URL for API requests.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Delay between retries in seconds.
            **kwargs: Additional arguments passed to the OpenAI client.
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)

        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize the OpenAI client
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: pip install openai"
            )

        # Build client kwargs
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout:
            client_kwargs["timeout"] = timeout

        # Add any additional kwargs (e.g., default_headers, organization)
        for key, value in self._kwargs.items():
            if key not in ("max_retries", "retry_delay"):
                client_kwargs[key] = value

        self._client = OpenAI(**client_kwargs)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response using the OpenAI chat completions API.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = very random).
            max_tokens: Maximum tokens to generate.
            stop_sequences: Sequences that stop generation.
            **kwargs: Additional OpenAI-specific arguments.

        Returns:
            LLMResponse containing the generated content and metadata.

        Raises:
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit is exceeded.
            InvalidRequestError: If the request is invalid.
            APIConnectionError: If connection fails.
            LLMError: For other errors.
        """
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if stop_sequences:
            request_kwargs["stop"] = stop_sequences

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in request_kwargs:
                request_kwargs[key] = value

        try:
            # Make the API call
            response = self._client.chat.completions.create(**request_kwargs)

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""

            # Build usage info
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=content,
                raw_response=response,
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
            )

        except Exception as e:
            raise self._map_error(e)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """
        Convert FAMA Message objects to OpenAI message format.

        Args:
            messages: List of FAMA Message objects.

        Returns:
            List of dictionaries in OpenAI message format.
        """
        openai_messages = []
        for msg in messages:
            role = self._map_role(msg.role)
            openai_messages.append({"role": role, "content": msg.content})
        return openai_messages

    def _map_role(self, role: Role) -> str:
        """
        Map FAMA Role to OpenAI role string.

        Args:
            role: FAMA Role enum.

        Returns:
            OpenAI role string.
        """
        mapping = {
            Role.SYSTEM: "system",
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
        }
        return mapping.get(role, "user")

    def _map_error(self, error: Exception) -> LLMError:
        """
        Map OpenAI errors to FAMA LLM errors.

        Args:
            error: The original exception.

        Returns:
            Appropriate LLMError subclass.
        """
        error_str = str(error).lower()

        if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            return AuthenticationError(
                message=f"OpenAI authentication failed: {error}",
                raw_error=error,
            )

        if "rate limit" in error_str or "429" in error_str:
            return RateLimitError(
                message=f"OpenAI rate limit exceeded: {error}",
                raw_error=error,
            )

        if "invalid request" in error_str or "400" in error_str or "validation" in error_str:
            return InvalidRequestError(
                message=f"Invalid OpenAI request: {error}",
                raw_error=error,
            )

        if "connection" in error_str or "timeout" in error_str or "network" in error_str:
            return APIConnectionError(
                message=f"OpenAI API connection failed: {error}",
                raw_error=error,
            )

        return LLMError(
            message=f"OpenAI API error: {error}",
            raw_error=error,
        )

    def generate_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response using system and user prompts.

        Args:
            system_prompt: System instructions.
            user_prompt: User message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments.

        Returns:
            LLMResponse with the generated content.
        """
        # For OpenAI, we can use native system messages
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
