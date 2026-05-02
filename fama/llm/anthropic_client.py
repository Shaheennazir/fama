"""
Anthropic LLM client implementation for FAMA.

This module provides the AnthropicClient class for interfacing with
Anthropic's Claude models through the Claude API.
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


class AnthropicClient(BaseLLMClient):
    """
    Anthropic LLM client using the Claude API.

    Supports Claude 3, Claude 3.5, and Claude 2 models through the official
    Anthropic Python library.

    Attributes:
        model: The Anthropic model identifier (e.g., "claude-3-opus-20240229").
        api_key: Anthropic API key for authentication.
        base_url: Optional custom base URL for API requests.
        timeout: Request timeout in seconds.

    Example:
        ```python
        from fama.llm.anthropic_client import AnthropicClient
        from fama.llm.base import Message, Role

        client = AnthropicClient(
            model="claude-3-opus-20240229",
            api_key="sk-ant-..."
        )
        response = client.generate(
            messages=[
                Message(role=Role.USER, content="Hello!"),
            ],
            temperature=0.7,
        )
        print(response.content)
        ```
    """

    # Default Anthropic models
    DEFAULT_MODELS = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        "claude-2": "claude-2.1",
    }

    def __init__(
        self,
        model: str = "claude-3-5-sonnet",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Anthropic client.

        Args:
            model: Anthropic model identifier (default: "claude-3-5-sonnet").
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env.
            base_url: Optional custom base URL for API requests.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Delay between retries in seconds.
            **kwargs: Additional arguments passed to the Anthropic client.
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)

        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Resolve model name to full versioned ID if needed
        self._model = self.DEFAULT_MODELS.get(model, model)

        # Initialize the Anthropic client
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            )

        # Build client kwargs
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout:
            client_kwargs["timeout"] = timeout

        # Add any additional kwargs
        for key, value in self._kwargs.items():
            if key not in ("max_retries", "retry_delay"):
                client_kwargs[key] = value

        self._client = Anthropic(**client_kwargs)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response using the Anthropic Claude API.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate. If not provided, uses a default.
            stop_sequences: Sequences that stop generation.
            system_prompt: Optional system prompt (for providers that support it separately).
            **kwargs: Additional Anthropic-specific arguments.

        Returns:
            LLMResponse containing the generated content and metadata.

        Raises:
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit is exceeded.
            InvalidRequestError: If the request is invalid.
            APIConnectionError: If connection fails.
            LLMError: For other errors.
        """
        # Determine max_tokens - Anthropic requires this
        if max_tokens is None:
            max_tokens = 4096

        # Extract system message if present
        system_content: Optional[str] = system_prompt
        conversation_messages: list[Message] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Anthropic handles system separately
                if system_content is None:
                    system_content = msg.content
                else:
                    # Append to existing system content
                    system_content += f"\n\n{msg.content}"
            else:
                conversation_messages.append(msg)

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages(conversation_messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_content:
            request_kwargs["system"] = system_content

        if stop_sequences:
            request_kwargs["stop_sequences"] = stop_sequences

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in request_kwargs:
                request_kwargs[key] = value

        try:
            # Make the API call
            response = self._client.messages.create(**request_kwargs)

            # Extract response data
            content = ""
            if response.content:
                # Handle both text and other content blocks
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            # Build usage info
            usage = None
            if hasattr(response, "usage"):
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }

            return LLMResponse(
                content=content.strip(),
                raw_response=response,
                model=response.model,
                usage=usage,
                finish_reason=response.stop_reason if hasattr(response, "stop_reason") else None,
            )

        except Exception as e:
            raise self._map_error(e)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """
        Convert FAMA Message objects to Anthropic message format.

        Args:
            messages: List of FAMA Message objects.

        Returns:
            List of dictionaries in Anthropic message format.
        """
        anthropic_messages = []
        for msg in messages:
            role = self._map_role(msg.role)
            anthropic_messages.append({"role": role, "content": msg.content})
        return anthropic_messages

    def _map_role(self, role: Role) -> str:
        """
        Map FAMA Role to Anthropic role string.

        Args:
            role: FAMA Role enum.

        Returns:
            Anthropic role string.
        """
        mapping = {
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
        }
        # Anthropic doesn't have a system role - we handle it separately
        return mapping.get(role, "user")

    def _map_error(self, error: Exception) -> LLMError:
        """
        Map Anthropic errors to FAMA LLM errors.

        Args:
            error: The original exception.

        Returns:
            Appropriate LLMError subclass.
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        if error_type in ("AuthenticationError", "UnauthorizedError") or "api key" in error_str or "unauthorized" in error_str:
            return AuthenticationError(
                message=f"Anthropic authentication failed: {error}",
                raw_error=error,
            )

        if "rate limit" in error_str or "429" in error_str or "OverloadedError" in error_type:
            return RateLimitError(
                message=f"Anthropic rate limit exceeded: {error}",
                raw_error=error,
            )

        if "invalid request" in error_str or "400" in error_str or "BadRequestError" in error_type:
            return InvalidRequestError(
                message=f"Invalid Anthropic request: {error}",
                raw_error=error,
            )

        if "connection" in error_str or "timeout" in error_str or "network" in error_str or "APIConnectionError" in error_type:
            return APIConnectionError(
                message=f"Anthropic API connection failed: {error}",
                raw_error=error,
            )

        return LLMError(
            message=f"Anthropic API error: {error}",
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
        messages = [
            Message(role=Role.USER, content=user_prompt),
        ]
        return self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs,
        )
