"""
LLM client implementations for FAMA.

This package provides LLM client classes for interfacing with various
language model providers. All clients implement the BaseLLMClient interface
for compatibility with the FAMA error analysis and agent systems.

Available Clients:
    - OpenAIClient: For OpenAI GPT models (GPT-4, GPT-3.5-Turbo, etc.)
    - AnthropicClient: For Anthropic Claude models (Claude 3, Claude 2, etc.)

Example:
    ```python
    from fama.llm import OpenAIClient, AnthropicClient

    # Using OpenAI
    openai_client = OpenAIClient(model="gpt-4", api_key="sk-...")

    # Using Anthropic
    anthropic_client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-...")

    # Generate responses
    from fama.llm.base import Message, Role

    response = openai_client.generate(
        messages=[
            Message(role=Role.USER, content="Hello!"),
        ],
        temperature=0.7,
    )
    print(response.content)
    ```

Base Classes:
    BaseLLMClient: Abstract base class for all LLM clients.
    Message: Data class for conversation messages.
    LLMResponse: Data class for LLM responses.

Exceptions:
    LLMError: Base exception for LLM-related errors.
    AuthenticationError: Raised when authentication fails.
    RateLimitError: Raised when rate limit is exceeded.
    InvalidRequestError: Raised when the request is invalid.
    APIConnectionError: Raised when connection to the API fails.
"""

from fama.llm.base import (
    BaseLLMClient,
    LLMError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    APIConnectionError,
    LLMResponse,
    Message,
    Role,
)
from fama.llm.openai_client import OpenAIClient
from fama.llm.anthropic_client import AnthropicClient

__all__ = [
    # Base classes
    "BaseLLMClient",
    "LLMResponse",
    "Message",
    "Role",
    # Exceptions
    "LLMError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "APIConnectionError",
    # Clients
    "OpenAIClient",
    "AnthropicClient",
]
