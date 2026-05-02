"""
Base tool classes for FAMA.

Provides the base Tool class and ToolResult that all FAMA tools must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Attributes:
        success: Whether the tool execution succeeded.
        output: The output from the tool (usually a string).
        error: Error message if execution failed.
        metadata: Additional metadata about the execution.
    """

    success: bool
    output: str = ""
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        """Create a ToolResult from a dictionary."""
        return cls(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def success_result(cls, output: str, metadata: Optional[dict[str, Any]] = None) -> ToolResult:
        """Create a successful result."""
        return cls(success=True, output=output, metadata=metadata or {})

    @classmethod
    def error_result(cls, error: str, metadata: Optional[dict[str, Any]] = None) -> ToolResult:
        """Create an error result."""
        return cls(success=False, error=error, metadata=metadata or {})


class Tool(ABC):
    """
    Abstract base class for FAMA tools.

    All tools used within the FAMA framework must inherit from this class
    and implement the execute method.

    Attributes:
        name: Unique name for the tool.
        description: Human-readable description of the tool.
        parameters: JSON schema for tool parameters.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a tool.

        Args:
            name: Unique name for the tool.
            description: Human-readable description.
            parameters: JSON schema for tool parameters.
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            ToolResult containing the execution outcome.
        """

    def validate_parameters(self, **kwargs: Any) -> tuple[bool, Optional[str]]:
        """
        Validate the provided parameters against the schema.

        Args:
            **kwargs: Parameters to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        required = self.parameters.get("required", [])
        for req_param in required:
            if req_param not in kwargs:
                return False, f"Missing required parameter: {req_param}"

        properties = self.parameters.get("properties", {})
        for param_name, param_value in kwargs.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type == "string" and not isinstance(param_value, str):
                    return False, f"Parameter {param_name} must be a string"
                elif expected_type == "number" and not isinstance(param_value, (int, float)):
                    return False, f"Parameter {param_name} must be a number"
                elif expected_type == "boolean" and not isinstance(param_value, bool):
                    return False, f"Parameter {param_name} must be a boolean"
                elif expected_type == "array" and not isinstance(param_value, list):
                    return False, f"Parameter {param_name} must be an array"

        return True, None

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, description={self.description!r})"
