"""
Toolkit for FAMA.

Provides a collection of related tools that can be used together.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fama.tools.base import Tool, ToolResult
from fama.tools.registry import get_registry

logger = logging.getLogger(__name__)


class Toolkit:
    """
    A collection of related tools.

    Toolkits group related tools together, making it easy to load
    and manage tool collections for different domains or purposes.

    Attributes:
        name: Name of the toolkit.
        description: Description of the toolkit's purpose.
        tools: Dictionary of tools in this toolkit.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        tools: Optional[list[Tool]] = None,
    ) -> None:
        """
        Initialize a toolkit.

        Args:
            name: Name of the toolkit.
            description: Description of the toolkit's purpose.
            tools: Initial list of tools to include.
        """
        self.name = name
        self.description = description
        self.tools: dict[str, Tool] = {}

        if tools:
            for tool in tools:
                self.add_tool(tool)

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the toolkit.

        Args:
            tool: Tool to add.

        Raises:
            ValueError: If a tool with the same name is already in the toolkit.
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name} already exists in toolkit {self.name}")
        self.tools[tool.name] = tool
        logger.debug(f"Added tool {tool.name} to toolkit {self.name}")

    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool from the toolkit.

        Args:
            name: Name of the tool to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self.tools:
            del self.tools[name]
            logger.debug(f"Removed tool {name} from toolkit {self.name}")
            return True
        return False

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool from the toolkit by name.

        Args:
            name: Name of the tool.

        Returns:
            The tool if found, None otherwise.
        """
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """
        List all tool names in this toolkit.

        Returns:
            List of tool names.
        """
        return list(self.tools.keys())

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Execute a tool from the toolkit.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Arguments to pass to the tool.

        Returns:
            ToolResult from the execution.

        Raises:
            KeyError: If the tool is not found in the toolkit.
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            return ToolResult.error_result(f"Tool {tool_name} not found in toolkit {self.name}")

        try:
            # Validate parameters
            is_valid, error_msg = tool.validate_parameters(**kwargs)
            if not is_valid:
                return ToolResult.error_result(error_msg or "Invalid parameters")

            return tool.execute(**kwargs)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolResult.error_result(f"Execution error: {str(e)}")

    def __len__(self) -> int:
        """Get number of tools in the toolkit."""
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is in the toolkit."""
        return name in self.tools

    def __repr__(self) -> str:
        return f"Toolkit(name={self.name!r}, tool_count={len(self.tools)})"


class ToolkitRegistry:
    """
    Registry for managing multiple toolkits.

    Provides centralized management of toolkit collections.
    """

    _instance: Optional[ToolkitRegistry] = None

    def __new__(cls) -> ToolkitRegistry:
        """Get singleton instance of ToolkitRegistry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._toolkits = {}
        return cls._instance

    def register(self, toolkit: Toolkit) -> None:
        """
        Register a toolkit.

        Args:
            toolkit: Toolkit to register.
        """
        self._toolkits[toolkit.name] = toolkit
        logger.debug(f"Registered toolkit: {toolkit.name}")

    def get(self, name: str) -> Optional[Toolkit]:
        """
        Get a toolkit by name.

        Args:
            name: Name of the toolkit.

        Returns:
            The toolkit if found, None otherwise.
        """
        return self._toolkits.get(name)

    def list_toolkits(self) -> list[str]:
        """List all registered toolkit names."""
        return list(self._toolkits.keys())

    def clear(self) -> None:
        """Clear all toolkits."""
        self._toolkits.clear()
