"""
Tool registry for FAMA.

Provides a singleton registry for managing tools and a decorator for
registering new tools.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from fama.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Singleton registry for FAMA tools.

    Provides centralized management of all available tools with
    registration and lookup capabilities.
    """

    _instance: Optional[ToolRegistry] = None
    _tools: dict[str, Tool]

    def __new__(cls) -> ToolRegistry:
        """Get singleton instance of ToolRegistry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: Tool) -> None:
        """
        Register a tool with the registry.

        Args:
            tool: The tool to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool with name {tool.name} is already registered")
        logger.debug(f"Registering tool: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool from the registry.

        Args:
            name: Name of the tool to unregister.

        Returns:
            True if the tool was unregistered, False if it wasn't found.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            The tool if found, None otherwise.
        """
        return self._tools.get(name)

    def get_all(self) -> dict[str, Tool]:
        """
        Get all registered tools.

        Returns:
            Dictionary mapping tool names to Tool instances.
        """
        return dict(self._tools)

    def list_names(self) -> list[str]:
        """
        Get list of all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        logger.debug("Cleared all tools from registry")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


# Global registry instance
_registry = ToolRegistry()


def register_tool(tool_class: Callable[[], Tool]) -> Callable[[], Tool]:
    """
    Decorator to register a tool class.

    The decorated class must be a subclass of Tool and will be
    instantiated and registered automatically.

    Example:
        @register_tool
        class MyTool(Tool):
            def __init__(self):
                super().__init__(name="my_tool", description="My tool")

            def execute(self, **kwargs) -> ToolResult:
                ...

    Returns:
        The decorated class unchanged.
    """
    tool = tool_class()
    _registry.register(tool)
    return tool_class


def get_tool(name: str) -> Optional[Tool]:
    """
    Get a registered tool by name.

    Args:
        name: Name of the tool to retrieve.

    Returns:
        The tool if found, None otherwise.
    """
    return _registry.get(name)


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.

    Returns:
        The ToolRegistry singleton instance.
    """
    return _registry
