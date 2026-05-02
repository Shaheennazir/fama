"""
Base classes for benchmark implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A task to be executed by an agent."""
    task_id: str
    domain: str
    user_request: str
    domain_policy: str
    available_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Task({self.task_id}, {self.domain}: {self.user_request[:50]}...)"


class BaseBenchmark(ABC):
    """Abstract base class for benchmark implementations."""

    @abstractmethod
    def get_tasks(self, domain: str | None = None) -> list[Task]:
        """Get tasks for a specific domain or all domains."""
        ...

    @abstractmethod
    def evaluate(self, trajectory: list[dict], task: Task) -> bool:
        """
        Evaluate whether a trajectory successfully completes the task.
        
        Args:
            trajectory: List of turns from the agent execution
            task: The task that was executed
            
        Returns:
            True if the task was completed successfully
        """
        ...

    @abstractmethod
    def get_domain_policy(self, domain: str) -> str:
        """Get the domain policy for a specific domain."""
        ...
