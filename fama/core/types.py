"""
Type definitions for FAMA framework.

This module contains all the core type definitions including enums for error
categories and agent types, as well as dataclasses for trajectories, task
results, and analysis outputs.
"""

from __future__ import annotations

import sys
from datetime import datetime
from enum import Enum
from typing import Any, Optional

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class ErrorCategory(Enum):
    """
    Error categories for failure analysis.

    Each category represents a distinct failure mode that agents may exhibit
    when executing tasks.
    """

    DCV = "domain_constraint_violation"
    WRCO = "wrong_retrieval_complex_outputs"
    CM = "contextual_misinterpretation"
    IFU = "incomplete_fulfillment"

    @property
    def name_short(self) -> str:
        """Return short name for the error category."""
        return self.value.split("_")[0].upper()

    def __str__(self) -> str:
        return self.value


class AgentType(Enum):
    """
    Agent types in the FAMA agent pool.

    Each agent specializes in addressing specific error categories and
    provides unique capabilities for task execution.
    """

    DCE = "domain_constraints_extractor"
    TSA = "tool_suggestion_agent"
    TOR = "tool_output_reformatter"
    PLANNER = "planner"
    VERIFIER = "verifier"
    MEMORY = "memory"

    def __str__(self) -> str:
        return self.value


# Agent pool metadata with descriptions
AGENT_POOL: dict[AgentType, str] = {
    AgentType.DCE: (
        "Domain Constraints Extractor - Extracts and enforces domain-specific "
        "constraints and policies to prevent policy violations"
    ),
    AgentType.TSA: (
        "Tool Suggestion Agent - Suggests appropriate tools for complex tasks "
        "based on task requirements and available tool capabilities"
    ),
    AgentType.TOR: (
        "Tool Output Reformatter - Reformats and clarifies complex tool outputs "
        "with nested structures, lists, and multiple items for better parsing"
    ),
    AgentType.PLANNER: (
        "Planner Agent - Creates and refines execution plans, breaking down "
        "complex tasks into manageable steps"
    ),
    AgentType.VERIFIER: (
        "Verifier Agent - Validates task completion and correctness, "
        "identifying potential issues before final response"
    ),
    AgentType.MEMORY: (
        "Memory Agent - Maintains context and history for long-running tasks, "
        "ensuring consistency across multiple turns"
    ),
}


# Mapping of error categories to the agents that can address them
ERROR_TO_AGENT_MAPPING: dict[ErrorCategory, list[AgentType]] = {
    ErrorCategory.DCV: [AgentType.DCE, AgentType.VERIFIER],
    ErrorCategory.WRCO: [AgentType.TOR, AgentType.TSA],
    ErrorCategory.CM: [AgentType.PLANNER, AgentType.TSA],
    ErrorCategory.IFU: [AgentType.PLANNER, AgentType.VERIFIER, AgentType.MEMORY],
}


# Error categories metadata with definitions and potential causes
ERROR_CATEGORIES_METADATA: dict[ErrorCategory, dict[str, Any]] = {
    ErrorCategory.DCV: {
        "name": "Domain Constraint Violation",
        "definition": (
            "The agent performs an action that is forbidden by domain policy "
            "or skips a step that is required by the domain rules."
        ),
        "causes": [
            "Agent lacks awareness of domain-specific constraints",
            "Agent misunderstands policy boundaries",
            "Agent prioritizes task completion over policy compliance",
            "Insufficient guidance on allowed/disallowed actions",
        ],
    },
    ErrorCategory.WRCO: {
        "name": "Wrong Retrieval from Complex Tool Outputs",
        "definition": (
            "The agent fails to correctly extract, parse, or utilize information "
            "from tool outputs that contain nested structures, lists, or multiple items."
        ),
        "causes": [
            "Agent misses relevant items in lists",
            "Agent fails to navigate nested data structures",
            "Agent extracts wrong fields from complex outputs",
            "Agent ignores multiple valid results",
        ],
    },
    ErrorCategory.CM: {
        "name": "Contextual Misinterpretation",
        "definition": (
            "The agent understands the literal meaning of words but fails to "
            "grasp the actual user intent or contextual nuance."
        ),
        "causes": [
            "Agent takes instructions too literally",
            "Agent ignores implicit constraints or preferences",
            "Agent fails to infer missing information from context",
            "Agent doesn't ask for clarification when needed",
        ],
    },
    ErrorCategory.IFU: {
        "name": "Incomplete Fulfillment / Early Stopping",
        "definition": (
            "The agent stops execution when encountering difficulty instead of "
            "trying alternative approaches or persisting to find a solution."
        ),
        "causes": [
            "Agent gives up too early when first attempt fails",
            "Agent doesn't explore alternative strategies",
            "Agent lacks confidence to try harder paths",
            "Agent mistakes difficulty for impossibility",
        ],
    },
}


class Turn:
    """
    Represents a single turn in a conversation trajectory.

    A turn consists of a user message, assistant response, any tool calls made,
    and the results of those tool calls.
    """

    def __init__(
        self,
        user_message: str,
        assistant_message: str,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        tool_results: Optional[list[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Initialize a turn.

        Args:
            user_message: The user's input message.
            assistant_message: The assistant's response.
            tool_calls: List of tool calls made in this turn.
            tool_results: List of tool results received.
            timestamp: When this turn occurred.
        """
        self.user_message = user_message
        self.assistant_message = assistant_message
        self.tool_calls = tool_calls or []
        self.tool_results = tool_results or []
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert turn to dictionary representation."""
        return {
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a Turn from a dictionary representation."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            user_message=data["user_message"],
            assistant_message=data["assistant_message"],
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", []),
            timestamp=timestamp,
        )


# Type alias for trajectory
Trajectory = list[Turn]


class TaskResult:
    """
    Result of executing a task.

    Contains the trajectory of turns, success status, detected error categories,
    and any additional metadata.
    """

    def __init__(
        self,
        task_id: str,
        trajectory: Trajectory,
        success: bool,
        error_categories: Optional[list[ErrorCategory]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a task result.

        Args:
            task_id: Unique identifier for the task.
            trajectory: List of turns representing the execution.
            success: Whether the task was completed successfully.
            error_categories: List of detected error categories.
            metadata: Additional metadata about the execution.
        """
        self.task_id = task_id
        self.trajectory = trajectory
        self.success = success
        self.error_categories = error_categories or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert task result to dictionary representation."""
        return {
            "task_id": self.task_id,
            "trajectory": [turn.to_dict() for turn in self.trajectory],
            "success": self.success,
            "error_categories": [cat.value for cat in self.error_categories],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a TaskResult from a dictionary representation."""
        return cls(
            task_id=data["task_id"],
            trajectory=[Turn.from_dict(t) for t in data["trajectory"]],
            success=data["success"],
            error_categories=[
                ErrorCategory(cat) for cat in data.get("error_categories", [])
            ],
            metadata=data.get("metadata", {}),
        )


class ErrorAnalysisResult:
    """
    Result of analyzing a trajectory for a specific error category.

    Contains information about whether the error was detected and the
    reasoning behind the analysis.
    """

    def __init__(
        self,
        error_category: ErrorCategory,
        detected: bool,
        rationale: str,
    ) -> None:
        """
        Initialize an error analysis result.

        Args:
            error_category: The category of error being analyzed.
            detected: Whether the error was detected in the trajectory.
            rationale: Explanation of why the error was or wasn't detected.
        """
        self.error_category = error_category
        self.detected = detected
        self.rationale = rationale

    def to_dict(self) -> dict[str, Any]:
        """Convert error analysis result to dictionary representation."""
        return {
            "error_category": self.error_category.value,
            "detected": self.detected,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create an ErrorAnalysisResult from a dictionary representation."""
        return cls(
            error_category=ErrorCategory(data["error_category"]),
            detected=data["detected"],
            rationale=data["rationale"],
        )


class OrchestratorResult:
    """
    Result of the orchestration process.

    Identifies the primary error(s) that caused task failure based on
    the analysis results from all error category analyzers.
    """

    def __init__(
        self,
        primary_errors: list[ErrorCategory],
        reasoning: str,
    ) -> None:
        """
        Initialize an orchestrator result.

        Args:
            primary_errors: List of error categories identified as primary causes.
            reasoning: Explanation of why these errors were selected.
        """
        self.primary_errors = primary_errors
        self.reasoning = reasoning

    def to_dict(self) -> dict[str, Any]:
        """Convert orchestrator result to dictionary representation."""
        return {
            "primary_errors": [cat.value for cat in self.primary_errors],
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create an OrchestratorResult from a dictionary representation."""
        return cls(
            primary_errors=[ErrorCategory(cat) for cat in data["primary_errors"]],
            reasoning=data["reasoning"],
        )


class MitigationResult:
    """
    Result of the mitigation process.

    Contains the minimal subset of agents selected to address the
    identified errors.
    """

    def __init__(
        self,
        selected_agents: list[AgentType],
        reasoning: str,
    ) -> None:
        """
        Initialize a mitigation result.

        Args:
            selected_agents: Minimal subset of agents selected for mitigation.
            reasoning: Explanation of why these agents were selected.
        """
        self.selected_agents = selected_agents
        self.reasoning = reasoning

    def to_dict(self) -> dict[str, Any]:
        """Convert mitigation result to dictionary representation."""
        return {
            "selected_agents": [agent.value for agent in self.selected_agents],
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a MitigationResult from a dictionary representation."""
        return cls(
            selected_agents=[AgentType(agent) for agent in data["selected_agents"]],
            reasoning=data["reasoning"],
        )
