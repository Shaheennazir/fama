"""
Base class for specialized agents in FAMA.

All specialized agents inherit from SpecializedAgent ABC which defines
the interface for error analysis and context generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fama.core.types import (
    AgentType,
    ErrorAnalysisResult,
    ErrorCategory,
    Trajectory,
)


class SpecializedAgent(ABC):
    """
    Abstract base class for specialized FAMA agents.

    Each specialized agent implements two main methods:
    1. analyze() - Analyzes a trajectory for a specific error category (Stage 1)
    2. generate_context() - Generates context for injection (Stage 2)

    Attributes:
        agent_type: The type of this agent.
        llm_client: The LLM client used for analysis.
    """

    def __init__(
        self,
        llm_client: Any,
        agent_type: AgentType,
    ) -> None:
        """
        Initialize a specialized agent.

        Args:
            llm_client: LLM client for generating analysis and context.
            agent_type: The type of this specialized agent.
        """
        self.llm_client = llm_client
        self.agent_type = agent_type

    @abstractmethod
    def analyze(
        self,
        trajectory: Trajectory,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """
        Analyze a trajectory for a specific error category.

        This method is called during Stage 1 analysis to determine if
        this agent's specialty area is a contributing factor to failure.

        Args:
            trajectory: The execution trajectory to analyze.
            error_category: The error category to check for.

        Returns:
            ErrorAnalysisResult indicating whether this error was detected
            and the reasoning behind the analysis.
        """

    @abstractmethod
    def generate_context(
        self,
        trajectory: Trajectory,
    ) -> str:
        """
        Generate context to inject when this agent is selected.

        This method is called during Stage 2 to provide the agent's
        insights and guidance for the re-execution.

        Args:
            trajectory: The original execution trajectory.

        Returns:
            String containing context/guidance from this agent for re-execution.
        """

    def get_error_categories_handled(self) -> list[ErrorCategory]:
        """
        Get list of error categories this agent can help address.

        Returns:
            List of ErrorCategory values this agent handles.
        """
        from fama.core.types import ERROR_TO_AGENT_MAPPING

        return [
            error
            for error, agents in ERROR_TO_AGENT_MAPPING.items()
            if self.agent_type in agents
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_type={self.agent_type.value})"
