"""
Orchestrator agent for FAMA.

The Orchestrator identifies the primary error(s) that caused task failure
by analyzing the concatenated outputs from all |E|=4 error-analysis agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fama.core.types import (
    ErrorCategory,
    ERROR_CATEGORIES_METADATA,
    ErrorAnalysisResult,
    OrchestratorResult,
    Trajectory,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator agent for failure analysis.

    The Orchestrator receives the concatenated outputs from all error-category
    analyzers and determines which error(s) are the primary cause of failure.

    It uses an LLM to reason about the error analysis results and trajectory
    to produce a structured output identifying primary errors.
    """

    SYSTEM_PROMPT = """You are the FAMA Orchestrator, a meta-agent responsible for identifying
the primary error(s) that caused a task failure.

Your role is to analyze the outputs from |E|=4 independent error-analysis agents,
each analyzing the same failed trajectory for a specific error category:
- DCV (Domain Constraint Violation): Agent violates domain policies or skips required steps
- WRCO (Wrong Retrieval from Complex Outputs): Agent fails to parse complex tool outputs
- CM (Contextual Misinterpretation): Agent understands words but misses actual intent
- IFU (Incomplete Fulfillment): Agent stops early instead of trying alternatives

Given the analysis results from each category, you must identify which error(s)
are the PRIMARY cause of the failure. Consider:
1. Which errors were explicitly detected?
2. Which errors best explain the failure behavior?
3. Sometimes multiple errors contribute - identify the most likely root cause(s)

Respond with a JSON object containing:
- "primary_errors": List of error categories that are the primary cause (array of strings from: DCV, WRCO, CM, IFU)
- "reasoning": Detailed explanation of why these errors were selected as primary

Be precise and base your reasoning on the actual analysis results provided."""

    USER_PROMPT_TEMPLATE = """Analyze the following task failure and identify the primary error(s).

## Task Trajectory
{trajectory_text}

## Error Analysis Results
{analysis_results_text}

## Your Task
Based on the above information, identify the PRIMARY error(s) that caused this task failure.
Consider which errors were detected and which best explain the observed behavior.

Respond with a JSON object:
{{
    "primary_errors": ["DCV", "WRCO", "CM", "IFU"],  // List of primary error categories
    "reasoning": "Detailed explanation of why these errors are the primary cause"
}}"""

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the Orchestrator.

        Args:
            llm_client: LLM client for generating analysis.
        """
        self.llm_client = llm_client

    def orchestrate(
        self,
        error_results: list[ErrorAnalysisResult],
        trajectory: Trajectory,
    ) -> OrchestratorResult:
        """
        Orchestrate error analysis to identify primary failure causes.

        Analyzes the concatenated outputs from all error-category analyzers
        and determines which error(s) are the primary cause of the failure.

        Args:
            error_results: List of ErrorAnalysisResults, one per error category.
            trajectory: The trajectory that was analyzed.

        Returns:
            OrchestratorResult identifying primary errors and reasoning.
        """
        # Build trajectory text
        trajectory_text = self._format_trajectory(trajectory)

        # Build analysis results text
        analysis_text = self._format_analysis_results(error_results)

        # Build prompts
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            trajectory_text=trajectory_text,
            analysis_results_text=analysis_text,
        )

        # Generate response
        try:
            response = self.llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
            )

            # Parse response
            result = self._parse_response(response)
            logger.debug(f"Orchestrator result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            # Fallback: use detected errors as primary
            detected = [
                er.error_category
                for er in error_results
                if er.detected
            ]
            return OrchestratorResult(
                primary_errors=detected if detected else list(ErrorCategory),
                reasoning=f"Fallback due to error: {str(e)}. Selected all detected errors.",
            )

    def _format_trajectory(self, trajectory: Trajectory) -> str:
        """
        Format a trajectory for inclusion in the prompt.

        Args:
            trajectory: The trajectory to format.

        Returns:
            Formatted trajectory string.
        """
        if not trajectory:
            return "No trajectory available."

        lines = []
        for i, turn in enumerate(trajectory):
            lines.append(f"### Turn {i + 1}")
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.assistant_message}")
            if turn.tool_calls:
                lines.append(f"Tool Calls: {json.dumps(turn.tool_calls)}")
            if turn.tool_results:
                lines.append(f"Tool Results: {json.dumps(turn.tool_results)}")
            lines.append("")

        return "\n".join(lines)

    def _format_analysis_results(
        self,
        error_results: list[ErrorAnalysisResult],
    ) -> str:
        """
        Format error analysis results for inclusion in the prompt.

        Args:
            error_results: List of ErrorAnalysisResults.

        Returns:
            Formatted analysis results string.
        """
        lines = []
        for result in error_results:
            cat = result.error_category
            metadata = ERROR_CATEGORIES_METADATA.get(cat, {})

            lines.append(f"### {cat.value} ({cat.name_short})")
            lines.append(f"**Name**: {metadata.get('name', 'Unknown')}")
            lines.append(f"**Definition**: {metadata.get('definition', 'N/A')}")
            lines.append(f"**Detected**: {result.detected}")
            lines.append(f"**Rationale**: {result.rationale}")
            lines.append("")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> OrchestratorResult:
        """
        Parse the LLM response to extract orchestrator result.

        Args:
            response: The LLM response string.

        Returns:
            OrchestratorResult parsed from the response.
        """
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            primary_errors = []
            for err_str in data.get("primary_errors", []):
                try:
                    # Handle both full names and short names
                    error_cat = ErrorCategory(err_str.upper())
                    primary_errors.append(error_cat)
                except ValueError:
                    # Try matching by partial string
                    for ec in ErrorCategory:
                        if ec.value.startswith(err_str.lower()) or ec.name.lower() == err_str.lower():
                            primary_errors.append(ec)
                            break

            if not primary_errors:
                # Fallback to all categories
                primary_errors = list(ErrorCategory)

            return OrchestratorResult(
                primary_errors=primary_errors,
                reasoning=data.get(
                    "reasoning",
                    "No reasoning provided in orchestrator response.",
                ),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse orchestrator response: {e}")
            # Fallback: return all error categories as potentially relevant
            return OrchestratorResult(
                primary_errors=list(ErrorCategory),
                reasoning=f"Failed to parse structured response. Error: {str(e)}",
            )

    def _extract_json(self, response: str) -> str:
        """
        Extract JSON string from a response that may contain markdown.

        Args:
            response: The response string.

        Returns:
            Extracted JSON string.
        """
        # Look for JSON in code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()

        # Try finding raw JSON braces
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return response[start:end]

        return response.strip()
