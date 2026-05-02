"""
Domain Constraints Extractor (DCE) agent.

This agent extracts and enforces domain-specific constraints and policies
to prevent domain constraint violations (DCV).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fama.agents.base import SpecializedAgent
from fama.core.types import (
    AgentType,
    ERROR_CATEGORIES_METADATA,
    ErrorAnalysisResult,
    ErrorCategory,
    Trajectory,
)

logger = logging.getLogger(__name__)


class DomainConstraintsExtractor(SpecializedAgent):
    """
    Domain Constraints Extractor agent.

    Responsible for:
    1. Analyzing trajectories for Domain Constraint Violations (DCV)
    2. Extracting relevant domain constraints from task context
    3. Generating guidance to prevent policy violations

    This agent helps prevent the agent from performing forbidden actions
    or skipping required steps defined by domain policies.
    """

    SYSTEM_PROMPT = """You are the Domain Constraints Extractor (DCE), a specialized agent
responsible for identifying and enforcing domain-specific constraints and policies.

Your specialty is detecting Domain Constraint Violations (DCV) where an agent:
- Performs an action that is forbidden by domain policy
- Skips a step that is required by domain rules
- Misunderstands the boundaries of allowed actions

## Error Category Definition
DCV (Domain Constraint Violation): The agent performs an action that is forbidden
by domain policy or skips a step that is required by the domain rules.

## Your Task
1. Analyze the provided trajectory for signs of domain constraint violations
2. Identify any constraints that should govern the agent's behavior
3. Detect if the agent violated or missed any domain constraints

Respond with a JSON object containing:
- "detected": Boolean indicating if DCV was detected
- "rationale": Detailed explanation of the analysis

Be thorough in examining the trajectory for constraint violations."""

    USER_PROMPT_TEMPLATE = """Analyze the following trajectory for Domain Constraint Violations (DCV).

## Trajectory
{trajectory_text}

## Error Category: DCV
Definition: {dcv_definition}

## Your Task
Determine if a Domain Constraint Violation (DCV) occurred in this trajectory.
Check for:
- Forbidden actions the agent might have taken
- Required steps the agent might have skipped
- Misunderstandings of domain policy boundaries

Respond with a JSON object:
{{
    "detected": true or false,
    "rationale": "Detailed explanation of your analysis"
}}"""

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the Domain Constraints Extractor agent.

        Args:
            llm_client: LLM client for generating analysis.
        """
        super().__init__(llm_client, AgentType.DCE)

    def analyze(
        self,
        trajectory: Trajectory,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """
        Analyze a trajectory for domain constraint violations.

        Args:
            trajectory: The execution trajectory to analyze.
            error_category: The error category (should be DCV).

        Returns:
            ErrorAnalysisResult indicating whether DCV was detected.
        """
        if error_category != ErrorCategory.DCV:
            # This agent only analyzes DCV
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"DomainConstraintsExtractor only analyzes DCV, not {error_category.value}",
            )

        # Format trajectory
        trajectory_text = self._format_trajectory(trajectory)

        # Get DCV definition
        dcw_definition = ERROR_CATEGORIES_METADATA.get(
            ErrorCategory.DCV, {}
        ).get("definition", "Domain constraint violation")

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            trajectory_text=trajectory_text,
            dcv_definition=dcw_definition,
        )

        # Generate analysis
        try:
            response = self.llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
            )

            return self._parse_response(response, error_category)

        except Exception as e:
            logger.error(f"Error analyzing trajectory for DCV: {e}")
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"Error during analysis: {str(e)}",
            )

    def generate_context(self, trajectory: Trajectory) -> str:
        """
        Generate domain constraint guidance for re-execution.

        Args:
            trajectory: The original execution trajectory.

        Returns:
            String containing domain constraint guidance.
        """
        # Extract potential domain constraints from trajectory
        constraints_text = self._extract_constraints(trajectory)

        context = f"""## Domain Constraints Extractor (DCE) Guidance

Based on analysis of the failed trajectory, the following domain constraints should be considered:

{constraints_text if constraints_text else "No specific domain constraints identified. Ensure all actions comply with domain policies."}

### General DCV Prevention Guidelines:
- Always verify actions are permitted before executing
- Check for required steps that must be completed
- When uncertain about constraints, seek clarification
- Prioritize policy compliance over task completion speed
"""
        return context

    def _format_trajectory(self, trajectory: Trajectory) -> str:
        """Format trajectory for analysis."""
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

    def _extract_constraints(self, trajectory: Trajectory) -> str:
        """Extract potential domain constraints from trajectory."""
        # This would ideally use more sophisticated extraction
        # For now, we provide general guidance
        return """- Ensure all tool calls comply with domain-specific policies
- Verify each step completes required domain-specific requirements
- Document any constraints inferred from user messages and domain context"""

    def _parse_response(
        self,
        response: str,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """Parse LLM response for error analysis result."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            return ErrorAnalysisResult(
                error_category=error_category,
                detected=data.get("detected", False),
                rationale=data.get(
                    "rationale",
                    "No rationale provided in analysis response.",
                ),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse DCE analysis response: {e}")
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"Failed to parse analysis response: {str(e)}",
            )

    def _extract_json(self, response: str) -> str:
        """Extract JSON from response."""
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()

        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return response[start:end]

        return response.strip()
