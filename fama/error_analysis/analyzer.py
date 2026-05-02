"""
Error analyzer for FAMA.

The ErrorAnalyzer runs per-category analysis agents to analyze trajectories
for the |E|=4 error categories.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fama.core.types import (
    ErrorCategory,
    ErrorAnalysisResult,
    Trajectory,
)
from fama.error_analysis.categories import (
    ERROR_CATEGORIES,
    get_error_category,
)

logger = logging.getLogger(__name__)


# System prompts for each error category
ANALYZER_SYSTEM_PROMPTS: dict[ErrorCategory, str] = {
    ErrorCategory.DCV: """You are an error analysis agent specializing in Domain Constraint Violations (DCV).

Your task is to analyze agent trajectories and determine if DCV errors occurred.

DCV Definition: The agent performs an action that is forbidden by domain policy or 
skips a step that is required by the domain rules.

Look for:
- Forbidden actions the agent might have taken
- Required steps the agent might have skipped
- Misunderstandings of domain policy boundaries
- Policy compliance violations

Respond with a JSON object containing:
- "detected": Boolean indicating if DCV was detected
- "rationale": Detailed explanation of your analysis""",

    ErrorCategory.WRCO: """You are an error analysis agent specializing in Wrong Retrieval from Complex Outputs (WRCO).

Your task is to analyze agent trajectories and determine if WRCO errors occurred.

WRCO Definition: The agent fails to correctly extract, parse, or utilize information 
from tool outputs that contain nested structures, lists, or multiple items.

Look for:
- Nested structures in tool outputs that weren't properly parsed
- Lists where only some items were considered
- Complex outputs where wrong fields were extracted
- Multiple valid results where only one was processed

Respond with a JSON object containing:
- "detected": Boolean indicating if WRCO was detected
- "rationale": Detailed explanation of your analysis""",

    ErrorCategory.CM: """You are an error analysis agent specializing in Contextual Misinterpretation (CM).

Your task is to analyze agent trajectories and determine if CM errors occurred.

CM Definition: The agent understands the literal meaning of words but fails to grasp 
the actual user intent or contextual nuance.

Look for:
- Instructions taken too literally
- Ignored implicit constraints or preferences
- Failure to infer missing information from context
- Missing clarification when needed
- Misinterpretation of user intent

Respond with a JSON object containing:
- "detected": Boolean indicating if CM was detected
- "rationale": Detailed explanation of your analysis""",

    ErrorCategory.IFU: """You are an error analysis agent specializing in Incomplete Fulfillment (IFU).

Your task is to analyze agent trajectories and determine if IFU errors occurred.

IFU Definition: The agent stops execution when encountering difficulty instead of 
trying alternative approaches or persisting to find a solution.

Look for:
- Early stopping when difficulty was encountered
- Lack of alternative strategy exploration
- Premature conclusion that task is impossible
- Giving up after single failed attempt
- Missing retry or escalation attempts

Respond with a JSON object containing:
- "detected": Boolean indicating if IFU was detected
- "rationale": Detailed explanation of your analysis""",
}


class ErrorAnalyzer:
    """
    ErrorAnalyzer runs per-category analysis on trajectories.

    The analyzer takes a trajectory and runs it through |E|=4 independent
    error-analysis agents, one per error category, to produce a list of
    ErrorAnalysisResults.

    Attributes:
        llm_client: The LLM client used for generating analysis.
    """

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the ErrorAnalyzer.

        Args:
            llm_client: LLM client for generating error analysis.
        """
        self.llm_client = llm_client

    def analyze(
        self,
        trajectory: Trajectory,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """
        Analyze a trajectory for a specific error category.

        Args:
            trajectory: The execution trajectory to analyze.
            error_category: The specific error category to analyze.

        Returns:
            ErrorAnalysisResult indicating whether the error was detected
            and the reasoning behind the analysis.
        """
        logger.debug(f"Analyzing trajectory for {error_category.value}")

        # Get category definition
        category_def = get_error_category(error_category)

        # Get system prompt for this category
        system_prompt = ANALYZER_SYSTEM_PROMPTS.get(
            error_category,
            f"You are an error analysis agent. Analyze for {error_category.value}.",
        )

        # Build user prompt with trajectory
        user_prompt = self._build_user_prompt(trajectory, category_def)

        # Generate analysis
        try:
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )

            return self._parse_response(response, error_category)

        except Exception as e:
            logger.error(f"Error analyzing trajectory for {error_category.value}: {e}")
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"Error during analysis: {str(e)}",
            )

    def analyze_all(
        self,
        trajectory: Trajectory,
    ) -> list[ErrorAnalysisResult]:
        """
        Analyze a trajectory for all |E|=4 error categories.

        Args:
            trajectory: The execution trajectory to analyze.

        Returns:
            List of ErrorAnalysisResults, one per error category.
        """
        results = []
        for error_category in ErrorCategory:
            result = self.analyze(trajectory, error_category)
            results.append(result)
        return results

    def _build_user_prompt(
        self,
        trajectory: Trajectory,
        category_def: Any,
    ) -> str:
        """
        Build the user prompt for error analysis.

        Args:
            trajectory: The trajectory to analyze.
            category_def: The error category definition.

        Returns:
            Formatted user prompt string.
        """
        # Format trajectory
        trajectory_text = self._format_trajectory(trajectory)

        prompt = f"""Analyze the following trajectory for {category_def.name} ({category_def.short_name}).

## Error Category Definition
{category_def.definition}

## Common Causes
{chr(10).join(f"- {cause}" for cause in category_def.causes)}

## Trajectory to Analyze
{trajectory_text}

## Your Task
Determine if a {category_def.short_name} error occurred in this trajectory.
Examine the trajectory carefully for signs of this error type.

Respond with a JSON object:
{{
    "detected": true or false,
    "rationale": "Detailed explanation of your analysis"
}}"""

        return prompt

    def _format_trajectory(self, trajectory: Trajectory) -> str:
        """
        Format a trajectory for inclusion in a prompt.

        Args:
            trajectory: The trajectory to format.

        Returns:
            Formatted trajectory string.
        """
        if not trajectory:
            return "No trajectory available."

        import json

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

    def _parse_response(
        self,
        response: str,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """
        Parse the LLM response to extract an ErrorAnalysisResult.

        Args:
            response: The LLM response string.
            error_category: The error category being analyzed.

        Returns:
            ErrorAnalysisResult parsed from the response.
        """
        import json

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
            logger.warning(
                f"Failed to parse error analysis response for {error_category.value}: {e}"
            )
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"Failed to parse analysis response: {str(e)}",
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
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Try finding raw JSON braces
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return response[start:end]

        return response.strip()
