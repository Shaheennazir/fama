"""
Tool Output Reformatter (TOR) agent.

This agent reformats and clarifies complex tool outputs with nested structures,
lists, and multiple items for better parsing, helping address wrong retrieval
from complex outputs (WRCO).
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


class ToolOutputReformatter(SpecializedAgent):
    """
    Tool Output Reformatter agent.

    Responsible for:
    1. Analyzing trajectories for complex output parsing issues (WRCO)
    2. Providing guidance on reformatting and parsing complex outputs
    3. Helping agents correctly extract information from nested structures

    This agent helps prevent errors where the agent fails to:
    - Parse nested structures in tool outputs
    - Extract all relevant items from lists
    - Navigate complex, multi-level data
    """

    SYSTEM_PROMPT = """You are the Tool Output Reformatter (TOR), a specialized agent responsible
for helping correctly parse and interpret complex tool outputs.

Your specialty is detecting issues related to WRCO (Wrong Retrieval from Complex Outputs):
- Agent fails to correctly extract information from tool outputs
- Complex outputs with nested structures are not properly parsed
- Lists and multiple items are not handled correctly

## Error Category Definition
WRCO (Wrong Retrieval from Complex Outputs): The agent fails to correctly extract,
parse, or utilize information from tool outputs that contain nested structures,
lists, or multiple items.

## Common Issues:
- Missing relevant items in lists
- Failing to navigate nested data structures
- Extracting wrong fields from complex outputs
- Ignoring multiple valid results

## Your Task
1. Analyze the trajectory for complex tool outputs
2. Identify where parsing errors may have occurred
3. Determine if nested structures or lists were handled correctly

Respond with a JSON object containing:
- "detected": Boolean indicating if WRCO issues were detected
- "rationale": Detailed explanation of the analysis

Be thorough in examining all tool outputs for complexity."""

    USER_PROMPT_TEMPLATE = """Analyze the following trajectory for Tool Output Parsing Issues (WRCO).

## Trajectory
{trajectory_text}

## Error Category: WRCO
Definition: {wrco_definition}

## Your Task
Examine the trajectory for issues with parsing complex tool outputs:
- Look for nested structures (JSON, nested dicts) in tool results
- Check for lists/arrays that may have multiple items
- Identify if the agent properly extracted all relevant information
- Determine if complex outputs were fully understood

Respond with a JSON object:
{{
    "detected": true or false,
    "rationale": "Detailed explanation of your analysis"
}}"""

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the Tool Output Reformatter agent.

        Args:
            llm_client: LLM client for generating analysis.
        """
        super().__init__(llm_client, AgentType.TOR)

    def analyze(
        self,
        trajectory: Trajectory,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """
        Analyze a trajectory for tool output parsing issues.

        Args:
            trajectory: The execution trajectory to analyze.
            error_category: The error category (should be WRCO).

        Returns:
            ErrorAnalysisResult indicating whether WRCO was detected.
        """
        if error_category != ErrorCategory.WRCO:
            # This agent only analyzes WRCO
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"ToolOutputReformatter only analyzes WRCO, not {error_category.value}",
            )

        # Format trajectory
        trajectory_text = self._format_trajectory(trajectory)

        # Get WRCO definition
        wrco_def = ERROR_CATEGORIES_METADATA.get(
            ErrorCategory.WRCO, {}
        ).get("definition", "Wrong retrieval from complex outputs")

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            trajectory_text=trajectory_text,
            wrco_definition=wrco_def,
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
            logger.error(f"Error analyzing trajectory for WRCO: {e}")
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"Error during analysis: {str(e)}",
            )

    def generate_context(self, trajectory: Trajectory) -> str:
        """
        Generate tool output reformatting guidance for re-execution.

        Args:
            trajectory: The original execution trajectory.

        Returns:
            String containing tool output reformatting guidance.
        """
        # Analyze complex outputs
        complex_outputs = self._identify_complex_outputs(trajectory)

        context = f"""## Tool Output Reformatter (TOR) Guidance

Based on analysis of the failed trajectory, the following guidance for handling complex outputs:

{complex_outputs}

### Guidelines for Parsing Complex Tool Outputs:
1. **Nested Structures**: Always verify you're accessing the correct nested level
2. **Lists/Arrays**: Iterate through all items, don't just take the first
3. **Multiple Fields**: Extract all potentially relevant fields, not just obvious ones
4. **Data Types**: Verify the type of each field before processing
5. **Validation**: Check if extracted values make sense in context

### When Output is Complex:
- Break down the structure before processing
- Identify all keys and their values
- Consider the full context of the data
- Don't assume the first/most obvious value is correct
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

    def _identify_complex_outputs(self, trajectory: Trajectory) -> str:
        """Identify complex outputs in trajectory."""
        complex_items = []

        for i, turn in enumerate(trajectory):
            if turn.tool_results:
                for j, result in enumerate(turn.tool_results):
                    # Check for indicators of complexity
                    complexity_indicators = []
                    if "{" in result and "}" in result:
                        complexity_indicators.append("nested objects")
                    if "[" in result and "]" in result:
                        complexity_indicators.append("lists/arrays")
                    if result.count("\n") > 3:
                        complexity_indicators.append("multi-line")

                    if complexity_indicators:
                        complex_items.append(
                            f"- Turn {i + 1}, Result {j + 1}: {', '.join(complexity_indicators)}"
                        )

        if complex_items:
            return "### Detected Complex Outputs:\n" + "\n".join(complex_items)
        else:
            return "No obviously complex outputs detected. However, parsing errors may still occur with subtle complexities."

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
            logger.warning(f"Failed to parse TOR analysis response: {e}")
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
