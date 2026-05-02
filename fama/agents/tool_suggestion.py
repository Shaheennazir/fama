"""
Tool Suggestion Agent (TSA).

This agent suggests appropriate tools for complex tasks based on task
requirements and available tool capabilities, helping address contextual
misinterpretation (CM) and wrong retrieval from complex outputs (WRCO).
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


class ToolSuggestionAgent(SpecializedAgent):
    """
    Tool Suggestion Agent.

    Responsible for:
    1. Analyzing trajectories for tool-related issues (WRCO, CM)
    2. Suggesting appropriate tools for complex tasks
    3. Providing guidance on tool selection and usage

    This agent helps prevent errors where the agent:
    - Fails to select the right tool for a task
    - Misuses or misinterprets tool outputs
    - Doesn't explore alternative tool options
    """

    SYSTEM_PROMPT = """You are the Tool Suggestion Agent (TSA), a specialized agent responsible
for helping select and use tools appropriately in complex task execution.

Your specialty is detecting issues related to:
- WRCO (Wrong Retrieval from Complex Outputs): Agent fails to parse complex tool outputs
- CM (Contextual Misinterpretation): Agent misunderstands task requirements for tool use

## Error Categories Definition
WRCO: The agent fails to correctly extract, parse, or utilize information from tool
outputs that contain nested structures, lists, or multiple items.

CM: The agent understands the literal meaning of words but fails to grasp the actual
user intent or contextual nuance regarding tool usage.

## Your Task
1. Analyze the trajectory for tool-related issues
2. Determine if the agent selected appropriate tools
3. Check if tool outputs were correctly interpreted
4. Identify missed opportunities for better tool selection

Respond with a JSON object containing:
- "detected": Boolean indicating if tool-related issues were detected
- "rationale": Detailed explanation of the analysis

Be thorough in examining tool calls and their results."""

    USER_PROMPT_TEMPLATE = """Analyze the following trajectory for tool-related issues.

## Trajectory
{trajectory_text}

## Error Categories
WRCO (Wrong Retrieval from Complex Outputs):
{werco_definition}

CM (Contextual Misinterpretation):
{cm_definition}

## Your Task
Determine if tool-related issues (WRCO or CM) occurred in this trajectory:
- Were the correct tools selected for the tasks?
- Were tool outputs correctly parsed and interpreted?
- Were complex tool outputs (nested structures, lists) handled properly?
- Did the agent miss any context that would change tool selection?

Respond with a JSON object:
{{
    "detected": true or false,
    "rationale": "Detailed explanation of your analysis"
}}"""

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the Tool Suggestion Agent.

        Args:
            llm_client: LLM client for generating analysis.
        """
        super().__init__(llm_client, AgentType.TSA)

    def analyze(
        self,
        trajectory: Trajectory,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """
        Analyze a trajectory for tool-related issues.

        Args:
            trajectory: The execution trajectory to analyze.
            error_category: The error category (WRCO or CM).

        Returns:
            ErrorAnalysisResult indicating whether issues were detected.
        """
        if error_category not in (ErrorCategory.WRCO, ErrorCategory.CM):
            # This agent only analyzes WRCO and CM
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"ToolSuggestionAgent only analyzes WRCO and CM, not {error_category.value}",
            )

        # Format trajectory
        trajectory_text = self._format_trajectory(trajectory)

        # Get definitions
        werco_def = ERROR_CATEGORIES_METADATA.get(
            ErrorCategory.WRCO, {}
        ).get("definition", "Wrong retrieval from complex outputs")

        cm_def = ERROR_CATEGORIES_METADATA.get(
            ErrorCategory.CM, {}
        ).get("definition", "Contextual misinterpretation")

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            trajectory_text=trajectory_text,
            werco_definition=werco_def,
            cm_definition=cm_def,
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
            logger.error(f"Error analyzing trajectory for tool issues: {e}")
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"Error during analysis: {str(e)}",
            )

    def generate_context(self, trajectory: Trajectory) -> str:
        """
        Generate tool suggestion guidance for re-execution.

        Args:
            trajectory: The original execution trajectory.

        Returns:
            String containing tool suggestion guidance.
        """
        # Analyze tool usage patterns
        tool_analysis = self._analyze_tool_usage(trajectory)

        context = f"""## Tool Suggestion Agent (TSA) Guidance

Based on analysis of the failed trajectory, the following tool-related guidance:

{tool_analysis}

### General Tool Selection Guidelines:
- Carefully examine available tools and select the most appropriate for the task
- When tool outputs are complex (nested, lists), verify your interpretation
- Consider alternative tools when initial attempts fail
- For complex outputs, extract and validate all relevant information
- When uncertain about tool selection, explore tool descriptions and capabilities
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

    def _analyze_tool_usage(self, trajectory: Trajectory) -> str:
        """Analyze tool usage patterns in trajectory."""
        tool_calls_count = 0
        complex_outputs = 0

        for turn in trajectory:
            if turn.tool_calls:
                tool_calls_count += len(turn.tool_calls)
            if turn.tool_results:
                for result in turn.tool_results:
                    if any(char in result for char in ["{", "[", "\n"]):
                        complex_outputs += 1

        return f"""### Observed Tool Usage:
- Total tool calls: {tool_calls_count}
- Complex tool outputs detected: {complex_outputs}

### Recommendations:
- Review tool selection strategy for complex tasks
- Ensure proper parsing of nested and list-based outputs
- Consider multiple tool options before settling on one"""

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
            logger.warning(f"Failed to parse TSA analysis response: {e}")
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
