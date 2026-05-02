"""
Planner agent.

This agent creates and refines execution plans, breaking down complex tasks
into manageable steps, helping address contextual misinterpretation (CM)
and incomplete fulfillment (IFU).
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


class PlannerAgent(SpecializedAgent):
    """
    Planner agent.

    Responsible for:
    1. Analyzing trajectories for planning-related issues (CM, IFU)
    2. Creating and refining execution plans
    3. Ensuring tasks are fully decomposed before execution

    This agent helps prevent errors where the agent:
    - Takes instructions too literally without proper planning
    - Gives up when encountering difficulty instead of planning alternatives
    - Fails to break down complex tasks into steps
    - Stops early instead of completing the full plan
    """

    SYSTEM_PROMPT = """You are the Planner Agent, a specialized agent responsible for
analyzing task execution and identifying planning-related issues.

Your specialty is detecting issues related to:
- CM (Contextual Misinterpretation): Agent understands words but misses intent
- IFU (Incomplete Fulfillment): Agent stops early instead of trying alternatives

## Error Categories Definition
CM: The agent understands the literal meaning of words but fails to grasp the
actual user intent or contextual nuance.

IFU: The agent stops execution when encountering difficulty instead of trying
alternative approaches or persisting to find a solution.

## Your Task
1. Analyze the trajectory for signs of incomplete planning
2. Determine if the agent properly understood the task scope
3. Check if the agent gave up too early or skipped steps
4. Identify missed alternative approaches

Respond with a JSON object containing:
- "detected": Boolean indicating if planning-related issues were detected
- "rationale": Detailed explanation of the analysis

Be thorough in examining the execution flow and decision points."""

    USER_PROMPT_TEMPLATE = """Analyze the following trajectory for Planning-Related Issues (CM, IFU).

## Trajectory
{trajectory_text}

## Error Categories
CM (Contextual Misinterpretation):
{cm_definition}

IFU (Incomplete Fulfillment):
{ifu_definition}

## Your Task
Examine the trajectory for planning-related issues:
- Did the agent fully understand the task requirements?
- Were all necessary steps taken to complete the task?
- Did the agent give up too early when encountering difficulty?
- Were alternative approaches considered when initial attempts failed?
- Were implicit requirements or constraints missed?

Respond with a JSON object:
{{
    "detected": true or false,
    "rationale": "Detailed explanation of your analysis"
}}"""

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the Planner agent.

        Args:
            llm_client: LLM client for generating analysis.
        """
        super().__init__(llm_client, AgentType.PLANNER)

    def analyze(
        self,
        trajectory: Trajectory,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """
        Analyze a trajectory for planning-related issues.

        Args:
            trajectory: The execution trajectory to analyze.
            error_category: The error category (CM or IFU).

        Returns:
            ErrorAnalysisResult indicating whether issues were detected.
        """
        if error_category not in (ErrorCategory.CM, ErrorCategory.IFU):
            # This agent only analyzes CM and IFU
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"PlannerAgent only analyzes CM and IFU, not {error_category.value}",
            )

        # Format trajectory
        trajectory_text = self._format_trajectory(trajectory)

        # Get definitions
        cm_def = ERROR_CATEGORIES_METADATA.get(
            ErrorCategory.CM, {}
        ).get("definition", "Contextual misinterpretation")

        ifu_def = ERROR_CATEGORIES_METADATA.get(
            ErrorCategory.IFU, {}
        ).get("definition", "Incomplete fulfillment")

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            trajectory_text=trajectory_text,
            cm_definition=cm_def,
            ifu_definition=ifu_def,
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
            logger.error(f"Error analyzing trajectory for planning issues: {e}")
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"Error during analysis: {str(e)}",
            )

    def generate_context(self, trajectory: Trajectory) -> str:
        """
        Generate planning guidance for re-execution.

        Args:
            trajectory: The original execution trajectory.

        Returns:
            String containing planning guidance.
        """
        # Analyze planning issues
        planning_analysis = self._analyze_planning(trajectory)

        context = f"""## Planner Agent Guidance

Based on analysis of the failed trajectory, the following planning guidance:

{planning_analysis}

### General Planning Guidelines:
1. **Understand Before Acting**: Ensure full comprehension of task requirements
2. **Decompose Complex Tasks**: Break down into clear, executable steps
3. **Anticipate Challenges**: Plan for potential difficulties in advance
4. **Explore Alternatives**: When one approach fails, consider alternatives
5. **Persist Appropriately**: Distinguish between impossible and difficult tasks
6. **Complete the Full Task**: Don't stop at the first partial success

### For Complex Tasks:
- Create a step-by-step plan before execution
- Identify dependencies between steps
- Set checkpoints to verify progress
- Have contingency plans for failure cases
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

    def _analyze_planning(self, trajectory: Trajectory) -> str:
        """Analyze planning patterns in trajectory."""
        turn_count = len(trajectory)
        tool_call_count = sum(len(turn.tool_calls) for turn in trajectory)

        return f"""### Observed Execution:
- Total turns: {turn_count}
- Total tool calls: {tool_call_count}
- Average tool calls per turn: {tool_call_count / turn_count if turn_count > 0 else 0:.2f}

### Analysis:
{"The execution appears relatively short, potentially indicating early stopping." if turn_count < 5 else "The execution has reasonable length but may have missed steps."}
{"Few tool calls may indicate incomplete task coverage." if tool_call_count < 3 else "Tool usage appears adequate."}

### Recommendations:
- Verify all task requirements are addressed
- Consider if more steps were needed
- Check if alternative approaches should have been tried
"""

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
            logger.warning(f"Failed to parse Planner analysis response: {e}")
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
