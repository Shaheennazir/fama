"""
Memory agent.

This agent maintains context and history for long-running tasks, ensuring
consistency across multiple turns and helping address incomplete fulfillment (IFU).
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


class MemoryAgent(SpecializedAgent):
    """
    Memory agent.

    Responsible for:
    1. Analyzing trajectories for memory/consistency issues (IFU)
    2. Maintaining context across long-running tasks
    3. Ensuring consistency between turns

    This agent helps prevent errors where the agent:
    - Loses track of earlier context
    - Forgets requirements mentioned earlier
    - Inconsistently references previous information
    - Fails to maintain task state across turns
    """

    SYSTEM_PROMPT = """You are the Memory Agent, a specialized agent responsible for
maintaining context and ensuring consistency across long-running task execution.

Your specialty is detecting issues related to IFU (Incomplete Fulfillment)
that stem from memory or consistency problems.

## Error Category Definition
IFU (Incomplete Fulfillment): The agent stops execution when encountering difficulty
instead of trying alternative approaches or persisting to find a solution.

Memory-related IFU occurs when:
- Agent forgets requirements mentioned earlier in the conversation
- Agent loses context of what has been accomplished
- Agent makes inconsistent statements across turns
- Agent fails to track pending tasks or unmet requirements

## Your Task
1. Analyze the trajectory for memory/consistency issues
2. Determine if the agent maintained context throughout
3. Check for forgotten requirements or lost context
4. Identify inconsistencies in the agent's statements

Respond with a JSON object containing:
- "detected": Boolean indicating if memory-related issues were detected
- "rationale": Detailed explanation of the analysis

Be thorough in examining the flow of information across turns."""

    USER_PROMPT_TEMPLATE = """Analyze the following trajectory for Memory/Consistency Issues (IFU).

## Trajectory
{trajectory_text}

## Error Category: IFU
Definition: {ifu_definition}

## Your Task
Examine the trajectory for memory-related issues:
- Did the agent maintain context throughout execution?
- Were requirements mentioned early in the conversation still tracked?
- Did the agent reference previous information consistently?
- Were there any forgotten or dropped requirements?
- Did the agent make inconsistent statements across turns?

Respond with a JSON object:
{{
    "detected": true or false,
    "rationale": "Detailed explanation of your analysis"
}}"""

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the Memory agent.

        Args:
            llm_client: LLM client for generating analysis.
        """
        super().__init__(llm_client, AgentType.MEMORY)

    def analyze(
        self,
        trajectory: Trajectory,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """
        Analyze a trajectory for memory-related issues.

        Args:
            trajectory: The execution trajectory to analyze.
            error_category: The error category (should be IFU).

        Returns:
            ErrorAnalysisResult indicating whether IFU was detected.
        """
        if error_category != ErrorCategory.IFU:
            # This agent only analyzes IFU
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"MemoryAgent only analyzes IFU, not {error_category.value}",
            )

        # Format trajectory
        trajectory_text = self._format_trajectory(trajectory)

        # Get IFU definition
        ifu_def = ERROR_CATEGORIES_METADATA.get(
            ErrorCategory.IFU, {}
        ).get("definition", "Incomplete fulfillment")

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            trajectory_text=trajectory_text,
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
            logger.error(f"Error analyzing trajectory for memory issues: {e}")
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"Error during analysis: {str(e)}",
            )

    def generate_context(self, trajectory: Trajectory) -> str:
        """
        Generate memory guidance for re-execution.

        Args:
            trajectory: The original execution trajectory.

        Returns:
            String containing memory guidance.
        """
        # Analyze memory patterns
        memory_analysis = self._analyze_memory(trajectory)

        context = f"""## Memory Agent Guidance

Based on analysis of the failed trajectory, the following memory guidance:

{memory_analysis}

### General Memory Guidelines:
1. **Track Requirements**: Keep a running list of task requirements
2. **Reference Earlier Context**: Regularly check back on initial requirements
3. **Maintain State**: Track what's been done and what remains
4. **Consistency Check**: Verify current statements match earlier ones
5. **Summarize Progress**: Periodically summarize what's been accomplished

### For Long-Running Tasks:
- Maintain an explicit record of completed and pending steps
- Check initial requirements against current state
- Don't assume earlier context is still fresh
- Verbally track progress to ensure nothing is forgotten
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

    def _analyze_memory(self, trajectory: Trajectory) -> str:
        """Analyze memory patterns in trajectory."""
        turn_count = len(trajectory)

        # Look for signs of context loss
        first_user = trajectory[0].user_message if trajectory else ""
        last_assistant = trajectory[-1].assistant_message if trajectory else ""

        context = f"""### Observed Execution:
- Total turns: {turn_count}
- Initial user request: {first_user[:100]}...

### Memory Analysis:
{"Long execution with many turns - high risk of context loss." if turn_count > 10 else "Moderate execution length." if turn_count > 5 else "Short execution, lower memory demand."}

### Key Memory Points to Track:
- Initial task requirements
- Tools used and their results
- Progress made so far
- Remaining requirements
- Any constraints mentioned

### Recommendations:
- Periodically summarize what has been accomplished
- Check that early requirements are still being addressed
- Maintain explicit tracking of pending items
- Don't assume earlier context is remembered without stating it
"""

        return context

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
            logger.warning(f"Failed to parse Memory analysis response: {e}")
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
