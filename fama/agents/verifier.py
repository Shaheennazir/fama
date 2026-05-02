"""
Verifier agent.

This agent validates task completion and correctness, identifying potential
issues before final response, helping address domain constraint violations (DCV)
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


class VerifierAgent(SpecializedAgent):
    """
    Verifier agent.

    Responsible for:
    1. Analyzing trajectories for verification-related issues (DCV, IFU)
    2. Validating task completion and correctness
    3. Identifying potential issues before final response

    This agent helps prevent errors where the agent:
    - Fails to verify task completion
    - Misses constraint violations before responding
    - Declares success prematurely without proper verification
    - Skips validation steps
    """

    SYSTEM_PROMPT = """You are the Verifier Agent, a specialized agent responsible for
validating task completion and ensuring correctness of agent responses.

Your specialty is detecting issues related to:
- DCV (Domain Constraint Violation): Agent violates domain policies
- IFU (Incomplete Fulfillment): Agent stops early without proper verification

## Error Categories Definition
DCV: The agent performs an action that is forbidden by domain policy or skips
a step that is required by the domain rules.

IFU: The agent stops execution when encountering difficulty instead of trying
alternative approaches or persisting to find a solution.

## Your Task
1. Analyze the trajectory for verification gaps
2. Determine if the agent properly verified task completion
3. Check if potential constraint violations were overlooked
4. Identify premature success declarations

Respond with a JSON object containing:
- "detected": Boolean indicating if verification issues were detected
- "rationale": Detailed explanation of the analysis

Be thorough in examining the final state and verification steps."""

    USER_PROMPT_TEMPLATE = """Analyze the following trajectory for Verification Issues (DCV, IFU).

## Trajectory
{trajectory_text}

## Error Categories
DCV (Domain Constraint Violation):
{dcv_definition}

IFU (Incomplete Fulfillment):
{ifu_definition}

## Your Task
Examine the trajectory for verification-related issues:
- Did the agent properly verify task completion?
- Were all constraints and requirements checked?
- Did the agent declare success without proper validation?
- Were potential issues identified before final response?
- Did the agent skip verification steps?

Respond with a JSON object:
{{
    "detected": true or false,
    "rationale": "Detailed explanation of your analysis"
}}"""

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the Verifier agent.

        Args:
            llm_client: LLM client for generating analysis.
        """
        super().__init__(llm_client, AgentType.VERIFIER)

    def analyze(
        self,
        trajectory: Trajectory,
        error_category: ErrorCategory,
    ) -> ErrorAnalysisResult:
        """
        Analyze a trajectory for verification-related issues.

        Args:
            trajectory: The execution trajectory to analyze.
            error_category: The error category (DCV or IFU).

        Returns:
            ErrorAnalysisResult indicating whether issues were detected.
        """
        if error_category not in (ErrorCategory.DCV, ErrorCategory.IFU):
            # This agent only analyzes DCV and IFU
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"VerifierAgent only analyzes DCV and IFU, not {error_category.value}",
            )

        # Format trajectory
        trajectory_text = self._format_trajectory(trajectory)

        # Get definitions
        dcv_def = ERROR_CATEGORIES_METADATA.get(
            ErrorCategory.DCV, {}
        ).get("definition", "Domain constraint violation")

        ifu_def = ERROR_CATEGORIES_METADATA.get(
            ErrorCategory.IFU, {}
        ).get("definition", "Incomplete fulfillment")

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            trajectory_text=trajectory_text,
            dcv_definition=dcv_def,
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
            logger.error(f"Error analyzing trajectory for verification issues: {e}")
            return ErrorAnalysisResult(
                error_category=error_category,
                detected=False,
                rationale=f"Error during analysis: {str(e)}",
            )

    def generate_context(self, trajectory: Trajectory) -> str:
        """
        Generate verification guidance for re-execution.

        Args:
            trajectory: The original execution trajectory.

        Returns:
            String containing verification guidance.
        """
        # Analyze verification patterns
        verification_analysis = self._analyze_verification(trajectory)

        context = f"""## Verifier Agent Guidance

Based on analysis of the failed trajectory, the following verification guidance:

{verification_analysis}

### General Verification Guidelines:
1. **Check Task Completion**: Verify all requirements are met before responding
2. **Validate Constraints**: Ensure all domain constraints are satisfied
3. **Review Tool Results**: Double-check that tool outputs support conclusions
4. **Cross-validate**: Use multiple sources to verify critical information
5. **Error Detection**: Actively look for potential issues before final response

### Before Declaring Success:
- Have all requirements been fulfilled?
- Are there any unmet constraints?
- Does the evidence support the conclusion?
- Could there be alternative interpretations?
- What might be wrong with this solution?
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

    def _analyze_verification(self, trajectory: Trajectory) -> str:
        """Analyze verification patterns in trajectory."""
        if not trajectory:
            return "No trajectory available for analysis."

        last_turn = trajectory[-1]
        has_success_keywords = any(
            keyword in last_turn.assistant_message.lower()
            for keyword in ["complete", "done", "finished", "success", "verified"]
        )

        return f"""### Observed Final State:
- Final assistant message: {last_turn.assistant_message[:100]}...
- Contains success indicators: {has_success_keywords}

### Analysis:
{"The agent declared completion - verify this was warranted." if has_success_keywords else "No explicit success declaration found."}

### Recommendations:
- Always verify before declaring success
- Check that all aspects of the task were addressed
- Look for potential issues that might have been overlooked
- Consider what could still be wrong even if task appears complete
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
            logger.warning(f"Failed to parse Verifier analysis response: {e}")
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
