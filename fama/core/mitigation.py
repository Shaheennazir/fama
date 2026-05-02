"""
Mitigation agent for FAMA.

The Mitigation agent selects the minimal subset of |A|=6 specialized agents
that can address the identified primary errors.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fama.core.types import (
    AgentType,
    AGENT_POOL,
    ERROR_TO_AGENT_MAPPING,
    ErrorCategory,
    MitigationResult,
)

logger = logging.getLogger(__name__)


class MitigationAgent:
    """
    Mitigation agent for selecting optimal agent subsets.

    Given the primary error(s) identified by the Orchestrator, this agent
    selects the minimal subset of specialized agents from the pool of |A|=6
    that can effectively address the failures.

    The agent mapping defines which agents can address which error categories:
    - DCV -> DCE, VERIFIER
    - WRCO -> TOR, TSA
    - CM -> PLANNER, TSA
    - IFU -> PLANNER, VERIFIER, MEMORY
    """

    SYSTEM_PROMPT = """You are the FAMA Mitigation Agent, responsible for selecting the minimal
subset of specialized agents that can address identified task failures.

You have access to a pool of |A|=6 specialized agents:
- DCE (Domain Constraints Extractor): Extracts and enforces domain constraints
- TSA (Tool Suggestion Agent): Suggests appropriate tools for complex tasks
- TOR (Tool Output Reformatter): Reformats complex tool outputs for better parsing
- Planner: Creates and refines execution plans
- Verifier: Validates task completion and correctness
- Memory: Maintains context and history for long-running tasks

## Error to Agent Mapping
Each error category can be addressed by specific agents:
- DCV (Domain Constraint Violation): DCE, VERIFIER
- WRCO (Wrong Retrieval from Complex Outputs): TOR, TSA
- CM (Contextual Misinterpretation): PLANNER, TSA
- IFU (Incomplete Fulfillment): PLANNER, VERIFIER, MEMORY

## Your Task
Given a list of primary errors that caused task failure, select the MINIMAL
subset of agents that can address ALL identified errors. The goal is to avoid
unnecessary agent injection while ensuring all failure causes are mitigated.

Consider:
1. Which agents can address each error?
2. Which agents appear multiple times (covering multiple errors)?
3. Can a single agent cover multiple errors (preferred)?

Respond with a JSON object containing:
- "selected_agents": List of selected agent types (array of strings from: DCE, TSA, TOR, PLANNER, VERIFIER, MEMORY)
- "reasoning": Detailed explanation of why these agents were selected

IMPORTANT: Select the MINIMAL subset that covers ALL errors. Prefer agents that
can address multiple errors when possible."""

    USER_PROMPT_TEMPLATE = """Given the following primary errors that caused a task failure,
select the minimal subset of agents that can address all of these errors.

## Primary Errors
{primary_errors_text}

## Agent Pool Descriptions
{agent_descriptions}

## Your Task
Select the MINIMAL subset of agents that can address ALL identified errors.
Prefer agents that can handle multiple errors to minimize agent injection.

Respond with a JSON object:
{{
    "selected_agents": ["DCE", "TSA", "TOR", "PLANNER", "VERIFIER", "MEMORY"],  // Minimal subset
    "reasoning": "Detailed explanation of agent selection rationale"
}}"""

    def __init__(
        self,
        llm_client: Any,
        agent_pool: dict[AgentType, Any],
    ) -> None:
        """
        Initialize the Mitigation Agent.

        Args:
            llm_client: LLM client for generating selection decisions.
            agent_pool: Dictionary mapping AgentType to agent instances.
        """
        self.llm_client = llm_client
        self.agent_pool = agent_pool

    def mitigate(
        self,
        errors: list[ErrorCategory],
        agent_pool: dict[AgentType, Any],
    ) -> MitigationResult:
        """
        Select minimal agent subset to mitigate identified errors.

        Args:
            errors: List of error categories to address.
            agent_pool: Dictionary mapping AgentType to agent instances.

        Returns:
            MitigationResult with selected agents and reasoning.
        """
        if not errors:
            return MitigationResult(
                selected_agents=[],
                reasoning="No errors to mitigate - no agents selected.",
            )

        # Build prompts
        primary_errors_text = self._format_errors(errors)
        agent_descriptions = self._format_agent_descriptions()

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            primary_errors_text=primary_errors_text,
            agent_descriptions=agent_descriptions,
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
            logger.debug(f"Mitigation result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in mitigation agent: {e}")
            # Fallback: use greedy selection
            return self._greedy_selection(errors)

    def _format_errors(self, errors: list[ErrorCategory]) -> str:
        """
        Format error list for inclusion in the prompt.

        Args:
            errors: List of ErrorCategory values.

        Returns:
            Formatted error string.
        """
        lines = []
        for error in errors:
            lines.append(f"- {error.value} ({error.name_short})")
        return "\n".join(lines)

    def _format_agent_descriptions(self) -> str:
        """
        Format agent pool descriptions for inclusion in the prompt.

        Returns:
            Formatted agent descriptions string.
        """
        lines = []
        for agent_type, description in AGENT_POOL.items():
            lines.append(f"- {agent_type.value}: {description}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> MitigationResult:
        """
        Parse the LLM response to extract mitigation result.

        Args:
            response: The LLM response string.

        Returns:
            MitigationResult parsed from the response.
        """
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            selected_agents = []
            for agent_str in data.get("selected_agents", []):
                try:
                    agent_type = AgentType(agent_str.lower())
                    selected_agents.append(agent_type)
                except ValueError:
                    # Try matching by partial string
                    for at in AgentType:
                        if at.value.startswith(agent_str.lower()) or at.name.lower() == agent_str.lower():
                            selected_agents.append(at)
                            break

            if not selected_agents:
                # Fallback to greedy selection
                return self._greedy_selection(
                    self._extract_errors_from_reasoning(data.get("reasoning", ""))
                )

            return MitigationResult(
                selected_agents=selected_agents,
                reasoning=data.get(
                    "reasoning",
                    "No reasoning provided in mitigation response.",
                ),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse mitigation response: {e}")
            # Fallback to greedy selection
            return MitigationResult(
                selected_agents=list(self.agent_pool.keys()),
                reasoning=f"Failed to parse structured response. Error: {str(e)}. Selected all agents as fallback.",
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

    def _greedy_selection(
        self,
        errors: list[ErrorCategory],
    ) -> MitigationResult:
        """
        Greedy fallback selection when LLM parsing fails.

        Selects agents using a greedy approach that minimizes the number
        of agents while covering all errors.

        Args:
            errors: List of error categories to address.

        Returns:
            MitigationResult with greedily selected agents.
        """
        selected: list[AgentType] = []
        uncovered_errors = set(errors)

        while uncovered_errors:
            # Find agent that covers the most uncovered errors
            best_agent: Optional[AgentType] = None
            best_coverage: set[ErrorCategory] = set()

            for error in uncovered_errors:
                for agent in ERROR_TO_AGENT_MAPPING.get(error, []):
                    if agent in self.agent_pool:
                        coverage = set(
                            err
                            for err in errors
                            if agent in ERROR_TO_AGENT_MAPPING.get(err, [])
                        )
                        if len(coverage - set(selected)) > len(best_coverage - set(selected)):
                            best_agent = agent
                            best_coverage = coverage

            if best_agent is None:
                # No agent found, break to avoid infinite loop
                break

            selected.append(best_agent)
            uncovered_errors = uncovered_errors - best_coverage

        return MitigationResult(
            selected_agents=selected,
            reasoning=f"Greedy selection: Selected {len(selected)} agents to cover {len(errors)} error categories.",
        )

    def _extract_errors_from_reasoning(self, reasoning: str) -> list[ErrorCategory]:
        """
        Extract error categories from reasoning text as a fallback.

        Args:
            reasoning: The reasoning text.

        Returns:
            List of extracted ErrorCategory values.
        """
        errors = []
        for error in ErrorCategory:
            if error.value in reasoning.lower() or error.name.lower() in reasoning.lower():
                errors.append(error)
        return errors if errors else list(ErrorCategory)
