"""
LLM-based user simulator for multi-turn conversational benchmarks.

The simulator generates realistic user responses based on conversation history,
enabling fully autonomous benchmark evaluation without human-in-the-loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class UserSimulator:
    """
    Simulates a user in a customer-service conversation.
    
    The simulator uses an LLM to generate natural, goal-oriented user
    responses that follow the domain policy and task requirements.
    
    Example:
        simulator = UserSimulator(llm_client, domain_policy=policy)
        response = simulator.generate_response(
            agent_message="I found your booking. Would you like to proceed with cancellation?",
            conversation_history=[...],
        )
    """
    
    def __init__(
        self,
        llm_client: Any,
        domain_policy: str,
        task_goal: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the user simulator.
        
        Args:
            llm_client: LLM client with a generate() method
            domain_policy: The domain policy governing the conversation
            task_goal: The user's underlying goal (optional, for guided simulation)
            temperature: LLM temperature for response generation
        """
        self.llm = llm_client
        self.domain_policy = domain_policy
        self.task_goal = task_goal or ""
        self.temperature = temperature
        self._conversation_complete = False
    
    def generate_response(
        self,
        agent_message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate the next user response in the conversation.
        
        Args:
            agent_message: The most recent message from the agent
            conversation_history: Full conversation history
            
        Returns:
            The user's response as a string
        """
        history = conversation_history or []
        
        system_prompt = f"""You are a realistic user in a customer service conversation.
        
## Domain Policy (what you can request and expect):
{self.domain_policy}

## Your Goal:
{self.task_goal}

## Instructions:
- Respond naturally as a customer would
- Provide information when asked (but verify it makes sense)
- Be cooperative but can express mild frustration if needed
- If the agent completes your request satisfactorily, confirm completion
- Keep responses concise (1-3 sentences typically)
- Do NOT reveal you are an AI or mention the policy
"""
        
        messages_for_llm = []
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages_for_llm.append({"role": "user", "content": content})
            elif role == "assistant":
                messages_for_llm.append({"role": "assistant", "content": content})
        
        messages_for_llm.append({
            "role": "user",
            "content": f"Agent: {agent_message}\n\nYour response as the user:",
        })
        
        try:
            response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt="\n".join([
                    f"{m['role']}: {m['content']}" for m in messages_for_llm
                ]),
                temperature=self.temperature,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Simulator LLM error: {e}")
            return "Thank you for your help."
    
    def is_task_complete(self) -> bool:
        """Check if the simulated user considers the task complete."""
        return self._conversation_complete
    
    def set_task_complete(self, complete: bool = True) -> None:
        """Mark the task as complete from the user's perspective."""
        self._conversation_complete = complete
    
    def reset(self) -> None:
        """Reset the simulator for a new conversation."""
        self._conversation_complete = False


class TauDialogueConverter:
    """
    Converts τ-bench style trajectories to/from FAMA Trajectory format.
    
    This allows interoperability between the benchmark's trajectory format
    and the FAMA framework's internal Trajectory representation.
    """
    
    @staticmethod
    def from_tau_trajectory(tau_trajectory: list[dict]) -> list[dict]:
        """
        Convert τ-bench trajectory format to FAMA Turn format.
        
        Args:
            tau_trajectory: List of τ-bench turns with keys like
                          user_message, assistant_message, tool_calls, tool_results
                          
        Returns:
            List of turns in FAMA format
        """
        from fama.core.types import Turn
        
        turns = []
        for tau_turn in tau_trajectory:
            turn = {
                "turn_number": len(turns) + 1,
                "user_message": tau_turn.get("user_message", ""),
                "assistant_message": tau_turn.get("assistant_message", ""),
                "tool_calls": tau_turn.get("tool_calls", []),
                "tool_results": tau_turn.get("tool_results", []),
            }
            turns.append(turn)
        
        return turns
    
    @staticmethod
    def to_tau_format(turns: list[dict]) -> list[dict]:
        """
        Convert FAMA Turn format to τ-bench trajectory format.
        """
        return [
            {
                "user_message": t.get("user_message", ""),
                "assistant_message": t.get("assistant_message", ""),
                "tool_calls": t.get("tool_calls", []),
                "tool_results": t.get("tool_results", []),
            }
            for t in turns
        ]
