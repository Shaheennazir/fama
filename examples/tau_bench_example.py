"""
TAU Bench example demonstrating the FAMA framework on dialogue tasks.

This example shows how to:
1. Use TAU Bench format for task definitions
2. Configure FAMA for dialogue/turn-level tasks
3. Run the FAMA pipeline with TAU-style trajectories
4. Analyze results using FAMA's error categories

TAU Bench (Task-based Agent Utility Benchmark) evaluates agents on
realistic dialogue tasks with multiple turns.

Run with: python examples/tau_bench_example.py
"""

from typing import Any, Optional
from dataclasses import dataclass
from fama.core.engine import FAMAEngine, Task
from fama.core.types import (
    AgentType,
    TaskResult,
    Turn,
    Trajectory,
)


# =============================================================================
# TAU Bench Data Structures
# =============================================================================

@dataclass
class TAUTurn:
    """Represents a single turn in a TAU dialogue."""
    role: str  # "user" or "agent"
    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_results: Optional[list[str]] = None


@dataclass 
class TAUDialogue:
    """Represents a complete TAU dialogue/task."""
    dialogue_id: str
    domain: str
    task_description: str
    turns: list[TAUTurn]
    ground_truth: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


# =============================================================================
# Mock LLM Client for TAU Bench
# =============================================================================

class TAUMockLLMClient:
    """
    Mock LLM client with TAU-specific response patterns.
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Generate a mock TAU response."""
        # Detect which agent is being simulated based on system prompt
        if "Orchestrator" in system_prompt:
            return self._orchestrator_response(user_prompt)
        elif "Mitigation" in system_prompt:
            return self._mitigation_response(user_prompt)
        elif "DCV" in system_prompt:
            return self._dcv_analysis_response(user_prompt)
        elif "WRCO" in system_prompt:
            return self._wrco_analysis_response(user_prompt)
        elif "CM" in system_prompt:
            return self._cm_analysis_response(user_prompt)
        elif "IFU" in system_prompt:
            return self._ifu_analysis_response(user_prompt)
        else:
            return '{"detected": false, "rationale": "No error detected"}'

    def _orchestrator_response(self, user_prompt: str) -> str:
        """Generate orchestrator response based on trajectory analysis."""
        # Simple heuristic based on trajectory content
        if "uncertain" in user_prompt.lower() or "not sure" in user_prompt.lower():
            return '''{
    "primary_errors": ["IFU", "CM"],
    "reasoning": "The agent showed uncertainty and failed to complete the task. IFU detected (early stopping) and CM (misinterpretation of user intent)."
}'''
        elif "constraint" in user_prompt.lower() or "policy" in user_prompt.lower():
            return '''{
    "primary_errors": ["DCV"],
    "reasoning": "Domain constraint violation detected - agent violated airline policy regarding booking modifications."
}'''
        else:
            return '''{
    "primary_errors": ["WRCO"],
    "reasoning": "Wrong retrieval from complex outputs - agent failed to properly extract information from tool results."
}'''

    def _mitigation_response(self, user_prompt: str) -> str:
        """Generate mitigation response with optimal agent subset."""
        if "IFU" in user_prompt and "CM" in user_prompt:
            return '''{
    "selected_agents": ["PLANNER"],
    "reasoning": "Planner can address both IFU (by creating better execution plans) and CM (by clarifying user intent in the plan)."
}'''
        elif "DCV" in user_prompt:
            return '''{
    "selected_agents": ["DCE", "VERIFIER"],
    "reasoning": "DCE extracts domain constraints and VERIFIER ensures policy compliance."
}'''
        else:
            return '''{
    "selected_agents": ["TSA", "TOR"],
    "reasoning": "TSA suggests appropriate tools and TOR reformats complex outputs for better parsing."
}'''

    def _dcv_analysis_response(self, user_prompt: str) -> str:
        """Generate DCV analysis response."""
        if "constraint" in user_prompt.lower() or "policy" in user_prompt.lower():
            return '{"detected": true, "rationale": "Agent violated domain constraint by attempting to modify a cancelled booking."}'
        return '{"detected": false, "rationale": "No domain constraint violation observed."}'

    def _wrco_analysis_response(self, user_prompt: str) -> str:
        """Generate WRCO analysis response."""
        if "tool" in user_prompt.lower() and ("list" in user_prompt.lower() or "nested" in user_prompt.lower()):
            return '{"detected": true, "rationale": "Agent failed to properly parse list of available flights from tool output."}'
        return '{"detected": false, "rationale": "No wrong retrieval from complex outputs detected."}'

    def _cm_analysis_response(self, user_prompt: str) -> str:
        """Generate CM analysis response."""
        if "intent" in user_prompt.lower() or "meaning" in user_prompt.lower():
            return '{"detected": true, "rationale": "Agent misunderstood user intent - user wanted to cancel, not modify."}'
        return '{"detected": false, "rationale": "No contextual misinterpretation detected."}'

    def _ifu_analysis_response(self, user_prompt: str) -> str:
        """Generate IFU analysis response."""
        if "uncertain" in user_prompt.lower() or "not sure" in user_prompt.lower():
            return '{"detected": true, "rationale": "Agent stopped after first failed attempt without trying alternative approaches."}'
        return '{"detected": false, "rationale": "Agent completed the task fully."}'


# =============================================================================
# TAU Bench Task Conversion
# =============================================================================

def convert_tau_dialogue_to_task(dialogue: TAUDialogue) -> Task:
    """
    Convert a TAU dialogue to a FAMA Task.
    
    Args:
        dialogue: TAUDialogue instance.
        
    Returns:
        Task suitable for FAMA pipeline.
    """
    return Task(
        task_id=dialogue.dialogue_id,
        description=dialogue.task_description,
        domain=dialogue.domain,
        metadata={
            "ground_truth": dialogue.ground_truth,
            "turn_count": len(dialogue.turns),
        },
    )


def convert_tau_dialogue_to_trajectory(dialogue: TAUDialogue) -> Trajectory:
    """
    Convert TAU dialogue turns to FAMA Trajectory.
    
    Args:
        dialogue: TAUDialogue instance.
        
    Returns:
        Trajectory (list of Turns) for FAMA.
    """
    turns = []
    for tau_turn in dialogue.turns:
        role = "user" if tau_turn.role == "user" else "assistant"
        content = tau_turn.content
        
        turn = Turn(
            user_message=content if role == "user" else "",
            assistant_message=content if role == "agent" else "",
            tool_calls=tau_turn.tool_calls or [],
            tool_results=tau_turn.tool_results or [],
        )
        turns.append(turn)
    
    return turns


# =============================================================================
# TAU-style Baseline Executor
# =============================================================================

def tau_baseline_executor(task: Task) -> TaskResult:
    """
    Simulate baseline execution for TAU task.
    
    In production, this would call the actual agent.
    
    Args:
        task: The task to execute.
        
    Returns:
        TaskResult with simulated trajectory and outcome.
    """
    # Simulate a failed execution for demonstration
    trajectory = [
        Turn(
            user_message=task.description,
            assistant_message="I need to process your request.",
        ),
        Turn(
            user_message="Can you help me modify my booking?",
            assistant_message="I'm not sure I understand. Could you please provide more details?",
        ),
        Turn(
            user_message="I want to change the date",
            assistant_message="I'm uncertain about how to proceed with date changes. I think the task cannot be completed.",
        ),
    ]
    
    # Determine success (for demo, task-1 succeeds, others fail)
    success = task.task_id == "tau-task-1"
    
    return TaskResult(
        task_id=task.task_id,
        trajectory=trajectory,
        success=success,
        error_categories=[] if success else [ErrorCategory.IFU, ErrorCategory.CM],
        metadata={"domain": task.domain, "execution_mode": "tau_baseline"},
    )


# =============================================================================
# TAU Example Dialogues
# =============================================================================

def get_sample_tau_dialogues() -> list[TAUDialogue]:
    """Get sample TAU dialogues for demonstration."""
    return [
        TAUDialogue(
            dialogue_id="tau-task-1",
            domain="airline",
            task_description="Book a flight from New York to Los Angeles for tomorrow",
            turns=[
                TAUTurn(role="user", content="I need a flight from New York to Los Angeles"),
                TAUTurn(role="agent", content="I'd be happy to help you book a flight. What date would you like to fly?"),
                TAUTurn(role="user", content="Tomorrow please"),
                TAUTurn(
                    role="agent",
                    content="I found several flights. Let me book the first one for you.",
                    tool_calls=[{"name": "search_flights", "args": {"from": "NYC", "to": "LA", "date": "2024-12-20"}}],
                    tool_results=['[{"id": "AA123", "price": 350, "departure": "10:00"}, {"id": "UA456", "price": 380, "departure": "14:00"}]'],
                ),
                TAUTurn(role="agent", content="I've successfully booked flight AA123 from New York to Los Angeles for $350."),
            ],
            ground_truth={"booking_id": "AA123", "confirmed": True},
        ),
        TAUDialogue(
            dialogue_id="tau-task-2",
            domain="airline",
            task_description="Modify existing booking AB123 to a different date",
            turns=[
                TAUTurn(role="user", content="I need to modify my booking AB123"),
                TAUTurn(role="agent", content="Let me look up your booking."),
                TAUTurn(
                    role="agent",
                    content="I found booking AB123. It's currently scheduled for Dec 20th. What date would you like to change to?",
                    tool_calls=[{"name": "get_booking", "args": {"booking_id": "AB123"}}],
                    tool_results=['{"booking_id": "AB123", "status": "cancelled", "original_date": "2024-12-20"}'],
                ),
                TAUTurn(role="user", content="Next week please"),
                TAUTurn(role="agent", content="I'm not sure I understand. Could you please provide more details?"),
                TAUTurn(role="user", content="I want to change the date to December 27th"),
                TAUTurn(role="agent", content="I'm uncertain about how to proceed with date changes. I think the task cannot be completed."),
            ],
            ground_truth={"booking_id": "AB123", "status": "cancelled", "action": "modify_failed"},
        ),
        TAUDialogue(
            dialogue_id="tau-task-3",
            domain="retail",
            task_description="Process a return for order ORD789",
            turns=[
                TAUTurn(role="user", content="I need to return my recent order"),
                TAUTurn(role="agent", content="I'd be happy to help with a return. Can you provide the order number?"),
                TAUTurn(role="user", content="It's ORD789"),
                TAUTurn(
                    role="agent",
                    content="Let me check the status of order ORD789.",
                    tool_calls=[{"name": "get_order", "args": {"order_id": "ORD789"}}],
                    tool_results=['{"order_id": "ORD789", "items": [{"sku": "ABC123", "qty": 2}, {"sku": "DEF456", "qty": 1}], "return_window": "closed"}'],
                ),
                TAUTurn(role="agent", content="I've found your order. Unfortunately, the return window has closed. Is there anything else I can help with?"),
            ],
            ground_truth={"order_id": "ORD789", "return_eligible": False, "reason": "window_closed"},
        ),
    ]


# =============================================================================
# Main TAU Example
# =============================================================================

def main():
    """Run the TAU Bench FAMA example."""
    print("=" * 70)
    print("FAMA Framework - TAU Bench Example")
    print("=" * 70)

    # 1. Initialize TAU-specific components
    print("\n[1] Initializing TAU components...")
    llm_client = TAUMockLLMClient()
    
    # Create mock agent pool
    class MockTAUAgent:
        def __init__(self, agent_type: AgentType):
            self.agent_type = agent_type
        def analyze(self, trajectory: Trajectory, error_category: Any) -> Any:
            from fama.core.types import ErrorAnalysisResult
            return ErrorAnalysisResult(error_category=error_category, detected=False, rationale="Mock")
        def generate_context(self, trajectory: Trajectory) -> str:
            return f"Mock context from {self.agent_type.value}"
    
    agent_pool = {at: MockTAUAgent(at) for at in AgentType}
    
    # Create engine
    engine = FAMAEngine(
        llm_client=llm_client,
        agent_pool=agent_pool,
        baseline_executor=tau_baseline_executor,
    )
    print("[1] TAU components initialized")

    # 2. Load TAU dialogues
    print("\n[2] Loading TAU dialogues...")
    tau_dialogues = get_sample_tau_dialogues()
    print(f"[2] Loaded {len(tau_dialogues)} TAU dialogues")

    # 3. Convert to FAMA tasks
    print("\n[3] Converting TAU dialogues to FAMA tasks...")
    tasks = [convert_tau_dialogue_to_task(d) for d in tau_dialogues]
    print(f"[3] Created {len(tasks)} FAMA tasks")

    # 4. Display task details
    print("\n[4] Task overview:")
    for task in tasks:
        print(f"\n  {task.task_id} ({task.domain})")
        print(f"    Description: {task.task_description}")
        tau_diag = next(d for d in tau_dialogues if d.dialogue_id == task.task_id)
        print(f"    Turns: {len(tau_diag.turns)}")

    # 5. Run FAMA pipeline
    print("\n[5] Running FAMA pipeline on TAU tasks...")
    results = engine.run(tasks, execute_baseline_first=True)
    print(f"[5] Pipeline complete")

    # 6. Analyze results
    print("\n[6] Results summary:")
    success_count = sum(1 for r in results.values() if r.success)
    fail_count = len(results) - success_count
    print(f"    Success: {success_count}")
    print(f"    Failures: {fail_count}")

    # 7. Detailed failure analysis
    print("\n[7] Detailed failure analysis:")
    for task_id, result in results.items():
        print(f"\n  {'='*60}")
        print(f"  Task: {task_id}")
        print(f"  {'='*60}")
        print(f"  Success: {result.success}")
        
        if not result.success:
            # Show trajectory
            print(f"\n  Trajectory ({len(result.trajectory)} turns):")
            for i, turn in enumerate(result.trajectory):
                user = turn.user_message[:50] + "..." if len(turn.user_message) > 50 else turn.user_message
                assistant = turn.assistant_message[:50] + "..." if len(turn.assistant_message) > 50 else turn.assistant_message
                print(f"    Turn {i+1}: User: '{user}'")
                print(f"           Agent: '{assistant}'")
            
            # Show error analysis
            error_analysis = engine.get_error_analysis_for_task(task_id)
            if error_analysis:
                print(f"\n  Error Analysis:")
                for ea in error_analysis:
                    status = "✓ DETECTED" if ea.detected else "✗ not detected"
                    print(f"    {ea.error_category.name_short}: {status}")
            
            # Show orchestrator result
            orch_result = engine.get_orchestrator_result_for_task(task_id)
            if orch_result:
                print(f"\n  Primary Errors: {[e.name_short for e in orch_result.primary_errors]}")
                print(f"  Orchestrator Reasoning: {orch_result.reasoning[:150]}...")
            
            # Show mitigation result
            mit_result = engine.get_mitigation_result_for_task(task_id)
            if mit_result:
                print(f"\n  Selected Mitigation Agents: {[a.value for a in mit_result.selected_agents]}")
                print(f"  Mitigation Reasoning: {mit_result.reasoning[:150]}...")

    # 8. Show agent selection summary
    print("\n" + "=" * 70)
    print("[8] Agent Selection Summary:")
    print("=" * 70)
    
    agent_selection_count: dict[str, int] = {}
    for task_id, result in engine._mitigation_results.items():
        for agent in result.selected_agents:
            agent_name = agent.value
            agent_selection_count[agent_name] = agent_selection_count.get(agent_name, 0) + 1
    
    if agent_selection_count:
        print("\n  Agent selection frequency:")
        for agent_name, count in sorted(agent_selection_count.items()):
            print(f"    {agent_name}: {count} task(s)")
    else:
        print("\n  No agents selected (all tasks succeeded)")

    print("\n" + "=" * 70)
    print("TAU Bench example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
