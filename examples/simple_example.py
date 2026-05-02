"""
Simple example demonstrating the FAMA framework.

This example shows how to:
1. Set up the FAMA Engine with an LLM client and agent pool
2. Define tasks for execution
3. Run the FAMA pipeline to analyze and mitigate failures
4. Retrieve results for failed tasks

Run with: python examples/simple_example.py
"""

from typing import Any
from fama.core.engine import FAMAEngine, Task
from fama.core.types import (
    AgentType,
    TaskResult,
    Turn,
)


# =============================================================================
# Mock LLM Client (Replace with your actual LLM client in production)
# =============================================================================

class MockLLMClient:
    """
    Mock LLM client for demonstration purposes.
    
    In production, replace this with your actual LLM client
    (e.g., OpenAI, Anthropic, local model, etc.)
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            system_prompt: The system prompt defining the agent's role.
            user_prompt: The user prompt with the task.
            temperature: Sampling temperature (0.0 for deterministic).
            
        Returns:
            Generated response string.
        """
        # This is a placeholder - in production, call your actual LLM
        print(f"[MockLLM] Generating response (temp={temperature})")
        print(f"[MockLLM] System prompt length: {len(system_prompt)} chars")
        print(f"[MockLLM] User prompt length: {len(user_prompt)} chars")
        
        # Return a mock JSON response for the orchestrator/mitigation
        return '''{
    "primary_errors": ["DCV", "CM"],
    "reasoning": "Mock response: Identified DCV and CM as primary errors based on trajectory analysis."
}'''


# =============================================================================
# Mock Agent Pool (Replace with your specialized agents in production)
# =============================================================================

class MockSpecializedAgent:
    """
    Mock specialized agent for demonstration.
    
    In production, implement SpecializedAgent interface from fama.agents.base
    """

    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type

    def analyze(self, trajectory: list, error_category: Any) -> Any:
        """Analyze trajectory for specific error category."""
        # Placeholder - real implementation would do actual analysis
        from fama.core.types import ErrorAnalysisResult
        return ErrorAnalysisResult(
            error_category=error_category,
            detected=False,
            rationale="Mock analysis - no error detected",
        )

    def generate_context(self, trajectory: list) -> str:
        """Generate context for agent injection."""
        return f"Mock context from {self.agent_type.value}"


def create_agent_pool() -> dict[AgentType, Any]:
    """Create a pool of mock specialized agents."""
    return {
        agent_type: MockSpecializedAgent(agent_type)
        for agent_type in AgentType
    }


# =============================================================================
# Custom Baseline Executor
# =============================================================================

def baseline_executor(task: Task) -> TaskResult:
    """
    Custom baseline executor that simulates task execution.
    
    In production, this would execute the actual agent on the task
    and return the trajectory and success status.
    
    Args:
        task: The task to execute.
        
    Returns:
        TaskResult containing the execution trajectory and outcome.
    """
    print(f"\n[BaselineExecutor] Executing task: {task.task_id}")
    print(f"[BaselineExecutor] Description: {task.description}")
    
    # Simulate execution - in production, run actual agent
    # For demo, we'll simulate a failure for certain task patterns
    if "fail" in task.description.lower() or "error" in task.description.lower():
        trajectory = [
            Turn(
                user_message=task.description,
                assistant_message="I'm not sure how to proceed with this task.",
            )
        ]
        success = False
    else:
        trajectory = [
            Turn(
                user_message=task.description,
                assistant_message=f"I have completed the task: {task.description}",
            )
        ]
        success = True
    
    return TaskResult(
        task_id=task.task_id,
        trajectory=trajectory,
        success=success,
        metadata={"executor": "mock_baseline"},
    )


# =============================================================================
# Main Example
# =============================================================================

def main():
    """Run the simple FAMA example."""
    print("=" * 60)
    print("FAMA Framework - Simple Example")
    print("=" * 60)

    # 1. Initialize components
    print("\n[1] Initializing FAMA components...")
    llm_client = MockLLMClient()
    agent_pool = create_agent_pool()
    
    # Create FAMA Engine with custom baseline executor
    engine = FAMAEngine(
        llm_client=llm_client,
        agent_pool=agent_pool,
        baseline_executor=baseline_executor,
    )
    print("[1] FAMA Engine initialized successfully")

    # 2. Define tasks
    print("\n[2] Defining tasks...")
    tasks = [
        Task(
            task_id="task-1",
            description="Book a flight from New York to Los Angeles for tomorrow",
            domain="airline",
        ),
        Task(
            task_id="task-2", 
            description="This task is designed to fail for testing purposes",
            domain="test",
        ),
        Task(
            task_id="task-3",
            description="Change the date of existing booking AB123 to next week",
            domain="airline",
        ),
    ]
    print(f"[2] Created {len(tasks)} tasks")

    # 3. Run FAMA pipeline
    print("\n[3] Running FAMA pipeline...")
    results = engine.run(tasks, execute_baseline_first=True)
    print(f"[3] Pipeline complete - {len(results)} results")

    # 4. Analyze results
    print("\n[4] Analyzing results...")
    success_count = sum(1 for r in results.values() if r.success)
    fail_count = len(results) - success_count
    print(f"[4] Success: {success_count}, Failures: {fail_count}")

    # 5. Display failure details
    print("\n[5] Failure details:")
    for task_id, result in results.items():
        if not result.success:
            print(f"\n  Task: {task_id}")
            print(f"  Success: {result.success}")
            
            # Get error analysis
            error_analysis = engine.get_error_analysis_for_task(task_id)
            if error_analysis:
                print(f"  Error Analysis ({len(error_analysis)} categories):")
                for ea in error_analysis:
                    status = "DETECTED" if ea.detected else "not detected"
                    print(f"    - {ea.error_category.name_short}: {status}")
                    print(f"      {ea.rationale[:100]}...")

            # Get orchestrator result
            orch_result = engine.get_orchestrator_result_for_task(task_id)
            if orch_result:
                print(f"  Primary Errors: {[e.name_short for e in orch_result.primary_errors]}")
                print(f"  Reasoning: {orch_result.reasoning[:100]}...")

            # Get mitigation result
            mit_result = engine.get_mitigation_result_for_task(task_id)
            if mit_result:
                print(f"  Selected Agents: {[a.value for a in mit_result.selected_agents]}")
                print(f"  Mitigation Reasoning: {mit_result.reasoning[:100]}...")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
