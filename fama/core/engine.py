"""
FAMA Engine implementation.

The FAMAEngine implements the core FAMA algorithm for failure-aware
meta-agentic optimization.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from fama.core.types import (
    AgentType,
    ERROR_CATEGORIES_METADATA,
    AGENT_POOL,
    ErrorCategory,
    ErrorAnalysisResult,
    MitigationResult,
    OrchestratorResult,
    TaskResult,
    Trajectory,
    Turn,
)
from fama.core.orchestrator import Orchestrator
from fama.core.mitigation import MitigationAgent
from fama.error_analysis.analyzer import ErrorAnalyzer
from fama.agents.base import SpecializedAgent

logger = logging.getLogger(__name__)


class FAMAEngine:
    """
    Failure-Aware Meta-Agentic Engine.

    Implements the two-stage FAMA optimization pipeline:
    1. Execute baseline agent on tasks, analyze failures with |E|=4 error categories
    2. Select minimal subset of |A|=6 agents and re-run with mitigation

    Attributes:
        llm_client: The LLM client used for analysis and decision making.
        agent_pool: Dictionary mapping AgentType to SpecializedAgent instances.
        error_analyzer: ErrorAnalyzer instance for performing category-specific analysis.
        orchestrator: Orchestrator instance for identifying primary errors.
        mitigation_agent: MitigationAgent for selecting optimal agent subsets.
    """

    def __init__(
        self,
        llm_client: Any,
        agent_pool: dict[AgentType, SpecializedAgent],
        baseline_executor: Optional[Callable[[Task], TaskResult]] = None,
    ) -> None:
        """
        Initialize the FAMA Engine.

        Args:
            llm_client: LLM client for analysis and decision making.
            agent_pool: Dictionary mapping AgentType to SpecializedAgent instances.
            baseline_executor: Optional callable that executes a task and returns TaskResult.
                              If not provided, a default executor is used.
        """
        self.llm_client = llm_client
        self.agent_pool = agent_pool
        self.baseline_executor = baseline_executor

        # Initialize components
        self.error_analyzer = ErrorAnalyzer(llm_client)
        self.orchestrator = Orchestrator(llm_client)
        self.mitigation_agent = MitigationAgent(llm_client, agent_pool)

        # Track all tasks and results
        self._tasks: list[Task] = []
        self._results: dict[str, TaskResult] = {}
        self._failed_tasks: list[TaskResult] = []
        self._error_analysis: dict[str, list[ErrorAnalysisResult]] = {}
        self._orchestrator_results: dict[str, OrchestratorResult] = {}
        self._mitigation_results: dict[str, MitigationResult] = {}

    def run(
        self,
        tasks: list[Task],
        execute_baseline_first: bool = True,
    ) -> dict[str, TaskResult]:
        """
        Run the complete FAMA pipeline on a list of tasks.

        This implements the full Algorithm 1 from the paper:
        1. Execute baseline agent on all tasks
        2. For failed tasks, analyze with |E|=4 error categories
        3. Orchestrate to identify primary errors
        4. Mitigate by selecting minimal agent subset

        Args:
            tasks: List of tasks to execute and optimize.
            execute_baseline_first: Whether to run baseline first (default True).

        Returns:
            Dictionary mapping task_id to TaskResult for all tasks.
        """
        self._tasks = tasks
        self._results = {}
        self._failed_tasks = []

        logger.info(f"Starting FAMA pipeline on {len(tasks)} tasks")

        # Stage 1: Execute baseline on all tasks
        if execute_baseline_first:
            logger.info("Stage 1: Executing baseline agents")
            for task in tasks:
                result = self._execute_baseline(task)
                self._results[task.task_id] = result
                if not result.success:
                    self._failed_tasks.append(result)

            logger.info(f"Baseline execution complete. {len(self._failed_tasks)} failures")

        # Stage 2: Analyze failures
        if self._failed_tasks:
            logger.info("Stage 2: Analyzing failures")
            self._error_analysis = self.analyze_failures(self._failed_tasks)

            # Stage 3: Orchestrate and mitigate for each failed task
            logger.info("Stage 3: Orchestrating and selecting mitigation agents")
            for task_result in self._failed_tasks:
                task_id = task_result.task_id

                # Orchestrate to identify primary errors
                orchestrator_result = self.orchestrator.orchestrate(
                    self._error_analysis[task_id],
                    task_result.trajectory,
                )
                self._orchestrator_results[task_id] = orchestrator_result

                # Mitigate to select optimal agent subset
                mitigation_result = self.mitigation_agent.mitigate(
                    orchestrator_result.primary_errors,
                    self.agent_pool,
                )
                self._mitigation_results[task_id] = mitigation_result

                logger.info(
                    f"Task {task_id}: Primary errors: {[e.value for e in orchestrator_result.primary_errors]}, "
                    f"Selected agents: {[a.value for a in mitigation_result.selected_agents]}"
                )

        logger.info("FAMA pipeline complete")
        return self._results

    def _execute_baseline(self, task: Task) -> TaskResult:
        """
        Execute a task using the baseline agent.

        If a baseline_executor was provided, use it. Otherwise, use a
        default execution strategy.

        Args:
            task: The task to execute.

        Returns:
            TaskResult containing the trajectory and success status.
        """
        if self.baseline_executor is not None:
            return self.baseline_executor(task)

        # Default execution: simulate a simple execution
        trajectory = self._simulate_execution(task)
        success = self._evaluate_success(task, trajectory)

        return TaskResult(
            task_id=task.task_id,
            trajectory=trajectory,
            success=success,
            error_categories=[] if success else self._infer_error_categories(trajectory),
            metadata={"execution_mode": "baseline"},
        )

    def _simulate_execution(self, task: Task) -> Trajectory:
        """
        Simulate a baseline execution for a task.

        This is a placeholder that creates a simple trajectory.
        In practice, this would be replaced with actual agent execution.

        Args:
            task: The task to simulate.

        Returns:
            Trajectory representing the execution.
        """
        # Create a simple trajectory with the task
        turn = Turn(
            user_message=task.description,
            assistant_message=f"Processing task: {task.task_id}",
            tool_calls=[],
            tool_results=[],
        )
        return [turn]

    def _evaluate_success(self, task: Task, trajectory: Trajectory) -> bool:
        """
        Evaluate whether a trajectory successfully completes a task.

        Args:
            task: The task that was executed.
            trajectory: The execution trajectory.

        Returns:
            True if the task was completed successfully, False otherwise.
        """
        # Default evaluation: check if we have a reasonable trajectory
        if not trajectory:
            return False

        # Check if the final response indicates success
        final_turn = trajectory[-1]
        success_indicators = ["complete", "done", "finished", "success"]
        response_lower = final_turn.assistant_message.lower()

        return any(indicator in response_lower for indicator in success_indicators)

    def _infer_error_categories(self, trajectory: Trajectory) -> list[ErrorCategory]:
        """
        Infer potential error categories from a trajectory.

        This is a placeholder for more sophisticated error inference.

        Args:
            trajectory: The execution trajectory.

        Returns:
            List of potentially relevant error categories.
        """
        # Default: return all categories as potential issues
        return list(ErrorCategory)

    def analyze_failures(
        self,
        failed_tasks: list[TaskResult],
    ) -> dict[str, list[ErrorAnalysisResult]]:
        """
        Analyze failed tasks using |E|=4 independent error-analysis agents.

        For each failed task, runs all four error category analyzers and
        concatenates their outputs.

        Args:
            failed_tasks: List of TaskResults that represent failures.

        Returns:
            Dictionary mapping task_id to list of ErrorAnalysisResults.
        """
        results: dict[str, list[ErrorAnalysisResult]] = {}

        for task_result in failed_tasks:
            logger.debug(f"Analyzing failures for task {task_result.task_id}")
            results[task_result.task_id] = self._stage1_analyze(task_result)

        return results

    def _stage1_analyze(self, task_result: TaskResult) -> list[ErrorAnalysisResult]:
        """
        Run Stage 1 analysis on a failed task.

        Executes all |E|=4 error-analysis agents in parallel and concatenates outputs.

        Args:
            task_result: The failed task result to analyze.

        Returns:
            List of ErrorAnalysisResults, one per error category.
        """
        results: list[ErrorAnalysisResult] = []

        # Run analysis for each error category
        for error_category in ErrorCategory:
            logger.debug(
                f"Analyzing {error_category.value} for task {task_result.task_id}"
            )
            analysis_result = self.error_analyzer.analyze(
                task_result.trajectory,
                error_category,
            )
            results.append(analysis_result)

        return results

    def select_agents(
        self,
        error_results: dict[str, list[ErrorAnalysisResult]],
    ) -> dict[str, list[AgentType]]:
        """
        Select optimal agent subsets based on error analysis results.

        For each task, uses the orchestrator to identify primary errors and
        the mitigation agent to select the minimal agent subset.

        Args:
            error_results: Dictionary mapping task_id to ErrorAnalysisResults.

        Returns:
            Dictionary mapping task_id to list of selected AgentTypes.
        """
        selections: dict[str, list[AgentType]] = {}

        for task_id, error_analysis in error_results.items():
            # Note: This requires trajectory which we don't have here
            # In practice, we'd look up the task result
            orchestrator_result = self.orchestrator.orchestrate(
                error_analysis,
                [],  # trajectory not used in current implementation
            )

            mitigation_result = self.mitigation_agent.mitigate(
                orchestrator_result.primary_errors,
                self.agent_pool,
            )

            selections[task_id] = mitigation_result.selected_agents

        return selections

    def _stage2_select(
        self,
        task_id: str,
        error_results: dict[str, list[ErrorAnalysisResult]],
    ) -> list[AgentType]:
        """
        Run Stage 2 agent selection for a task.

        Args:
            task_id: The task identifier.
            error_results: Error analysis results.

        Returns:
            List of selected AgentTypes.
        """
        # Get error analysis for this task
        error_analysis = error_results.get(task_id, [])

        # Orchestrate to identify primary errors
        # Note: In practice, we'd pass the actual trajectory
        orchestrator_result = self.orchestrator.orchestrate(
            error_analysis,
            [],
        )

        # Mitigate to select optimal agent subset
        mitigation_result = self.mitigation_agent.mitigate(
            orchestrator_result.primary_errors,
            self.agent_pool,
        )

        return mitigation_result.selected_agents

    def get_mitigation_agents_for_task(self, task_id: str) -> list[SpecializedAgent]:
        """
        Get the list of mitigation agents selected for a specific task.

        Args:
            task_id: The task identifier.

        Returns:
            List of SpecializedAgent instances for the task.
        """
        if task_id not in self._mitigation_results:
            return []

        selected_types = self._mitigation_results[task_id].selected_agents
        return [self.agent_pool[agent_type] for agent_type in selected_types]

    def get_error_analysis_for_task(
        self,
        task_id: str,
    ) -> Optional[list[ErrorAnalysisResult]]:
        """
        Get error analysis results for a specific task.

        Args:
            task_id: The task identifier.

        Returns:
            List of ErrorAnalysisResults or None if not available.
        """
        return self._error_analysis.get(task_id)

    def get_orchestrator_result_for_task(
        self,
        task_id: str,
    ) -> Optional[OrchestratorResult]:
        """
        Get orchestrator result for a specific task.

        Args:
            task_id: The task identifier.

        Returns:
            OrchestratorResult or None if not available.
        """
        return self._orchestrator_results.get(task_id)

    def get_mitigation_result_for_task(
        self,
        task_id: str,
    ) -> Optional[MitigationResult]:
        """
        Get mitigation result for a specific task.

        Args:
            task_id: The task identifier.

        Returns:
            MitigationResult or None if not available.
        """
        return self._mitigation_results.get(task_id)


# Type alias for Task
class Task:
    """Represents a task to be executed by the agent."""

    def __init__(
        self,
        task_id: str,
        description: str,
        domain: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a task.

        Args:
            task_id: Unique identifier for the task.
            description: Natural language description of the task.
            domain: Optional domain context (e.g., 'airline', 'retail').
            metadata: Additional metadata about the task.
        """
        self.task_id = task_id
        self.description = description
        self.domain = domain
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "domain": self.domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create a Task from a dictionary representation."""
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            domain=data.get("domain"),
            metadata=data.get("metadata", {}),
        )
