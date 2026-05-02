"""
Unit tests for the FAMA Engine.

Tests the FAMAEngine class which implements the core two-stage
failure-aware meta-agentic optimization pipeline.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Any

from fama.core.engine import FAMAEngine, Task
from fama.core.types import (
    AgentType,
    ErrorCategory,
    ErrorAnalysisResult,
    OrchestratorResult,
    MitigationResult,
    TaskResult,
    Turn,
)


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "{}"):
        self.response = response
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        self.call_count += 1
        return self.response


class TestTask:
    """Tests for Task class."""

    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            task_id="test-1",
            description="Book a flight from NYC to LA",
            domain="airline",
        )
        assert task.task_id == "test-1"
        assert task.description == "Book a flight from NYC to LA"
        assert task.domain == "airline"
        assert task.metadata == {}

    def test_task_with_metadata(self):
        """Test task creation with metadata."""
        metadata = {"priority": "high", "customer_id": "12345"}
        task = Task(
            task_id="test-2",
            description="Cancel booking",
            metadata=metadata,
        )
        assert task.metadata == metadata

    def test_task_to_dict(self):
        """Test task serialization to dict."""
        task = Task(
            task_id="test-3",
            description="Change flight date",
            domain="airline",
            metadata={"key": "value"},
        )
        result = task.to_dict()
        assert result["task_id"] == "test-3"
        assert result["description"] == "Change flight date"
        assert result["domain"] == "airline"
        assert result["metadata"] == {"key": "value"}

    def test_task_from_dict(self):
        """Test task deserialization from dict."""
        data = {
            "task_id": "test-4",
            "description": "Refund request",
            "domain": "retail",
            "metadata": {"order_id": "abc123"},
        }
        task = Task.from_dict(data)
        assert task.task_id == "test-4"
        assert task.description == "Refund request"
        assert task.domain == "retail"
        assert task.metadata == {"order_id": "abc123"}


class TestFAMAEngine:
    """Tests for FAMAEngine class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return MockLLMClient()

    @pytest.fixture
    def mock_agent_pool(self):
        """Create a mock agent pool."""
        pool = {}
        for agent_type in AgentType:
            mock_agent = Mock()
            mock_agent.agent_type = agent_type
            pool[agent_type] = mock_agent
        return pool

    @pytest.fixture
    def engine(self, mock_llm_client, mock_agent_pool):
        """Create a FAMAEngine instance for testing."""
        return FAMAEngine(
            llm_client=mock_llm_client,
            agent_pool=mock_agent_pool,
        )

    def test_engine_initialization(self, engine, mock_llm_client, mock_agent_pool):
        """Test engine initializes with correct components."""
        assert engine.llm_client == mock_llm_client
        assert engine.agent_pool == mock_agent_pool
        assert engine.error_analyzer is not None
        assert engine.orchestrator is not None
        assert engine.mitigation_agent is not None

    def test_engine_initialization_with_baseline_executor(self, mock_llm_client, mock_agent_pool):
        """Test engine initialization with custom baseline executor."""
        baseline_fn = Mock(return_value=Mock(success=True))
        engine = FAMAEngine(
            llm_client=mock_llm_client,
            agent_pool=mock_agent_pool,
            baseline_executor=baseline_fn,
        )
        assert engine.baseline_executor is not None

    def test_execute_baseline_with_custom_executor(self, mock_llm_client, mock_agent_pool):
        """Test baseline execution with custom executor."""
        mock_result = TaskResult(
            task_id="test-task",
            trajectory=[],
            success=True,
        )
        baseline_fn = Mock(return_value=mock_result)
        engine = FAMAEngine(
            llm_client=mock_llm_client,
            agent_pool=mock_agent_pool,
            baseline_executor=baseline_fn,
        )

        task = Task(task_id="test-task", description="Test task")
        result = engine._execute_baseline(task)

        baseline_fn.assert_called_once_with(task)
        assert result.success is True

    def test_successful_task_evaluation(self, engine):
        """Test evaluation of a successful task trajectory."""
        task = Task(task_id="test-1", description="Complete the task successfully")
        trajectory = [
            Turn(
                user_message="Do X",
                assistant_message="I have completed X successfully.",
            )
        ]
        assert engine._evaluate_success(task, trajectory) is True

    def test_failed_task_evaluation(self, engine):
        """Test evaluation of a failed task trajectory."""
        task = Task(task_id="test-2", description="Process the request")
        trajectory = [
            Turn(
                user_message="Do Y",
                assistant_message="I'm not sure how to proceed.",
            )
        ]
        assert engine._evaluate_success(task, trajectory) is False

    def test_empty_trajectory_evaluation(self, engine):
        """Test evaluation of empty trajectory returns False."""
        task = Task(task_id="test-3", description="Empty test")
        assert engine._evaluate_success(task, []) is False

    def test_infer_error_categories_returns_all_by_default(self, engine):
        """Test that default error inference returns all categories."""
        trajectory = [Turn(user_message="test", assistant_message="test")]
        errors = engine._infer_error_categories(trajectory)
        assert len(errors) == len(ErrorCategory)
        assert all(e in errors for e in ErrorCategory)

    def test_simulate_execution_creates_trajectory(self, engine):
        """Test that simulation creates a valid trajectory."""
        task = Task(task_id="sim-test", description="Simulate this task")
        trajectory = engine._simulate_execution(task)
        assert len(trajectory) == 1
        assert trajectory[0].user_message == task.description

    def test_run_with_no_tasks(self, engine):
        """Test running engine with empty task list."""
        results = engine.run([])
        assert results == {}

    def test_run_tracks_tasks_and_results(self, engine):
        """Test that run() properly tracks tasks and results."""
        tasks = [
            Task(task_id="task-1", description="First task"),
            Task(task_id="task-2", description="Second task"),
        ]
        results = engine.run(tasks, execute_baseline_first=True)
        assert len(results) == 2
        assert "task-1" in results
        assert "task-2" in results

    def test_run_identifies_failed_tasks(self, engine):
        """Test that failed tasks are properly identified."""
        tasks = [
            Task(task_id="success-task", description="Complete the task done"),
            Task(task_id="fail-task", description="Try this but uncertain"),
        ]
        engine.run(tasks, execute_baseline_first=True)
        assert len(engine._failed_tasks) == 1
        assert engine._failed_tasks[0].task_id == "fail-task"

    def test_get_mitigation_agents_for_task(self, engine):
        """Test retrieving mitigation agents for a specific task."""
        # Setup: Add a mitigation result
        engine._mitigation_results["task-1"] = MitigationResult(
            selected_agents=[AgentType.DCE, AgentType.TSA],
            reasoning="Selected DCE and TSA",
        )
        agents = engine.get_mitigation_agents_for_task("task-1")
        assert len(agents) == 2

    def test_get_mitigation_agents_for_unknown_task(self, engine):
        """Test that unknown task returns empty list."""
        agents = engine.get_mitigation_agents_for_task("unknown-task")
        assert agents == []

    def test_get_error_analysis_for_task(self, engine):
        """Test retrieving error analysis for a task."""
        mock_analysis = [Mock(spec=ErrorAnalysisResult)]
        engine._error_analysis["task-1"] = mock_analysis
        result = engine.get_error_analysis_for_task("task-1")
        assert result == mock_analysis

    def test_get_error_analysis_for_unknown_task(self, engine):
        """Test that unknown task returns None for error analysis."""
        result = engine.get_error_analysis_for_task("unknown-task")
        assert result is None

    def test_get_orchestrator_result_for_task(self, engine):
        """Test retrieving orchestrator result for a task."""
        mock_result = OrchestratorResult(
            primary_errors=[ErrorCategory.DCV],
            reasoning="DCV detected",
        )
        engine._orchestrator_results["task-1"] = mock_result
        result = engine.get_orchestrator_result_for_task("task-1")
        assert result == mock_result

    def test_get_mitigation_result_for_task(self, engine):
        """Test retrieving mitigation result for a task."""
        mock_result = MitigationResult(
            selected_agents=[AgentType.PLANNER],
            reasoning="Planner selected",
        )
        engine._mitigation_results["task-1"] = mock_result
        result = engine.get_mitigation_result_for_task("task-1")
        assert result == mock_result

    def test_analyze_failures_empty_list(self, engine):
        """Test analyzing empty failure list."""
        results = engine.analyze_failures([])
        assert results == {}

    def test_analyze_failures_returns_dict(self, engine):
        """Test that analyze_failures returns proper dict structure."""
        failed_result = TaskResult(
            task_id="failed-1",
            trajectory=[Turn(user_message="test", assistant_message="fail")],
            success=False,
        )
        results = engine.analyze_failures([failed_result])
        assert "failed-1" in results
        assert len(results["failed-1"]) == len(ErrorCategory)


class TestFAMAEngineSelectAgents:
    """Tests for agent selection functionality."""

    @pytest.fixture
    def engine_with_mocks(self):
        """Create engine with fully mocked dependencies."""
        mock_llm = MockLLMClient()
        mock_pool = {at: Mock() for at in AgentType}
        engine = FAMAEngine(
            llm_client=mock_llm,
            agent_pool=mock_pool,
        )
        return engine

    def test_select_agents_returns_dict(self, engine_with_mocks):
        """Test that select_agents returns a dict mapping task_id to agents."""
        error_results = {
            "task-1": [
                ErrorAnalysisResult(ErrorCategory.DCV, True, "DCV rationale"),
                ErrorAnalysisResult(ErrorCategory.WRCO, False, "WRCO rationale"),
                ErrorAnalysisResult(ErrorCategory.CM, True, "CM rationale"),
                ErrorAnalysisResult(ErrorCategory.IFU, False, "IFU rationale"),
            ],
        }
        # Mock the orchestrator and mitigation
        with patch.object(engine_with_mocks.orchestrator, 'orchestrate') as mock_orch:
            mock_orch.return_value = OrchestratorResult(
                primary_errors=[ErrorCategory.DCV, ErrorCategory.CM],
                reasoning="Test reasoning",
            )
            with patch.object(engine_with_mocks.mitigation_agent, 'mitigate') as mock_mit:
                mock_mit.return_value = MitigationResult(
                    selected_agents=[AgentType.PLANNER],
                    reasoning="Test mitigation",
                )
                result = engine_with_mocks.select_agents(error_results)
                assert "task-1" in result
