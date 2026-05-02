"""
Unit tests for the Orchestrator.

Tests the Orchestrator class which identifies primary error(s) causing
task failure by analyzing concatenated outputs from |E|=4 error-analysis agents.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock
from typing import Any

from fama.core.orchestrator import Orchestrator
from fama.core.types import (
    ErrorCategory,
    ErrorAnalysisResult,
    OrchestratorResult,
    Trajectory,
    Turn,
)


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "{}"):
        self.response = response
        self.last_call = None

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        self.last_call = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
        }
        return self.response


class TestOrchestratorInitialization:
    """Tests for Orchestrator initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        mock_client = Mock()
        orchestrator = Orchestrator(mock_client)
        assert orchestrator.llm_client == mock_client

    def test_initialization_accepts_any_llm_client(self):
        """Test that any object can be used as LLM client."""
        mock_client = MagicMock()
        orchestrator = Orchestrator(mock_client)
        assert orchestrator.llm_client is mock_client


class TestOrchestratorFormat:
    """Tests for formatting methods."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return Orchestrator(MockLLMClient())

    def test_format_empty_trajectory(self, orchestrator):
        """Test formatting empty trajectory."""
        result = orchestrator._format_trajectory([])
        assert "No trajectory available" in result

    def test_format_trajectory_single_turn(self, orchestrator):
        """Test formatting trajectory with single turn."""
        trajectory = [
            Turn(
                user_message="Book a flight",
                assistant_message="I'll book a flight for you",
            )
        ]
        result = orchestrator._format_trajectory(trajectory)
        assert "Book a flight" in result
        assert "I'll book a flight for you" in result

    def test_format_trajectory_multiple_turns(self, orchestrator):
        """Test formatting trajectory with multiple turns."""
        trajectory = [
            Turn(user_message="First message", assistant_message="First response"),
            Turn(user_message="Second message", assistant_message="Second response"),
        ]
        result = orchestrator._format_trajectory(trajectory)
        assert "Turn 1" in result
        assert "Turn 2" in result

    def test_format_trajectory_with_tool_calls(self, orchestrator):
        """Test formatting trajectory with tool calls."""
        trajectory = [
            Turn(
                user_message="Search for flights",
                assistant_message="Searching...",
                tool_calls=[{"name": "search_flights", "arguments": {"from": "NYC"}}],
            )
        ]
        result = orchestrator._format_trajectory(trajectory)
        assert "search_flights" in result

    def test_format_trajectory_with_tool_results(self, orchestrator):
        """Test formatting trajectory with tool results."""
        trajectory = [
            Turn(
                user_message="Search",
                assistant_message="Found results",
                tool_results=['[{"flight": "AA123", "price": 500}]'],
            )
        ]
        result = orchestrator._format_trajectory(trajectory)
        assert "flight" in result

    def test_format_analysis_results_empty(self, orchestrator):
        """Test formatting empty analysis results."""
        result = orchestrator._format_analysis_results([])
        assert result == ""

    def test_format_analysis_results_single(self, orchestrator):
        """Test formatting single analysis result."""
        analysis_results = [
            ErrorAnalysisResult(
                error_category=ErrorCategory.DCV,
                detected=True,
                rationale="Domain constraint was violated",
            )
        ]
        result = orchestrator._format_analysis_results(analysis_results)
        assert "DCV" in result
        assert "Domain constraint was violated" in result
        assert "Detected: True" in result

    def test_format_analysis_results_multiple(self, orchestrator):
        """Test formatting multiple analysis results."""
        analysis_results = [
            ErrorAnalysisResult(ErrorCategory.DCV, True, "DCV rationale"),
            ErrorAnalysisResult(ErrorCategory.WRCO, False, "WRCO rationale"),
            ErrorAnalysisResult(ErrorCategory.CM, True, "CM rationale"),
            ErrorAnalysisResult(ErrorCategory.IFU, False, "IFU rationale"),
        ]
        result = orchestrator._format_analysis_results(analysis_results)
        for cat in ErrorCategory:
            assert cat.value in result or cat.name_short in result


class TestOrchestratorParse:
    """Tests for response parsing methods."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return Orchestrator(MockLLMClient())

    def test_extract_json_from_plain_json(self, orchestrator):
        """Test extracting JSON from plain JSON string."""
        json_str = '{"primary_errors": ["DCV"], "reasoning": "test"}'
        result = orchestrator._extract_json(json_str)
        assert '"primary_errors"' in result

    def test_extract_json_from_markdown_code_block(self, orchestrator):
        """Test extracting JSON from markdown code block."""
        response = '```json\n{"primary_errors": ["DCV"], "reasoning": "test"}\n```'
        result = orchestrator._extract_json(response)
        assert '"primary_errors"' in result

    def test_extract_json_from_plain_code_block(self, orchestrator):
        """Test extracting JSON from plain code block."""
        response = '```\n{"primary_errors": ["DCV"], "reasoning": "test"}\n```'
        result = orchestrator._extract_json(response)
        assert '"primary_errors"' in result

    def test_parse_valid_response(self, orchestrator):
        """Test parsing a valid JSON response."""
        response = json.dumps({
            "primary_errors": ["DCV", "CM"],
            "reasoning": "DCV and CM were the primary errors",
        })
        result = orchestrator._parse_response(response)
        assert len(result.primary_errors) == 2
        assert ErrorCategory.DCV in result.primary_errors
        assert ErrorCategory.CM in result.primary_errors
        assert "DCV and CM were the primary errors" in result.reasoning

    def test_parse_response_with_short_names(self, orchestrator):
        """Test parsing response using short error names."""
        response = json.dumps({
            "primary_errors": ["WRCO", "IFU"],
            "reasoning": "Test reasoning",
        })
        result = orchestrator._parse_response(response)
        assert ErrorCategory.WRCO in result.primary_errors
        assert ErrorCategory.IFU in result.primary_errors

    def test_parse_invalid_json_fallback(self, orchestrator):
        """Test that invalid JSON falls back to all categories."""
        response = "not valid json at all"
        result = orchestrator._parse_response(response)
        # Fallback returns all error categories
        assert len(result.primary_errors) == len(ErrorCategory)

    def test_parse_empty_primary_errors_fallback(self, orchestrator):
        """Test that empty primary_errors falls back to all categories."""
        response = json.dumps({
            "primary_errors": [],
            "reasoning": "No errors identified",
        })
        result = orchestrator._parse_response(response)
        assert len(result.primary_errors) == len(ErrorCategory)


class TestOrchestratorOrchestrate:
    """Tests for orchestrate method."""

    def test_orchestrate_with_mock_llm(self):
        """Test orchestration with mocked LLM response."""
        mock_response = json.dumps({
            "primary_errors": ["DCV"],
            "reasoning": "DCV detected as primary error",
        })
        mock_client = MockLLMClient(mock_response)
        orchestrator = Orchestrator(mock_client)

        error_results = [
            ErrorAnalysisResult(ErrorCategory.DCV, True, "DCV rationale"),
        ]
        trajectory = [Turn(user_message="test", assistant_message="response")]

        result = orchestrator.orchestrate(error_results, trajectory)

        assert ErrorCategory.DCV in result.primary_errors
        assert mock_client.last_call is not None
        assert "system_prompt" in mock_client.last_call

    def test_orchestrate_empty_error_results(self):
        """Test orchestration with empty error results."""
        mock_response = json.dumps({
            "primary_errors": ["DCV"],
            "reasoning": "Test",
        })
        mock_client = MockLLMClient(mock_response)
        orchestrator = Orchestrator(mock_client)

        result = orchestrator.orchestrate([], [])

        # Should still call LLM and parse response
        assert result is not None

    def test_orchestrate_llm_exception_fallback(self):
        """Test that LLM exceptions trigger fallback behavior."""
        mock_client = Mock()
        mock_client.generate = Mock(side_effect=Exception("LLM error"))
        orchestrator = Orchestrator(mock_client)

        error_results = [
            ErrorAnalysisResult(ErrorCategory.DCV, True, "DCV rationale"),
        ]

        result = orchestrator.orchestrate(error_results, [])

        # Fallback should select detected errors
        assert ErrorCategory.DCV in result.primary_errors
        assert "Fallback" in result.reasoning or "error" in result.reasoning.lower()

    def test_orchestrate_all_detected_errors(self):
        """Test orchestration when all errors are detected."""
        mock_response = json.dumps({
            "primary_errors": ["DCV", "WRCO", "CM", "IFU"],
            "reasoning": "All errors detected",
        })
        mock_client = MockLLMClient(mock_response)
        orchestrator = Orchestrator(mock_client)

        error_results = [
            ErrorAnalysisResult(ErrorCategory.DCV, True, "DCV rationale"),
            ErrorAnalysisResult(ErrorCategory.WRCO, True, "WRCO rationale"),
            ErrorAnalysisResult(ErrorCategory.CM, True, "CM rationale"),
            ErrorAnalysisResult(ErrorCategory.IFU, True, "IFU rationale"),
        ]

        result = orchestrator.orchestrate(error_results, [])

        assert len(result.primary_errors) == 4
        assert all(cat in result.primary_errors for cat in ErrorCategory)

    def test_orchestrate_multiple_turn_trajectory(self):
        """Test orchestration with multi-turn trajectory."""
        mock_response = json.dumps({
            "primary_errors": ["CM"],
            "reasoning": "Contextual misinterpretation detected",
        })
        mock_client = MockLLMClient(mock_response)
        orchestrator = Orchestrator(mock_client)

        trajectory = [
            Turn(user_message="First", assistant_message="Response 1"),
            Turn(user_message="Second", assistant_message="Response 2"),
            Turn(user_message="Third", assistant_message="Response 3"),
        ]
        error_results = [
            ErrorAnalysisResult(ErrorCategory.CM, True, "CM rationale"),
        ]

        result = orchestrator.orchestrate(error_results, trajectory)

        assert ErrorCategory.CM in result.primary_errors
        # Verify trajectory was included in prompt
        assert "Turn 1" in mock_client.last_call["user_prompt"]
        assert "Turn 3" in mock_client.last_call["user_prompt"]


class TestOrchestratorSystemPrompt:
    """Tests for system prompt content."""

    def test_system_prompt_contains_error_categories(self):
        """Test that system prompt defines all error categories."""
        orchestrator = Orchestrator(MockLLMClient())
        prompt = orchestrator.SYSTEM_PROMPT
        assert "DCV" in prompt
        assert "WRCO" in prompt
        assert "CM" in prompt
        assert "IFU" in prompt

    def test_system_prompt_mentions_json_response(self):
        """Test that system prompt instructs JSON output."""
        orchestrator = Orchestrator(MockLLMClient())
        prompt = orchestrator.SYSTEM_PROMPT
        assert "JSON" in prompt

    def test_user_prompt_template_has_placeholders(self):
        """Test that user prompt template has required placeholders."""
        orchestrator = Orchestrator(MockLLMClient())
        template = orchestrator.USER_PROMPT_TEMPLATE
        assert "{trajectory_text}" in template
        assert "{analysis_results_text}" in template
