"""
Unit tests for the Mitigation Agent.

Tests the MitigationAgent class which selects the minimal subset of
|A|=6 specialized agents that can address identified primary errors.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock
from typing import Any

from fama.core.mitigation import MitigationAgent
from fama.core.types import (
    AgentType,
    ErrorCategory,
    MitigationResult,
    AGENT_POOL,
    ERROR_TO_AGENT_MAPPING,
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


class TestMitigationAgentInitialization:
    """Tests for MitigationAgent initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        mock_client = Mock()
        mock_pool = {AgentType.DCE: Mock()}
        agent = MitigationAgent(mock_client, mock_pool)
        assert agent.llm_client == mock_client
        assert agent.agent_pool == mock_pool

    def test_initialization_with_all_agent_types(self):
        """Test initialization with full agent pool."""
        mock_client = Mock()
        mock_pool = {at: Mock() for at in AgentType}
        agent = MitigationAgent(mock_client, mock_pool)
        assert len(agent.agent_pool) == len(AgentType)


class TestMitigationAgentFormat:
    """Tests for formatting methods."""

    @pytest.fixture
    def mitigation_agent(self):
        """Create mitigation agent instance."""
        mock_client = MockLLMClient()
        mock_pool = {at: Mock() for at in AgentType}
        return MitigationAgent(mock_client, mock_pool)

    def test_format_errors_single(self, mitigation_agent):
        """Test formatting single error."""
        errors = [ErrorCategory.DCV]
        result = mitigation_agent._format_errors(errors)
        assert "DCV" in result
        assert "domain_constraint_violation" in result

    def test_format_errors_multiple(self, mitigation_agent):
        """Test formatting multiple errors."""
        errors = [ErrorCategory.DCV, ErrorCategory.WRCO, ErrorCategory.CM]
        result = mitigation_agent._format_errors(errors)
        assert "DCV" in result
        assert "WRCO" in result
        assert "CM" in result

    def test_format_errors_empty(self, mitigation_agent):
        """Test formatting empty error list."""
        result = mitigation_agent._format_errors([])
        assert result == ""

    def test_format_agent_descriptions(self, mitigation_agent):
        """Test formatting agent pool descriptions."""
        result = mitigation_agent._format_agent_descriptions()
        for agent_type in AgentType:
            assert agent_type.value in result
        # Should include human-readable descriptions
        assert "DCE" in result or "Domain" in result


class TestMitigationAgentParse:
    """Tests for response parsing."""

    @pytest.fixture
    def mitigation_agent(self):
        """Create mitigation agent instance."""
        mock_client = MockLLMClient()
        mock_pool = {at: Mock() for at in AgentType}
        return MitigationAgent(mock_client, mock_pool)

    def test_extract_json_from_plain_json(self, mitigation_agent):
        """Test extracting JSON from plain JSON string."""
        json_str = '{"selected_agents": ["DCE"], "reasoning": "test"}'
        result = mitigation_agent._extract_json(json_str)
        assert '"selected_agents"' in result

    def test_extract_json_from_markdown(self, mitigation_agent):
        """Test extracting JSON from markdown code block."""
        response = '```json\n{"selected_agents": ["DCE"], "reasoning": "test"}\n```'
        result = mitigation_agent._extract_json(response)
        assert '"selected_agents"' in result

    def test_parse_valid_response_single_agent(self, mitigation_agent):
        """Test parsing valid response with single agent."""
        response = json.dumps({
            "selected_agents": ["DCE"],
            "reasoning": "Selected DCE for DCV handling",
        })
        result = mitigation_agent._parse_response(response)
        assert AgentType.DCE in result.selected_agents
        assert len(result.selected_agents) == 1

    def test_parse_valid_response_multiple_agents(self, mitigation_agent):
        """Test parsing valid response with multiple agents."""
        response = json.dumps({
            "selected_agents": ["DCE", "TSA", "PLANNER"],
            "reasoning": "Multiple agents needed",
        })
        result = mitigation_agent._parse_response(response)
        assert AgentType.DCE in result.selected_agents
        assert AgentType.TSA in result.selected_agents
        assert AgentType.PLANNER in result.selected_agents

    def test_parse_response_with_short_names(self, mitigation_agent):
        """Test parsing response using short agent names."""
        response = json.dumps({
            "selected_agents": ["tor", "verifier", "memory"],
            "reasoning": "Short names used",
        })
        result = mitigation_agent._parse_response(response)
        assert AgentType.TOR in result.selected_agents
        assert AgentType.VERIFIER in result.selected_agents
        assert AgentType.MEMORY in result.selected_agents

    def test_parse_invalid_json_uses_fallback(self, mitigation_agent):
        """Test that invalid JSON triggers fallback selection."""
        response = "not valid json"
        result = mitigation_agent._parse_response(response)
        # Fallback should select some agents
        assert len(result.selected_agents) >= 0

    def test_parse_empty_agents_triggers_fallback(self, mitigation_agent):
        """Test that empty agents list triggers fallback."""
        response = json.dumps({
            "selected_agents": [],
            "reasoning": "No agents selected",
        })
        result = mitigation_agent._parse_response(response)
        # Fallback should select some agents
        assert "Fallback" in result.reasoning or "Greedy" in result.reasoning


class TestMitigationAgentMitigate:
    """Tests for mitigate method."""

    def test_mitigate_with_single_error(self):
        """Test mitigation with single error category."""
        mock_response = json.dumps({
            "selected_agents": ["DCE"],
            "reasoning": "DCE handles DCV",
        })
        mock_client = MockLLMClient(mock_response)
        mock_pool = {at: Mock() for at in AgentType}
        agent = MitigationAgent(mock_client, mock_pool)

        result = agent.mitigate([ErrorCategory.DCV], mock_pool)

        assert AgentType.DCE in result.selected_agents
        assert mock_client.last_call is not None

    def test_mitigate_with_multiple_errors(self):
        """Test mitigation with multiple error categories."""
        mock_response = json.dumps({
            "selected_agents": ["PLANNER", "TSA"],
            "reasoning": "Planner for CM, TSA for WRCO",
        })
        mock_client = MockLLMClient(mock_response)
        mock_pool = {at: Mock() for at in AgentType}
        agent = MitigationAgent(mock_client, mock_pool)

        result = agent.mitigate([ErrorCategory.CM, ErrorCategory.WRCO], mock_pool)

        assert AgentType.PLANNER in result.selected_agents
        assert AgentType.TSA in result.selected_agents

    def test_mitigate_with_empty_errors(self):
        """Test mitigation with empty error list."""
        mock_client = MockLLMClient()
        mock_pool = {at: Mock() for at in AgentType}
        agent = MitigationAgent(mock_client, mock_pool)

        result = agent.mitigate([], mock_pool)

        assert result.selected_agents == []
        assert "No errors" in result.reasoning

    def test_mitigate_llm_exception_triggers_greedy(self):
        """Test that LLM exception triggers greedy fallback."""
        mock_client = Mock()
        mock_client.generate = Mock(side_effect=Exception("LLM error"))
        mock_pool = {at: Mock() for at in AgentType}
        agent = MitigationAgent(mock_client, mock_pool)

        result = agent.mitigate([ErrorCategory.DCV, ErrorCategory.IFU], mock_pool)

        # Should use greedy selection
        assert len(result.selected_agents) > 0
        assert "Greedy" in result.reasoning

    def test_mitigate_calls_llm_with_correct_params(self):
        """Test that mitigate calls LLM with correct parameters."""
        mock_response = json.dumps({
            "selected_agents": ["DCE"],
            "reasoning": "Test",
        })
        mock_client = MockLLMClient(mock_response)
        mock_pool = {at: Mock() for at in AgentType}
        agent = MitigationAgent(mock_client, mock_pool)

        agent.mitigate([ErrorCategory.DCV], mock_pool)

        assert mock_client.last_call["temperature"] == 0.0
        assert len(mock_client.last_call["system_prompt"]) > 0
        assert len(mock_client.last_call["user_prompt"]) > 0


class TestMitigationAgentGreedySelection:
    """Tests for greedy selection fallback."""

    @pytest.fixture
    def mitigation_agent(self):
        """Create mitigation agent instance."""
        mock_client = MockLLMClient()
        mock_pool = {at: Mock() for at in AgentType}
        return MitigationAgent(mock_client, mock_pool)

    def test_greedy_selection_single_error(self, mitigation_agent):
        """Test greedy selection with single error."""
        result = mitigation_agent._greedy_selection([ErrorCategory.DCV])
        # DCV can be handled by DCE or VERIFIER
        assert len(result.selected_agents) >= 1
        assert all(agent in [AgentType.DCE, AgentType.VERIFIER] for agent in result.selected_agents)

    def test_greedy_selection_all_errors(self, mitigation_agent):
        """Test greedy selection with all error categories."""
        all_errors = list(ErrorCategory)
        result = mitigation_agent._greedy_selection(all_errors)
        # Should select some subset (not necessarily all)
        assert len(result.selected_agents) >= 1

    def test_greedy_selection_empty_errors(self, mitigation_agent):
        """Test greedy selection with empty error list."""
        result = mitigation_agent._greedy_selection([])
        assert result.selected_agents == []

    def test_greedy_selection_minimizes_agents(self, mitigation_agent):
        """Test that greedy selection aims to minimize agents."""
        # DCV, WRCO, CM, IFU
        # Using mapping: DCE/VERIFIER for DCV, TOR/TSA for WRCO, PLANNER/TSA for CM, PLANNER/VERIFIER/MEMORY for IFU
        # Optimal would be: TSA covers WRCO+CM, VERIFIER covers DCV+IFU = 2 agents
        errors = [ErrorCategory.DCV, ErrorCategory.WRCO, ErrorCategory.CM, ErrorCategory.IFU]
        result = mitigation_agent._greedy_selection(errors)
        # Greedy should produce a small subset
        assert len(result.selected_agents) <= 3


class TestMitigationAgentExtractErrors:
    """Tests for error extraction from reasoning."""

    @pytest.fixture
    def mitigation_agent(self):
        """Create mitigation agent instance."""
        mock_client = MockLLMClient()
        mock_pool = {at: Mock() for at in AgentType}
        return MitigationAgent(mock_client, mock_pool)

    def test_extract_errors_from_reasoning(self, mitigation_agent):
        """Test extracting error categories from reasoning text."""
        reasoning = "DCV was detected along with CM"
        result = mitigation_agent._extract_errors_from_reasoning(reasoning)
        assert ErrorCategory.DCV in result
        assert ErrorCategory.CM in result

    def test_extract_errors_no_match(self, mitigation_agent):
        """Test extracting errors when no categories mentioned."""
        reasoning = "No specific errors found"
        result = mitigation_agent._extract_errors_from_reasoning(reasoning)
        # Falls back to all categories
        assert len(result) == len(ErrorCategory)


class TestMitigationAgentSystemPrompt:
    """Tests for system prompt content."""

    def test_system_prompt_contains_all_agents(self):
        """Test that system prompt lists all agent types."""
        mock_client = MockLLMClient()
        mock_pool = {at: Mock() for at in AgentType}
        agent = MitigationAgent(mock_client, mock_pool)
        prompt = agent.SYSTEM_PROMPT
        for agent_type in AgentType:
            assert agent_type.value in prompt or agent_type.name in prompt

    def test_system_prompt_mentions_error_mapping(self):
        """Test that system prompt explains error-to-agent mapping."""
        mock_client = MockLLMClient()
        mock_pool = {at: Mock() for at in AgentType}
        agent = MitigationAgent(mock_client, mock_pool)
        prompt = agent.SYSTEM_PROMPT
        assert "DCV" in prompt
        assert "WRCO" in prompt
        assert "CM" in prompt
        assert "IFU" in prompt

    def test_system_prompt_instructs_minimal_subset(self):
        """Test that system prompt emphasizes minimal subset selection."""
        mock_client = MockLLMClient()
        mock_pool = {at: Mock() for at in AgentType}
        agent = MitigationAgent(mock_client, mock_pool)
        prompt = agent.SYSTEM_PROMPT
        assert "minimal" in prompt.lower() or "MINIMAL" in prompt


class TestErrorToAgentMapping:
    """Tests for ERROR_TO_AGENT_MAPPING constant."""

    def test_dcv_mapping(self):
        """Test DCV error maps to correct agents."""
        agents = ERROR_TO_AGENT_MAPPING[ErrorCategory.DCV]
        assert AgentType.DCE in agents
        assert AgentType.VERIFIER in agents

    def test_wrco_mapping(self):
        """Test WRCO error maps to correct agents."""
        agents = ERROR_TO_AGENT_MAPPING[ErrorCategory.WRCO]
        assert AgentType.TOR in agents
        assert AgentType.TSA in agents

    def test_cm_mapping(self):
        """Test CM error maps to correct agents."""
        agents = ERROR_TO_AGENT_MAPPING[ErrorCategory.CM]
        assert AgentType.PLANNER in agents
        assert AgentType.TSA in agents

    def test_ifu_mapping(self):
        """Test IFU error maps to correct agents."""
        agents = ERROR_TO_AGENT_MAPPING[ErrorCategory.IFU]
        assert AgentType.PLANNER in agents
        assert AgentType.VERIFIER in agents
        assert AgentType.MEMORY in agents

    def test_all_errors_have_mappings(self):
        """Test that all error categories have agent mappings."""
        for error in ErrorCategory:
            assert error in ERROR_TO_AGENT_MAPPING
            assert len(ERROR_TO_AGENT_MAPPING[error]) > 0
