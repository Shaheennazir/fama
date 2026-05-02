"""
FAMA: Failure-Aware Meta-Agentic Framework

A meta-agentic framework that optimizes agent performance by dynamically
selecting specialized agents based on failure analysis.
"""

from __future__ import annotations

from fama.core.types import (
    ErrorCategory,
    AgentType,
    Turn,
    Trajectory,
    TaskResult,
    ErrorAnalysisResult,
    OrchestratorResult,
    MitigationResult,
    AGENT_POOL,
    ERROR_TO_AGENT_MAPPING,
    ERROR_CATEGORIES_METADATA,
)
from fama.core.engine import FAMAEngine
from fama.core.orchestrator import Orchestrator
from fama.core.mitigation import MitigationAgent

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ErrorCategory",
    "AgentType",
    "Turn",
    "Trajectory",
    "TaskResult",
    "ErrorAnalysisResult",
    "OrchestratorResult",
    "MitigationResult",
    "FAMAEngine",
    "Orchestrator",
    "MitigationAgent",
    "AGENT_POOL",
    "ERROR_TO_AGENT_MAPPING",
    "ERROR_CATEGORIES_METADATA",
]
