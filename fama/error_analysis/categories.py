"""
Error category definitions for FAMA.

This module contains the canonical definitions of the |E|=4 error categories
used in FAMA's failure analysis, including their definitions, causes, and
relationships to agent types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fama.core.types import ErrorCategory


@dataclass(frozen=True)
class ErrorCategoryDefinition:
    """
    Immutable definition of an error category.

    Attributes:
        category: The ErrorCategory enum value.
        name: Human-readable name.
        short_name: Abbreviated name.
        definition: Full definition of the error category.
        causes: List of common causes for this error type.
        examples: Optional list of example scenarios.
    """

    category: ErrorCategory
    name: str
    short_name: str
    definition: str
    causes: tuple[str, ...]
    examples: Optional[tuple[str, ...]] = None


# Canonical error category definitions
ERROR_CATEGORIES: dict[ErrorCategory, ErrorCategoryDefinition] = {
    ErrorCategory.DCV: ErrorCategoryDefinition(
        category=ErrorCategory.DCV,
        name="Domain Constraint Violation",
        short_name="DCV",
        definition=(
            "The agent performs an action that is forbidden by domain policy "
            "or skips a step that is required by the domain rules. This includes "
            "actions that violate stated constraints, skip mandatory verification "
            "steps, or fail to comply with domain-specific regulations."
        ),
        causes=(
            "Agent lacks awareness of domain-specific constraints",
            "Agent misunderstands policy boundaries",
            "Agent prioritizes task completion over policy compliance",
            "Insufficient guidance on allowed/disallowed actions",
            "Agent fails to recognize when a constraint applies",
            "Agent makes incorrect assumptions about domain rules",
        ),
        examples=(
            "Agent books a flight for a customer without verifying their ID "
            "when domain policy requires ID verification",
            "Agent processes a refund after the return window has closed",
            "Agent shares customer data with unauthorized third parties",
        ),
    ),
    ErrorCategory.WRCO: ErrorCategoryDefinition(
        category=ErrorCategory.WRCO,
        name="Wrong Retrieval from Complex Tool Outputs",
        short_name="WRCO",
        definition=(
            "The agent fails to correctly extract, parse, or utilize information "
            "from tool outputs that contain nested structures, lists, or multiple items. "
            "This includes missing relevant items in lists, failing to navigate nested "
            "data structures, extracting wrong fields, or ignoring multiple valid results."
        ),
        causes=(
            "Agent misses relevant items in lists",
            "Agent fails to navigate nested data structures",
            "Agent extracts wrong fields from complex outputs",
            "Agent ignores multiple valid results",
            "Agent assumes single-item responses when multiple items exist",
            "Agent doesn't properly iterate through list results",
        ),
        examples=(
            "Agent retrieves a flight booking but only extracts the first item "
            "when the user actually needed the second flight in a multi-leg journey",
            "Agent fails to find a customer's frequent flyer number because "
            "it's nested under a 'loyalty' key rather than directly accessible",
            "Agent processes the wrong item when a list of matching items is returned",
        ),
    ),
    ErrorCategory.CM: ErrorCategoryDefinition(
        category=ErrorCategory.CM,
        name="Contextual Misinterpretation",
        short_name="CM",
        definition=(
            "The agent understands the literal meaning of words but fails to grasp "
            "the actual user intent or contextual nuance. This includes taking "
            "instructions too literally, ignoring implicit constraints or preferences, "
            "failing to infer missing information from context, and not asking for "
            "clarification when needed."
        ),
        causes=(
            "Agent takes instructions too literally",
            "Agent ignores implicit constraints or preferences",
            "Agent fails to infer missing information from context",
            "Agent doesn't ask for clarification when needed",
            "Agent doesn't consider the broader context of the request",
            "Agent misinterprets vague or ambiguous instructions",
        ),
        examples=(
            "User says 'get me a flight to New York' and agent books to New York, "
            "NY when the user actually meant New York/Newark (EWR) based on their "
            "past preferences",
            "Agent processes a 'cancel' request literally without checking if "
            "cancellation is actually what the user wanted given the context "
            "of their previous message",
            "User asks to 'change' a booking without specifying change vs. cancel "
            "and agent assumes change when the user wanted cancellation",
        ),
    ),
    ErrorCategory.IFU: ErrorCategoryDefinition(
        category=ErrorCategory.IFU,
        name="Incomplete Fulfillment / Early Stopping",
        short_name="IFU",
        definition=(
            "The agent stops execution when encountering difficulty instead of trying "
            "alternative approaches or persisting to find a solution. This includes "
            "giving up too early when first attempt fails, not exploring alternative "
            "strategies, lacking confidence to try harder paths, and mistaking "
            "difficulty for impossibility."
        ),
        causes=(
            "Agent gives up too early when first attempt fails",
            "Agent doesn't explore alternative strategies",
            "Agent lacks confidence to try harder paths",
            "Agent mistakes difficulty for impossibility",
            "Agent stops at the first error rather than retrying",
            "Agent doesn't consider escalation or fallback options",
        ),
        examples=(
            "Agent tries to find available flights but stops after the first "
            "query returns no results instead of trying alternative dates",
            "Agent fails to complete a booking after a payment error and doesn't "
            "try an alternative payment method",
            "Agent reports a task cannot be completed after one failed attempt "
            "when it could succeed with different parameters",
        ),
    ),
}


def get_error_category(category: ErrorCategory) -> ErrorCategoryDefinition:
    """
    Get the definition for an error category.

    Args:
        category: The ErrorCategory to look up.

    Returns:
        ErrorCategoryDefinition for the given category.

    Raises:
        KeyError: If the category is not recognized.
    """
    if category not in ERROR_CATEGORIES:
        raise KeyError(f"Unknown error category: {category}")
    return ERROR_CATEGORIES[category]


def get_all_error_categories() -> list[ErrorCategoryDefinition]:
    """
    Get all error category definitions.

    Returns:
        List of all ErrorCategoryDefinition values, ordered by category.
    """
    return [ERROR_CATEGORIES[cat] for cat in ErrorCategory]


def get_category_by_name(name: str) -> Optional[ErrorCategory]:
    """
    Get an error category by name (full or short).

    Args:
        name: Name to search for (case-insensitive).

    Returns:
        ErrorCategory if found, None otherwise.
    """
    name_lower = name.lower()

    # Try exact match on value
    for cat in ErrorCategory:
        if cat.value.lower() == name_lower:
            return cat

    # Try match on name
    for cat in ErrorCategory:
        if cat.name.lower() == name_lower:
            return cat

    # Try short name match
    for cat in ErrorCategory:
        if cat.name_short.lower() == name_lower:
            return cat

    return None
