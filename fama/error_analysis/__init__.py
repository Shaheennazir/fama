"""Error analysis components for FAMA."""

from fama.error_analysis.analyzer import ErrorAnalyzer
from fama.error_analysis.categories import (
    ERROR_CATEGORIES,
    ErrorCategoryDefinition,
    get_error_category,
    get_all_error_categories,
)

__all__ = [
    "ErrorAnalyzer",
    "ERROR_CATEGORIES",
    "ErrorCategoryDefinition",
    "get_error_category",
    "get_all_error_categories",
]
