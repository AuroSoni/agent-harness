"""Cost calculation from cumulative token usage and model pricing data.

Loads pricing from the bundled CSV and provides functions to calculate
costs from a Usage object as tracked by the agent run loop.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_base.logging import get_logger

if TYPE_CHECKING:
    from agent_base.core.config import CostBreakdown
    from agent_base.core.messages import Usage

logger = get_logger(__name__)


@dataclass
class ModelPricing:
    """Pricing data for a single Anthropic model."""
    model_id: str
    display_name: str
    input_per_mtok: float
    cache_write_5m_per_mtok: float
    cache_write_1h_per_mtok: float
    cache_read_per_mtok: float
    output_per_mtok: float
    long_context_input_multiplier: float
    long_context_output_multiplier: float
    long_context_threshold: int  # 0 means no long context pricing


# Module-level cache for loaded pricing data
_pricing_cache: dict[str, ModelPricing] | None = None


def load_pricing() -> dict[str, ModelPricing]:
    """Load pricing data from the bundled CSV file.

    Returns:
        Dictionary mapping model_id to ModelPricing.
        Cached after first load.
    """
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache

    csv_path = os.path.join(os.path.dirname(__file__), "models.csv")
    pricing: dict[str, ModelPricing] = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mp = ModelPricing(
                model_id=row["model_id"],
                display_name=row["display_name"],
                input_per_mtok=float(row["input_per_mtok"]),
                cache_write_5m_per_mtok=float(row["cache_write_5m_per_mtok"]),
                cache_write_1h_per_mtok=float(row["cache_write_1h_per_mtok"]),
                cache_read_per_mtok=float(row["cache_read_per_mtok"]),
                output_per_mtok=float(row["output_per_mtok"]),
                long_context_input_multiplier=float(row["long_context_input_multiplier"]),
                long_context_output_multiplier=float(row["long_context_output_multiplier"]),
                long_context_threshold=int(row["long_context_threshold"]),
            )
            pricing[mp.model_id] = mp

    _pricing_cache = pricing
    return pricing


def resolve_model_pricing(model_name: str) -> ModelPricing | None:
    """Resolve an API model name to its pricing entry.

    Handles versioned model names like "claude-sonnet-4-5-20250929"
    by matching against base model IDs using substring matching.

    Args:
        model_name: The model name from the API response.

    Returns:
        ModelPricing if found, None if model is unknown.
    """
    pricing = load_pricing()

    # Exact match first
    if model_name in pricing:
        return pricing[model_name]

    # Substring match: most-specific (longest) model_id first
    for model_id in sorted(pricing.keys(), key=len, reverse=True):
        if model_id in model_name:
            return pricing[model_id]

    logger.warning("unknown_model_for_cost", model=model_name)
    return None


def calculate_cost(usage: Usage, model_name: str) -> CostBreakdown | None:
    #TODO: Cost can depend on LLMConfig also.
    """Calculate the total cost for an agent run from cumulative usage.

    Applies model pricing to the cumulative token counts. Cache tokens
    are a subset of input_tokens in the Anthropic API, so base input
    cost is calculated on input_tokens minus cache tokens to avoid
    double-counting.

    Args:
        usage: Cumulative Usage object with token counts across all steps.
        model_name: API model name (e.g., "claude-sonnet-4-5-20250929").

    Returns:
        CostBreakdown with total_cost and per-category breakdown,
        or None if model pricing is unknown.
    """
    from agent_base.core.config import CostBreakdown

    pricing = resolve_model_pricing(model_name)
    if pricing is None:
        return None

    total_input = usage.input_tokens
    total_output = usage.output_tokens
    total_cache_write = usage.cache_write_tokens or 0
    total_cache_read = usage.cache_read_tokens or 0

    # Long-context detection: apply if cumulative input exceeds threshold.
    long_context = (
        pricing.long_context_threshold > 0
        and total_input > pricing.long_context_threshold
    )

    input_multiplier = pricing.long_context_input_multiplier if long_context else 1.0
    output_multiplier = pricing.long_context_output_multiplier if long_context else 1.0

    # Base input tokens = total_input - cache_write - cache_read
    # (cache tokens are a subset of input_tokens in the API response)
    base_input = max(0, total_input - total_cache_write - total_cache_read)

    input_cost = (base_input / 1_000_000) * pricing.input_per_mtok * input_multiplier
    output_cost = (total_output / 1_000_000) * pricing.output_per_mtok * output_multiplier
    cache_write_cost = (total_cache_write / 1_000_000) * pricing.cache_write_5m_per_mtok * input_multiplier
    cache_read_cost = (total_cache_read / 1_000_000) * pricing.cache_read_per_mtok * input_multiplier

    total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost

    return CostBreakdown(
        total_cost=round(total_cost, 6),
        breakdown={
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "cache_write_cost": round(cache_write_cost, 6),
            "cache_read_cost": round(cache_read_cost, 6),
        },
    )
