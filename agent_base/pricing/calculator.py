"""Per-step cost calculation from token usage and model pricing data.

Loads pricing from the bundled CSV and provides ``calculate_step_cost``
to compute the USD cost of a single Anthropic API call.  The caller
is responsible for summing step costs into a run-level total.

Pricing multipliers (long context, fast mode, batch, data residency)
are stacked multiplicatively on the CSV base rates per Anthropic docs.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from agent_base.logging import get_logger

if TYPE_CHECKING:
    from agent_base.core.config import CostBreakdown
    from agent_base.core.messages import Usage

logger = get_logger(__name__)

_MTOK = 1_000_000


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
    long_context_threshold: int
    batch_multiplier: float
    fast_mode_multiplier: float
    data_residency_multiplier: float
    web_search_per_request: float
    web_fetch_per_request: float


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
                batch_multiplier=float(row["batch_multiplier"]),
                fast_mode_multiplier=float(row["fast_mode_multiplier"]),
                data_residency_multiplier=float(row["data_residency_multiplier"]),
                web_search_per_request=float(row["web_search_per_request"]),
                web_fetch_per_request=float(row["web_fetch_per_request"]),
            )
            pricing[mp.model_id] = mp

    _pricing_cache = pricing
    return pricing


def resolve_model_pricing(model_name: str) -> ModelPricing | None:
    """Resolve an API model name to its pricing entry.

    Handles versioned model names like "claude-sonnet-4-5-20250929"
    by matching against base model IDs using substring matching.
    """
    pricing = load_pricing()

    if model_name in pricing:
        return pricing[model_name]

    for model_id in sorted(pricing.keys(), key=len, reverse=True):
        if model_id in model_name:
            return pricing[model_id]

    logger.warning("unknown_model_for_cost", model=model_name)
    return None


# ---------------------------------------------------------------------------
# Helpers to extract Anthropic-specific fields from Usage.raw_usage
# ---------------------------------------------------------------------------

def _extract_cache_write_breakdown(usage: Usage) -> tuple[int, int]:
    """Return (cache_write_5m_tokens, cache_write_1h_tokens) from raw_usage.

    Falls back to treating all cache_write_tokens as 5-minute TTL.
    """
    raw: dict[str, Any] = usage.raw_usage or {}
    cache_creation = raw.get("cache_creation")

    if isinstance(cache_creation, dict):
        return (
            cache_creation.get("ephemeral_5m_input_tokens", 0),
            cache_creation.get("ephemeral_1h_input_tokens", 0),
        )

    return (usage.cache_write_tokens or 0, 0)


def _extract_server_tool_counts(usage: Usage) -> tuple[int, int]:
    """Return (web_search_requests, web_fetch_requests) from raw_usage."""
    raw: dict[str, Any] = usage.raw_usage or {}
    stu = raw.get("server_tool_use")
    if isinstance(stu, dict):
        return (
            stu.get("web_search_requests", 0),
            stu.get("web_fetch_requests", 0),
        )
    return (0, 0)


def _raw_field(usage: Usage, key: str) -> Any:
    raw: dict[str, Any] = usage.raw_usage or {}
    return raw.get(key)


# ---------------------------------------------------------------------------
# Per-step cost calculation
# ---------------------------------------------------------------------------

def calculate_step_cost(usage: Usage, model_name: str) -> CostBreakdown | None:
    """Calculate the cost for a single API call.

    Reads Anthropic-specific billing context (cache TTL breakdown,
    server tool counts, inference_geo, service_tier, speed) from
    ``usage.raw_usage`` to apply the correct pricing multipliers.

    Multipliers stack multiplicatively on CSV base rates:
      effective_rate = base_rate * M_fast * M_longctx * M_batch * M_geo

    Fast mode and long context are mutually exclusive — fast mode
    covers the full context window with no additional long-context charge.

    Returns:
        CostBreakdown with per-category breakdown, or None if the
        model is unknown.
    """
    from agent_base.core.config import CostBreakdown

    pricing = resolve_model_pricing(model_name)
    if pricing is None:
        return None

    # --- Token counts ---
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cache_write_5m, cache_write_1h = _extract_cache_write_breakdown(usage)
    cache_read = usage.cache_read_tokens or 0

    # --- Billing context from raw_usage ---
    speed = _raw_field(usage, "speed")
    service_tier = _raw_field(usage, "service_tier")
    inference_geo = _raw_field(usage, "inference_geo")

    # --- Multipliers ---
    is_fast = speed == "fast" and pricing.fast_mode_multiplier > 0
    m_fast = pricing.fast_mode_multiplier if is_fast else 1.0

    total_step_input = input_tokens + cache_write_5m + cache_write_1h + cache_read
    is_long_context = (
        not is_fast
        and pricing.long_context_threshold > 0
        and total_step_input > pricing.long_context_threshold
    )
    m_longctx_in = pricing.long_context_input_multiplier if is_long_context else 1.0
    m_longctx_out = pricing.long_context_output_multiplier if is_long_context else 1.0

    m_batch = pricing.batch_multiplier if service_tier == "batch" else 1.0
    m_geo = pricing.data_residency_multiplier if inference_geo == "us" else 1.0

    # --- Effective rates (multipliers stack) ---
    input_rate = pricing.input_per_mtok * m_fast * m_longctx_in * m_batch * m_geo
    output_rate = pricing.output_per_mtok * m_fast * m_longctx_out * m_batch * m_geo
    cache_w5m_rate = pricing.cache_write_5m_per_mtok * m_fast * m_longctx_in * m_batch * m_geo
    cache_w1h_rate = pricing.cache_write_1h_per_mtok * m_fast * m_longctx_in * m_batch * m_geo
    cache_r_rate = pricing.cache_read_per_mtok * m_fast * m_longctx_in * m_batch * m_geo

    # Base input tokens = input_tokens minus cache tokens (they are a subset)
    base_input = max(0, input_tokens - (cache_write_5m + cache_write_1h) - cache_read)

    input_cost = (base_input / _MTOK) * input_rate
    output_cost = (output_tokens / _MTOK) * output_rate
    cache_write_5m_cost = (cache_write_5m / _MTOK) * cache_w5m_rate
    cache_write_1h_cost = (cache_write_1h / _MTOK) * cache_w1h_rate
    cache_read_cost = (cache_read / _MTOK) * cache_r_rate

    # --- Server tool costs ---
    web_search_reqs, web_fetch_reqs = _extract_server_tool_counts(usage)
    web_search_cost = web_search_reqs * pricing.web_search_per_request
    web_fetch_cost = web_fetch_reqs * pricing.web_fetch_per_request

    total_cost = (
        input_cost + output_cost
        + cache_write_5m_cost + cache_write_1h_cost + cache_read_cost
        + web_search_cost + web_fetch_cost
    )

    return CostBreakdown(
        total_cost=round(total_cost, 6),
        breakdown={
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "cache_write_5m_cost": round(cache_write_5m_cost, 6),
            "cache_write_1h_cost": round(cache_write_1h_cost, 6),
            "cache_read_cost": round(cache_read_cost, 6),
            "web_search_cost": round(web_search_cost, 6),
            "web_fetch_cost": round(web_fetch_cost, 6),
        },
    )
