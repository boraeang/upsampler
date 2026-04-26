r"""
Round-trip and moment-preservation consistency checks.

These checks complement :mod:`stage_validation` by giving the user a single
entry point to verify that a pipeline run preserved the load-bearing
properties of the input data. They are pure functions; the pipeline runner
calls them at the end and surfaces any failures via :class:`FallbackHandler`.

Three checks:

    * :func:`assert_round_trip` — high-frequency series compounds to
      low-frequency input within tolerance. Mirrors the Stage 3 → 4 gate
      but operates on the final pipeline output (which has been through
      additional transformations).
    * :func:`assert_moment_preservation` — annualised volatility of the
      disaggregated series matches the desmoothed-native-frequency vol
      within sampling error.
    * :func:`assert_cross_strategy_correlation_preserved` — for multi-
      strategy runs, cross-sectional correlation at the disaggregated
      frequency matches the input cross-sectional correlation up to
      sampling error.

Each function returns a :class:`ConsistencyResult` and never raises;
strict-mode escalation is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.protocols import AggregationType
from ..utils.returns import aggregate_returns, realised_volatility


@dataclass(frozen=True)
class ConsistencyResult:
    """Outcome of a consistency check.

    Attributes
    ----------
    name
        Identifier (``'round_trip'``, ``'moment_preservation'``,
        ``'cross_strategy_correlation'``).
    passed
        ``True`` if the check passed under its tolerance.
    value
        Numerical statistic (max error, ratio, etc.).
    tolerance
        Threshold used.
    message
        Free-form description.
    metadata
        Free-form mapping with extra context.
    """

    name: str
    passed: bool
    value: float
    tolerance: float
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Round-trip aggregation
# ──────────────────────────────────────────────────────────────────


def assert_round_trip(
    low_frequency: pd.Series,
    high_frequency: pd.Series,
    ratio: int,
    *,
    aggregation: AggregationType | str = AggregationType.MULTIPLICATIVE,
    tolerance: float = 1e-10,
) -> ConsistencyResult:
    """High-frequency aggregated must equal low-frequency input within ``tolerance``."""
    aggregated = aggregate_returns(high_frequency, ratio=ratio, method=aggregation)
    if isinstance(aggregated, pd.Series):
        agg_arr = aggregated.to_numpy()
    else:
        agg_arr = np.asarray(aggregated)
    lf_arr = low_frequency.to_numpy()
    if agg_arr.shape != lf_arr.shape:
        return ConsistencyResult(
            name="round_trip",
            passed=False,
            value=float("inf"),
            tolerance=tolerance,
            message=(
                f"shape mismatch — aggregated {agg_arr.shape} vs "
                f"low-freq {lf_arr.shape}."
            ),
        )
    err = float(np.max(np.abs(agg_arr - lf_arr)))
    passed = err <= tolerance
    return ConsistencyResult(
        name="round_trip",
        passed=passed,
        value=err,
        tolerance=tolerance,
        message=f"max |round-trip error| = {err:.3e} (tolerance {tolerance:.1e})",
        metadata={"ratio": int(ratio), "aggregation": str(aggregation)},
    )


# ──────────────────────────────────────────────────────────────────
# Moment preservation
# ──────────────────────────────────────────────────────────────────


def assert_moment_preservation(
    desmoothed_native: pd.Series,
    high_frequency: pd.Series,
    *,
    native_periods_per_year: int = 4,
    high_frequency_periods_per_year: int = 12,
    tolerance: float = 0.30,
) -> ConsistencyResult:
    r"""Annualised vol of the disaggregated series matches the native-frequency vol.

    The two annualised volatilities should agree up to sampling error. The
    default tolerance ``0.30`` (30 % relative deviation) is generous to
    absorb the noise that disaggregation injects via the indicator/factor
    path; tighten if the moment match matters for downstream risk numbers.
    """
    sigma_native = realised_volatility(
        desmoothed_native, annualisation=native_periods_per_year
    )
    sigma_hf = realised_volatility(
        high_frequency, annualisation=high_frequency_periods_per_year
    )
    if sigma_native <= 0.0:
        return ConsistencyResult(
            name="moment_preservation",
            passed=False,
            value=float("nan"),
            tolerance=tolerance,
            message="native-frequency series has zero variance.",
        )
    rel_diff = abs(sigma_hf - sigma_native) / sigma_native
    passed = rel_diff <= tolerance
    return ConsistencyResult(
        name="moment_preservation",
        passed=passed,
        value=float(rel_diff),
        tolerance=tolerance,
        message=(
            f"annualised σ: native {sigma_native:.2%}, high-freq {sigma_hf:.2%} "
            f"(rel Δ {rel_diff:.1%}, tolerance {tolerance:.0%})"
        ),
        metadata={
            "sigma_native": sigma_native,
            "sigma_high_frequency": sigma_hf,
        },
    )


# ──────────────────────────────────────────────────────────────────
# Cross-strategy correlation
# ──────────────────────────────────────────────────────────────────


def assert_cross_strategy_correlation_preserved(
    low_frequency_panel: pd.DataFrame,
    high_frequency_panel: pd.DataFrame,
    *,
    tolerance: float = 0.20,
) -> ConsistencyResult:
    """Off-diagonal correlations should match between LF and HF panels.

    Compares the off-diagonal entries of ``corr(LF)`` and ``corr(HF)``;
    fails if any entry differs by more than ``tolerance`` in absolute terms.
    """
    if not isinstance(low_frequency_panel, pd.DataFrame):
        raise TypeError("low_frequency_panel must be a DataFrame.")
    if not isinstance(high_frequency_panel, pd.DataFrame):
        raise TypeError("high_frequency_panel must be a DataFrame.")
    if list(low_frequency_panel.columns) != list(high_frequency_panel.columns):
        raise ValueError(
            "panel columns must match: "
            f"{list(low_frequency_panel.columns)} vs {list(high_frequency_panel.columns)}"
        )
    if low_frequency_panel.shape[1] < 2:
        return ConsistencyResult(
            name="cross_strategy_correlation",
            passed=True,
            value=0.0,
            tolerance=tolerance,
            message="< 2 strategies — correlation check skipped.",
        )
    lf_corr = low_frequency_panel.corr().to_numpy()
    hf_corr = high_frequency_panel.corr().to_numpy()
    triu = np.triu_indices(lf_corr.shape[0], k=1)
    diff = np.abs(lf_corr[triu] - hf_corr[triu])
    max_diff = float(np.max(diff))
    passed = max_diff <= tolerance
    return ConsistencyResult(
        name="cross_strategy_correlation",
        passed=passed,
        value=max_diff,
        tolerance=tolerance,
        message=(
            f"max |Δρ_off-diag| = {max_diff:.3f} (tolerance {tolerance:.2f})"
        ),
        metadata={"strategies": list(low_frequency_panel.columns)},
    )


__all__ = [
    "ConsistencyResult",
    "assert_cross_strategy_correlation_preserved",
    "assert_moment_preservation",
    "assert_round_trip",
]
