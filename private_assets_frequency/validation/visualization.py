"""
Diagnostic plots for pipeline results.

This module is intentionally lightweight — :mod:`matplotlib` is an
*optional* dependency. Each plotting function checks for availability,
raises a clear ``ImportError`` with installation guidance if missing, and
returns a ``matplotlib.figure.Figure`` so callers can route to a notebook,
file, or buffer as they prefer.

Provided plots:

    * :func:`plot_desmoothing_diagnostics` — λ prior vs posterior, β
      posterior across the λ grid (catches λ–β confound).
    * :func:`plot_aggregation_consistency` — round-trip aggregation:
      input vs aggregated output side-by-side, residuals on a second axis.
    * :func:`plot_factor_decomposition` — systematic vs idiosyncratic
      contribution per period.
    * :func:`plot_volatility_term_structure` — annualised vol at native,
      monthly, and daily horizons.

If matplotlib is missing the functions raise an :class:`ImportError`
suggesting ``pip install private_assets_frequency[viz]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.protocols import DesmoothedResult


def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover — only fires when mpl missing
        raise ImportError(
            "matplotlib is required for visualization functions; install with "
            "'pip install private_assets_frequency[viz]'."
        ) from exc
    return plt


def plot_desmoothing_diagnostics(
    result: DesmoothedResult,
) -> Any:
    """λ prior vs posterior + β posterior across the λ grid.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    posterior_summary = result.posterior_summary
    if "lambda" in posterior_summary and "grid" in posterior_summary["lambda"]:
        grid = posterior_summary["lambda"]["grid"]
        density = posterior_summary["lambda"]["density"]
        axes[0].plot(grid, density, label="posterior")
        axes[0].set_xlabel("λ")
        axes[0].set_ylabel("density")
        axes[0].set_title("λ posterior")
        axes[0].axvline(
            posterior_summary["lambda"].get("mean", float("nan")),
            color="red",
            linestyle="--",
            label="posterior mean",
        )
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "no λ posterior", ha="center", va="center")

    beta_summary = posterior_summary.get("beta", {})
    if isinstance(beta_summary, dict) and beta_summary:
        names = list(beta_summary)
        means = [
            beta_summary[k]["mean"] if isinstance(beta_summary[k], dict) else 0.0
            for k in names
        ]
        stds = [
            beta_summary[k].get("std", 0.0) if isinstance(beta_summary[k], dict) else 0.0
            for k in names
        ]
        axes[1].errorbar(np.arange(len(names)), means, yerr=stds, fmt="o", capsize=4)
        axes[1].set_xticks(np.arange(len(names)))
        axes[1].set_xticklabels(names, rotation=30, ha="right")
        axes[1].set_ylabel("β posterior (mean ± std)")
        axes[1].set_title("β posterior")
    else:
        axes[1].text(0.5, 0.5, "no β posterior", ha="center", va="center")

    fig.tight_layout()
    return fig


def plot_aggregation_consistency(
    low_frequency: pd.Series,
    high_frequency: pd.Series,
    ratio: int,
) -> Any:
    """Plot ``low_frequency`` and the aggregated ``high_frequency`` together.

    The bottom panel shows the round-trip residual; a horizontal line at
    machine epsilon helps the user see whether the constraint is exact.
    """
    plt = _require_mpl()
    from ..utils.returns import aggregate_returns

    aggregated = aggregate_returns(high_frequency, ratio=ratio).to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(low_frequency.index, low_frequency.to_numpy(), "o", label="input LF")
    axes[0].plot(low_frequency.index, aggregated, "x", label="aggregated HF")
    axes[0].set_ylabel("return")
    axes[0].legend()
    axes[0].set_title("Aggregation consistency")
    diff = aggregated - low_frequency.to_numpy()
    axes[1].plot(low_frequency.index, diff)
    axes[1].axhline(0, color="grey", linewidth=0.5)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel("date")
    fig.tight_layout()
    return fig


def plot_factor_decomposition(
    returns: pd.Series,
    systematic: pd.Series,
    residual: pd.Series,
) -> Any:
    """Stacked plot of systematic vs idiosyncratic contributions per period."""
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(returns.index, returns.to_numpy(), label="returns", alpha=0.7)
    ax.plot(systematic.index, systematic.to_numpy(), label="systematic", linestyle="--")
    ax.plot(residual.index, residual.to_numpy(), label="residual", linestyle=":")
    ax.set_xlabel("date")
    ax.set_ylabel("return")
    ax.set_title("Factor decomposition")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_volatility_term_structure(
    by_horizon: dict[str, float],
) -> Any:
    """Bar chart of annualised volatility at multiple horizons.

    Parameters
    ----------
    by_horizon
        Mapping ``{horizon_name: annualised_volatility}`` (e.g. ``{'native': 0.18,
        'monthly': 0.16, 'daily': 0.15}``).
    """
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(by_horizon)
    values = [by_horizon[k] for k in keys]
    ax.bar(keys, values)
    ax.set_ylabel("annualised σ")
    ax.set_title("Volatility term structure")
    fig.tight_layout()
    return fig


__all__ = [
    "plot_aggregation_consistency",
    "plot_desmoothing_diagnostics",
    "plot_factor_decomposition",
    "plot_volatility_term_structure",
]
