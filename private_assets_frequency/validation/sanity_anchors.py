r"""
Public-market sanity anchors (Stage post-pipeline cross-checks).

The spec mandates four automated sanity anchors against a public proxy
benchmark — every pipeline run should compute these and surface warnings
when the output looks economically implausible. Unlike inter-stage
validation, these are *not* pass/fail gates; they are diagnostic checks
that flag suspicious results.

The four anchors (Stage Testing Strategy / Sanity Anchors):

    1. **Volatility anchor.** Desmoothed annualized vol within
       ``[0.5×, 3.0×]`` the public proxy's vol.
    2. **Correlation anchor.** Correlation between desmoothed and proxy
       within ``[0.4, 0.95]``.
    3. **Drawdown anchor.** Maximum drawdown during a known crisis is at
       least 50% of the proxy's drawdown in the same window.
    4. **Beta anchor.** Estimated beta to the proxy is consistent with
       economic leverage (configurable expected band per asset class).

Each anchor returns a :class:`SanityAnchorResult` with a numerical value,
the expected band, and a human-readable message. The pipeline runner
aggregates them into a ``sanity_anchors`` mapping on the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils.returns import realised_volatility


@dataclass(frozen=True)
class SanityAnchorResult:
    """Outcome of a single sanity-anchor check.

    Attributes
    ----------
    name
        Short identifier (``'vol'``, ``'correlation'``, ``'drawdown'``, ``'beta'``).
    passed
        ``True`` if the value falls inside ``expected_range``.
    value
        Numerical statistic computed from the data.
    expected_range
        ``(low, high)`` inclusive interval the value should land in.
    message
        Free-form description.
    metadata
        Free-form mapping with extra context.
    """

    name: str
    passed: bool
    value: float
    expected_range: tuple[float, float]
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Anchor 1: volatility
# ──────────────────────────────────────────────────────────────────


def vol_anchor(
    desmoothed: pd.Series,
    public_proxy: pd.Series,
    *,
    periods_per_year: int = 12,
    band: tuple[float, float] = (0.5, 3.0),
) -> SanityAnchorResult:
    """Desmoothed annualised vol should lie within ``band × proxy_vol``."""
    aligned = pd.concat([desmoothed.rename("d"), public_proxy.rename("p")], axis=1).dropna()
    if len(aligned) < 12:
        return _underpowered("vol", band, aligned)
    sigma_d = realised_volatility(aligned["d"], annualisation=periods_per_year)
    sigma_p = realised_volatility(aligned["p"], annualisation=periods_per_year)
    if sigma_p <= 0.0:
        return _underpowered("vol", band, aligned, msg="proxy has zero variance.")
    ratio = sigma_d / sigma_p
    low, high = band
    passed = low <= ratio <= high
    msg = (
        f"vol ratio σ_des/σ_proxy = {ratio:.2f} (band {low:.2f}–{high:.2f}); "
        f"σ_des = {sigma_d:.2%}, σ_proxy = {sigma_p:.2%}"
    )
    return SanityAnchorResult(
        name="vol",
        passed=passed,
        value=float(ratio),
        expected_range=(float(low), float(high)),
        message=msg,
        metadata={
            "sigma_desmoothed": sigma_d,
            "sigma_proxy": sigma_p,
            "n_observations": int(len(aligned)),
        },
    )


# ──────────────────────────────────────────────────────────────────
# Anchor 2: correlation
# ──────────────────────────────────────────────────────────────────


def correlation_anchor(
    desmoothed: pd.Series,
    public_proxy: pd.Series,
    *,
    band: tuple[float, float] = (0.4, 0.95),
) -> SanityAnchorResult:
    """Desmoothed-vs-proxy correlation should lie within ``band``."""
    aligned = pd.concat([desmoothed.rename("d"), public_proxy.rename("p")], axis=1).dropna()
    if len(aligned) < 12:
        return _underpowered("correlation", band, aligned)
    corr = float(aligned["d"].corr(aligned["p"]))
    low, high = band
    passed = low <= corr <= high
    msg = (
        f"corr(desmoothed, proxy) = {corr:.3f} (band {low:.2f}–{high:.2f}); "
        f"below {low:.2f} → factor model may be missing factors; above {high:.2f} → "
        "output may be just the public proxy."
    )
    return SanityAnchorResult(
        name="correlation",
        passed=passed,
        value=corr,
        expected_range=(float(low), float(high)),
        message=msg,
        metadata={"n_observations": int(len(aligned))},
    )


# ──────────────────────────────────────────────────────────────────
# Anchor 3: drawdown
# ──────────────────────────────────────────────────────────────────


def drawdown_anchor(
    desmoothed: pd.Series,
    public_proxy: pd.Series,
    *,
    crisis_window: tuple[str, str] | None = None,
    min_ratio: float = 0.5,
) -> SanityAnchorResult:
    """Crisis drawdown of the desmoothed series should be ≥ ``min_ratio`` × proxy's.

    Parameters
    ----------
    desmoothed, public_proxy
        Aligned (or alignable) return series.
    crisis_window
        Inclusive ``(start, end)`` strings defining the crisis. Defaults to
        the GFC window 2008-09-01 – 2009-03-31. Pass ``None`` to evaluate
        on the whole series.
    min_ratio
        Required ratio between desmoothed crisis drawdown and proxy's.
    """
    if crisis_window is None:
        crisis_window = ("2008-09-01", "2009-03-31")
    start, end = crisis_window
    aligned = pd.concat([desmoothed.rename("d"), public_proxy.rename("p")], axis=1).dropna()
    sub = aligned.loc[start:end]
    if len(sub) < 3:
        return _underpowered(
            "drawdown",
            (min_ratio, float("inf")),
            aligned,
            msg=f"crisis window {start}..{end} has < 3 observations.",
        )
    dd_d = _max_drawdown(sub["d"])
    dd_p = _max_drawdown(sub["p"])
    if dd_p == 0.0:
        return _underpowered(
            "drawdown",
            (min_ratio, float("inf")),
            aligned,
            msg="proxy had zero drawdown in window.",
        )
    ratio = dd_d / dd_p  # both negative — ratio > 0 if both drawdowns
    passed = ratio >= min_ratio
    msg = (
        f"crisis drawdown ratio (desmoothed/proxy) = {ratio:.2f} "
        f"(target ≥ {min_ratio:.2f}); desm DD {dd_d:.2%}, proxy DD {dd_p:.2%} "
        f"in {start}..{end}."
    )
    return SanityAnchorResult(
        name="drawdown",
        passed=passed,
        value=float(ratio),
        expected_range=(float(min_ratio), float("inf")),
        message=msg,
        metadata={
            "crisis_window": crisis_window,
            "drawdown_desmoothed": dd_d,
            "drawdown_proxy": dd_p,
        },
    )


# ──────────────────────────────────────────────────────────────────
# Anchor 4: beta
# ──────────────────────────────────────────────────────────────────


def beta_anchor(
    desmoothed: pd.Series,
    public_proxy: pd.Series,
    *,
    expected_band: tuple[float, float] = (0.3, 2.0),
) -> SanityAnchorResult:
    """Beta of desmoothed to proxy should sit in ``expected_band``.

    Default band ``(0.3, 2.0)`` is the spec's "consistent with economic
    leverage" window for buyout-style strategies; tune per asset class
    (lower for venture, narrower for infra).
    """
    aligned = pd.concat([desmoothed.rename("d"), public_proxy.rename("p")], axis=1).dropna()
    if len(aligned) < 12:
        return _underpowered("beta", expected_band, aligned)
    cov = float(aligned.cov().iloc[0, 1])
    var_p = float(aligned["p"].var())
    if var_p <= 0.0:
        return _underpowered(
            "beta", expected_band, aligned, msg="proxy has zero variance."
        )
    beta = cov / var_p
    low, high = expected_band
    passed = low <= beta <= high
    msg = (
        f"β to proxy = {beta:.2f} (expected {low:.2f}–{high:.2f}); "
        f"outside this band typically indicates leverage mis-specification."
    )
    return SanityAnchorResult(
        name="beta",
        passed=passed,
        value=float(beta),
        expected_range=(float(low), float(high)),
        message=msg,
    )


# ──────────────────────────────────────────────────────────────────
# Run all anchors
# ──────────────────────────────────────────────────────────────────


def run_all_anchors(
    desmoothed: pd.Series,
    public_proxy: pd.Series,
    *,
    periods_per_year: int = 12,
    vol_band: tuple[float, float] = (0.5, 3.0),
    correlation_band: tuple[float, float] = (0.4, 0.95),
    drawdown_window: tuple[str, str] | None = None,
    drawdown_min_ratio: float = 0.5,
    beta_band: tuple[float, float] = (0.3, 2.0),
) -> dict[str, SanityAnchorResult]:
    """Run all four sanity anchors and return a dict keyed by anchor name."""
    return {
        "vol": vol_anchor(
            desmoothed,
            public_proxy,
            periods_per_year=periods_per_year,
            band=vol_band,
        ),
        "correlation": correlation_anchor(
            desmoothed, public_proxy, band=correlation_band
        ),
        "drawdown": drawdown_anchor(
            desmoothed,
            public_proxy,
            crisis_window=drawdown_window,
            min_ratio=drawdown_min_ratio,
        ),
        "beta": beta_anchor(
            desmoothed, public_proxy, expected_band=beta_band
        ),
    }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of cumulative returns. Returns a non-positive float."""
    log_ret = np.log1p(returns.to_numpy())
    cumulative = np.cumsum(log_ret)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak  # log-space drawdown
    return float(np.expm1(drawdown.min()))


def _underpowered(
    name: str,
    band: tuple[float, float],
    aligned: pd.DataFrame,
    *,
    msg: str | None = None,
) -> SanityAnchorResult:
    return SanityAnchorResult(
        name=name,
        passed=False,
        value=float("nan"),
        expected_range=band,
        message=msg
        or f"{name} anchor underpowered: only {len(aligned)} aligned observations.",
        metadata={"underpowered": True, "n_observations": int(len(aligned))},
    )


__all__ = [
    "SanityAnchorResult",
    "beta_anchor",
    "correlation_anchor",
    "drawdown_anchor",
    "run_all_anchors",
    "vol_anchor",
]
