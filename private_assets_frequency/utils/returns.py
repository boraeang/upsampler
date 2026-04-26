"""
Return arithmetic utilities for ``private_assets_frequency``.

The whole pipeline rests on getting return aggregation exactly right. The
spec is explicit (Critical Implementation Note 3): use
``(1 + r_low) = ∏(1 + r_high)`` for *every* aggregation check; the additive
approximation drifts by 50+ bps for PE-magnitude quarterly returns.

This module exposes:

    * :func:`to_log_returns` / :func:`from_log_returns` — log/expm1 transforms
      that round-trip exactly through floating-point arithmetic.
    * :func:`compound_returns` — compound a vector of returns to a single
      total return.
    * :func:`aggregate_returns` — fold a high-frequency series into a
      lower-frequency one of length ``n_high // ratio``. Multiplicative is the
      default (and the only correct choice for return *levels*).
    * :func:`aggregation_matrix` — build the ``C`` matrix used by Chow-Lin
      and Fernández in log-space.
    * :func:`assert_aggregation_consistent` — round-trip assertion used by
      Stage 3 → Stage 4 validation.
    * :func:`annualize_volatility` and :func:`annualize_return` — standard
      annualisation helpers.

Conventions
-----------
* All functions accept either a ``numpy.ndarray`` or a ``pandas.Series`` /
  ``pandas.DataFrame``. Pandas inputs return pandas outputs at the
  appropriate frequency-reduced index.
* ``axis`` semantics on 2-D inputs match pandas: rows are observations,
  columns are series.
* Aggregation requires the input length to be an integer multiple of the
  ratio. ``trim='none'`` (default) raises; ``trim='leading'`` or
  ``'trailing'`` discards the appropriate edge to make the length divisible.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import pandas as pd

from ..core.protocols import AggregationType, Frequency

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

PERIODS_PER_YEAR: dict[str, int] = {
    Frequency.QUARTERLY.value: 4,
    Frequency.MONTHLY.value: 12,
    Frequency.WEEKLY.value: 52,
    Frequency.DAILY.value: 252,
}


# ──────────────────────────────────────────────────────────────────
# Log / simple return transforms
# ──────────────────────────────────────────────────────────────────


def to_log_returns(simple_returns: np.ndarray | pd.Series | pd.DataFrame):
    """Convert simple returns ``r`` to log returns ``log(1 + r)``.

    Equivalent to :func:`numpy.log1p` but preserves pandas containers.

    Raises
    ------
    ValueError
        If any input value is ``≤ -1`` (would imply log of a non-positive
        number — not a valid return).
    """
    arr = _values(simple_returns)
    if (arr <= -1.0).any():
        bad = float(arr[arr <= -1.0].min())
        raise ValueError(
            f"to_log_returns: input contains values ≤ -1 (min={bad:.6g}); "
            "cannot take log of a non-positive return level."
        )
    return _wrap_like(np.log1p(arr), simple_returns)


def from_log_returns(log_returns: np.ndarray | pd.Series | pd.DataFrame):
    """Inverse of :func:`to_log_returns`: ``exp(lr) - 1``."""
    arr = _values(log_returns)
    return _wrap_like(np.expm1(arr), log_returns)


# ──────────────────────────────────────────────────────────────────
# Compounding
# ──────────────────────────────────────────────────────────────────


def compound_returns(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    *,
    axis: int = 0,
) -> float | np.ndarray | pd.Series:
    """Compound a series of simple returns into a single total return.

    Computes ``∏(1 + r) − 1`` in log-space (``expm1(sum(log1p(r)))``) for
    numerical stability. Returns a scalar when ``returns`` is 1-D, a
    ``numpy.ndarray`` of length ``n_columns`` for a 2-D ``ndarray``, and a
    ``pandas.Series`` named after the columns for a ``DataFrame``.

    Parameters
    ----------
    returns
        Simple returns, possibly 2-D.
    axis
        Axis along which to compound (default 0 — rows / time).
    """
    arr = _values(returns)
    if (arr <= -1.0).any():
        raise ValueError("compound_returns: input contains values ≤ -1.")
    total = np.expm1(np.log1p(arr).sum(axis=axis))
    if isinstance(returns, pd.DataFrame):
        if axis == 0:
            return pd.Series(total, index=returns.columns, name="compound_return")
        return pd.Series(total, index=returns.index, name="compound_return")
    if isinstance(returns, pd.Series):
        return float(total)
    if arr.ndim == 1:
        return float(total)
    return total


# ──────────────────────────────────────────────────────────────────
# Frequency aggregation
# ──────────────────────────────────────────────────────────────────


_TrimMode = Literal["none", "leading", "trailing"]


def aggregate_returns(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    ratio: int,
    *,
    method: AggregationType | str = AggregationType.MULTIPLICATIVE,
    trim: _TrimMode = "none",
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Fold high-frequency returns into a lower-frequency series.

    For a 1-D input of length ``n_high`` and a ratio ``k``, returns a series
    of length ``n_high // k`` whose i-th entry is the compound (or sum) of
    the ``k`` consecutive high-frequency entries starting at offset ``i*k``.

    Parameters
    ----------
    returns
        High-frequency returns.
    ratio
        Number of high-frequency periods per low-frequency period (e.g. 3 for
        monthly → quarterly, 12 for monthly → annual).
    method
        ``AggregationType.MULTIPLICATIVE`` (default) or
        ``AggregationType.ADDITIVE``. Strings ``"multiplicative"`` /
        ``"additive"`` are also accepted.
    trim
        How to handle a length not divisible by ``ratio``: ``"none"``
        (default) raises, ``"leading"`` drops the first ``n_high % ratio``
        observations, ``"trailing"`` drops the last.

    Returns
    -------
    Same container type as the input. Pandas inputs return a pandas object
    indexed at the *last* date of each low-frequency block.
    """
    if not isinstance(ratio, (int, np.integer)) or ratio < 1:
        raise ValueError(f"ratio must be an integer >= 1, got {ratio!r}")
    method = AggregationType(method) if not isinstance(method, AggregationType) else method

    arr = _values(returns)
    if arr.ndim not in (1, 2):
        raise ValueError(f"returns must be 1-D or 2-D, got ndim={arr.ndim}")

    n_high = arr.shape[0]
    rem = n_high % ratio
    if rem != 0:
        if trim == "leading":
            arr = arr[rem:]
            if isinstance(returns, (pd.Series, pd.DataFrame)):
                returns = returns.iloc[rem:]
        elif trim == "trailing":
            arr = arr[: n_high - rem]
            if isinstance(returns, (pd.Series, pd.DataFrame)):
                returns = returns.iloc[: n_high - rem]
        else:
            raise ValueError(
                f"length {n_high} is not divisible by ratio {ratio}; "
                f"pass trim='leading' or 'trailing' to discard {rem} observation(s)."
            )
        n_high = arr.shape[0]
    n_low = n_high // ratio

    if method is AggregationType.MULTIPLICATIVE:
        if (arr <= -1.0).any():
            raise ValueError("aggregate_returns: input contains values ≤ -1.")
        log_arr = np.log1p(arr)
        reshaped = log_arr.reshape(n_low, ratio, *arr.shape[1:])
        agg = np.expm1(reshaped.sum(axis=1))
    else:  # ADDITIVE
        reshaped = arr.reshape(n_low, ratio, *arr.shape[1:])
        agg = reshaped.sum(axis=1)

    # Wrap back into pandas if needed
    if isinstance(returns, pd.DataFrame):
        # Index at the last high-frequency date of each block
        new_index = returns.index[ratio - 1 :: ratio][:n_low]
        return pd.DataFrame(agg, index=new_index, columns=returns.columns)
    if isinstance(returns, pd.Series):
        new_index = returns.index[ratio - 1 :: ratio][:n_low]
        return pd.Series(agg, index=new_index, name=returns.name)
    return agg


def aggregation_matrix(n_low: int, ratio: int) -> np.ndarray:
    """Build the temporal aggregation matrix ``C`` of shape ``(n_low, n_low * ratio)``.

    Each row of ``C`` selects a contiguous block of ``ratio`` columns and
    sums them. In log-space this implements the multiplicative aggregation
    constraint ``log(1 + r_low) = Σ log(1 + r_high)``; in arithmetic space
    it implements the additive constraint.

    The matrix is structurally sparse but returned dense — for typical
    pipeline sizes (a few hundred low-frequency periods) this is fine; switch
    to ``scipy.sparse`` only if profiling requires it.
    """
    if not (isinstance(n_low, (int, np.integer)) and n_low > 0):
        raise ValueError(f"n_low must be a positive integer, got {n_low!r}")
    if not (isinstance(ratio, (int, np.integer)) and ratio >= 1):
        raise ValueError(f"ratio must be an integer >= 1, got {ratio!r}")
    n_high = n_low * ratio
    C = np.zeros((n_low, n_high))
    for k in range(n_low):
        C[k, k * ratio : (k + 1) * ratio] = 1.0
    return C


# ──────────────────────────────────────────────────────────────────
# Round-trip checks
# ──────────────────────────────────────────────────────────────────


def assert_aggregation_consistent(
    low: np.ndarray | pd.Series | pd.DataFrame,
    high: np.ndarray | pd.Series | pd.DataFrame,
    ratio: int,
    *,
    method: AggregationType | str = AggregationType.MULTIPLICATIVE,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> None:
    """Assert that ``high`` aggregates back to ``low`` within tolerance.

    Used by the inter-stage validation gate Stage 3 → Stage 4 (Critical
    Implementation Note 7): "monthly returns must compound to quarterly
    within tolerance 1e-10."

    Raises
    ------
    AssertionError
        If aggregation fails. The message reports the per-block max absolute
        error.
    """
    aggregated = aggregate_returns(high, ratio, method=method)
    low_arr = _values(low)
    high_agg = _values(aggregated)
    if low_arr.shape != high_agg.shape:
        raise AssertionError(
            f"shape mismatch: low={low_arr.shape}, aggregated={high_agg.shape}"
        )
    diff = np.abs(low_arr - high_agg)
    tol = atol + rtol * np.abs(low_arr)
    if (diff > tol).any():
        max_err = float(diff.max())
        raise AssertionError(
            f"aggregation inconsistency: max |Δ| = {max_err:.3e} "
            f"(atol={atol:.1e}, rtol={rtol:.1e})."
        )


# ──────────────────────────────────────────────────────────────────
# Annualisation
# ──────────────────────────────────────────────────────────────────


def annualize_volatility(period_vol: float, periods_per_year: int) -> float:
    """Annualise a per-period volatility under iid assumption: ``σ √N``."""
    if period_vol < 0:
        raise ValueError(f"period_vol must be non-negative, got {period_vol}")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    return float(period_vol) * float(np.sqrt(periods_per_year))


def annualize_return(
    period_return: float,
    periods_per_year: int,
    *,
    method: Literal["compound", "arithmetic"] = "compound",
) -> float:
    """Annualise a per-period return.

    ``method='compound'`` (default) returns ``(1 + r)^N − 1``; ``'arithmetic'``
    returns ``r · N``. The compound form is the only correct one for return
    levels; the arithmetic form is sometimes useful for log returns.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    if method == "compound":
        if period_return <= -1.0:
            raise ValueError(
                f"period_return {period_return} <= -1; cannot compound."
            )
        return float((1.0 + period_return) ** periods_per_year - 1.0)
    if method == "arithmetic":
        return float(period_return) * float(periods_per_year)
    raise ValueError(f"method must be 'compound' or 'arithmetic', got {method!r}")


def realised_volatility(
    returns: np.ndarray | pd.Series,
    *,
    ddof: int = 1,
    annualisation: int | None = None,
) -> float:
    """Sample volatility of a return series.

    Parameters
    ----------
    returns
        1-D return series.
    ddof
        Delta degrees of freedom (default 1, i.e. sample standard deviation).
    annualisation
        If supplied, multiply the result by ``√annualisation``. Pass 4 for
        quarterly→annual, 12 for monthly→annual, etc.
    """
    arr = _values(returns)
    if arr.ndim != 1:
        raise ValueError(f"returns must be 1-D, got ndim={arr.ndim}")
    sigma = float(np.std(arr, ddof=ddof))
    if annualisation is not None:
        sigma *= float(np.sqrt(annualisation))
    return sigma


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────


def _values(x):
    """Return a numpy view of ``x`` regardless of input container."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    return np.asarray(x)


@overload
def _wrap_like(values: np.ndarray, template: pd.Series) -> pd.Series: ...
@overload
def _wrap_like(values: np.ndarray, template: pd.DataFrame) -> pd.DataFrame: ...
@overload
def _wrap_like(values: np.ndarray, template: np.ndarray) -> np.ndarray: ...
def _wrap_like(values, template):
    """Wrap ``values`` into the same container type as ``template``."""
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=template.name)
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(values, index=template.index, columns=template.columns)
    return values


__all__ = [
    "PERIODS_PER_YEAR",
    "aggregate_returns",
    "aggregation_matrix",
    "annualize_return",
    "annualize_volatility",
    "assert_aggregation_consistent",
    "compound_returns",
    "from_log_returns",
    "realised_volatility",
    "to_log_returns",
]
