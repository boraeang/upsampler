r"""
Aggregation primitives for Stage 3 (Chow-Lin / Fernández / Litterman).

This module owns the pieces that any disaggregator needs *before* the
estimator-specific GLS:

    * The aggregation matrix ``C`` of shape ``(n_LF, n_HF)`` with
      ``y_LF = C · y_HF`` for additive aggregation, or
      ``log(1 + y_LF) = C · log(1 + y_HF)`` for multiplicative.
    * Three high-frequency residual covariance structures (up to scale σ²):
      AR(1) for Chow-Lin, random-walk for Fernández, AR(1)-on-differences
      for Litterman.
    * Helpers to convert between additive and multiplicative workspace —
      the spec mandates multiplicative compounding for return aggregation
      so the safe thing is to operate the linear algebra in log-return
      space and exponentiate at the boundary.

References
----------
.. [1] Chow & Lin (1971) — "Best linear unbiased interpolation,
       distribution, and extrapolation of time series by related series."
.. [2] Fernández (1981) — "A methodological note on the estimation of
       time series."
.. [3] Litterman (1983) — "A random walk, Markov model for the
       distribution of time series."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.protocols import AggregationType
from ..utils.returns import aggregation_matrix as _agg_matrix_base

# ──────────────────────────────────────────────────────────────────
# Aggregation matrix
# ──────────────────────────────────────────────────────────────────


def aggregation_matrix(n_low: int, ratio: int) -> np.ndarray:
    """Block-summing aggregation matrix ``C`` of shape ``(n_low, n_low * ratio)``.

    Each row of ``C`` selects a contiguous block of ``ratio`` columns and
    sums them. Re-exports :func:`utils.returns.aggregation_matrix` for
    discoverability inside :mod:`disaggregation`.
    """
    return _agg_matrix_base(n_low, ratio)


# ──────────────────────────────────────────────────────────────────
# Residual covariance structures (returned at unit σ²)
# ──────────────────────────────────────────────────────────────────


def ar1_covariance(n: int, rho: float) -> np.ndarray:
    r"""High-frequency AR(1) covariance matrix at unit σ².

    For :math:`u_t = \rho u_{t-1} + \varepsilon_t` with
    :math:`\varepsilon_t \sim \mathcal{N}(0, 1)`, the stationary covariance
    is :math:`V_{ij} = \rho^{|i-j|} / (1 - \rho^2)`.

    Used by Chow-Lin disaggregation. Pass through the result with σ² scaling
    at the GLS step.
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"|rho| must be < 1, got {rho}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    factor = 1.0 / (1.0 - rho**2)
    return (rho**diff) * factor


def random_walk_covariance(n: int) -> np.ndarray:
    r"""Random-walk (Fernández) covariance matrix at unit σ².

    For :math:`u_t = u_{t-1} + \varepsilon_t`, the covariance of
    :math:`(u_1, \dots, u_n)` is :math:`V_{ij} = \min(i, j)`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    idx = np.arange(1, n + 1, dtype=float)
    return np.minimum(idx[:, None], idx[None, :])


def litterman_covariance(n: int, rho: float) -> np.ndarray:
    r"""Litterman covariance: AR(1) on first differences.

    For :math:`\Delta u_t = \rho \Delta u_{t-1} + \varepsilon_t`, the level
    series :math:`u_t = \sum_{s \leq t} \Delta u_s` has covariance
    :math:`L \cdot V_{\mathrm{AR}(1)} \cdot L^\top` where ``L`` is the
    lower-triangular cumulative-sum matrix.

    At ``rho = 0`` Litterman reduces exactly to Fernández, so we can
    cross-check our implementations.
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"|rho| must be < 1, got {rho}")
    L = np.tril(np.ones((n, n)))
    V_ar1 = ar1_covariance(n, rho)
    return L @ V_ar1 @ L.T


# ──────────────────────────────────────────────────────────────────
# Aggregation-type / log-space conversion
# ──────────────────────────────────────────────────────────────────


def to_workspace(
    series: pd.Series | np.ndarray,
    aggregation: AggregationType | str,
) -> np.ndarray:
    """Convert a return series into the linear-algebra workspace.

    For ``MULTIPLICATIVE`` the return space transforms into log-returns so
    the aggregation constraint becomes additive (``log(1+r_LF) = Σ log(1+r_HF)``).
    For ``ADDITIVE`` the series is passed through.
    """
    arr = series.to_numpy() if isinstance(series, pd.Series) else np.asarray(series)
    agg = AggregationType(aggregation) if not isinstance(aggregation, AggregationType) else aggregation
    if agg is AggregationType.MULTIPLICATIVE:
        if (arr <= -1.0).any():
            raise ValueError("multiplicative workspace requires returns > -1.")
        return np.log1p(arr)
    return arr.astype(float, copy=True)


def from_workspace(
    array: np.ndarray,
    aggregation: AggregationType | str,
) -> np.ndarray:
    """Inverse of :func:`to_workspace`."""
    agg = AggregationType(aggregation) if not isinstance(aggregation, AggregationType) else aggregation
    if agg is AggregationType.MULTIPLICATIVE:
        return np.expm1(array)
    return np.asarray(array, dtype=float)


__all__ = [
    "aggregation_matrix",
    "ar1_covariance",
    "from_workspace",
    "litterman_covariance",
    "random_walk_covariance",
    "to_workspace",
]
