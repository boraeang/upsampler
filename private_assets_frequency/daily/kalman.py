r"""
Stage 4 Mode A — Kalman smoother for daily disaggregation of monthly returns.

Given monthly returns from Stage 3 (post temporal disaggregation) and a
daily factor matrix, produce a daily return path :math:`r_d` such that

    1. **Constraint** holds exactly: within each month the daily simple
       returns compound to the monthly return,
       :math:`(1 + r_m) = \prod_{d \in m}(1 + r_d)`.
    2. **Systematic component** matches the factor model:
       :math:`r_d \approx \beta \cdot F_d + \alpha + \varepsilon_d`.
    3. **Idiosyncratic component** :math:`\varepsilon_d` follows an AR(1)
       process with user-specified persistence :math:`\phi`.

The smoothed series is the conditional expectation
:math:`\mathbb{E}[\varepsilon_d \mid \varepsilon_m]` under the model. For
Gaussian state-space models with linear aggregation, the Rauch-Tung-Striebel
recursion and the GLS / BLUE formula give the same answer; this module
implements the GLS formulation directly because it is shorter to read and
the dense linear algebra is fine at the typical Stage-4 scale (a few years
of daily data, low thousands of states). Long-span runs should switch to
banded scipy primitives — flagged in the docstring.

Workspace
---------
All linear algebra runs in *log-return* space::

    log(1 + r_d) = β · log(1 + F_d) + α/n_d + ε_d
    Σ_{d ∈ month} log(1 + r_d) = log(1 + r_m)         (constraint)

so the multiplicative aggregation constraint becomes additive in workspace
and the BLUE solution preserves it to machine precision. The output is
exponentiated back to simple returns at the boundary.

Note on intercept handling
~~~~~~~~~~~~~~~~~~~~~~~~~~
``α`` is provided at the *monthly* frequency by Stage 2; we distribute it
linearly across the days inside each month (``α_per_day = α / n_d``). For
small ``α`` (the typical case at any sensible halflife) this is the right
disaggregation; large ``α`` should ring alarm bells anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..core.protocols import AggregationType
from ..disaggregation.aggregation import ar1_covariance
from ..utils.returns import aggregate_returns
from .calendar import block_sizes_for_months, irregular_aggregation_matrix

# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DailyDisaggregationResult:
    """Output of :meth:`KalmanDailySmoother.fit`.

    Attributes
    ----------
    daily_returns
        High-frequency simple-return series indexed at the supplied daily
        index.
    daily_systematic
        Systematic component ``β · F_d + α/n_d`` per day (in simple-return
        space — exponentiated from the workspace systematic).
    daily_idiosyncratic
        Idiosyncratic component (smoothed) per day in workspace (log-return)
        space — i.e. the AR(1) component before exponentiation.
    monthly_input
        Echo of the input monthly returns.
    aggregation_error
        Maximum absolute round-trip error: max over months of
        ``|aggregate(daily) - monthly|``.
    diagnostics
        Free-form mapping (block sizes, AR(1) ρ used, factor R², …).
    """

    daily_returns: pd.Series
    daily_systematic: pd.Series
    daily_idiosyncratic: pd.Series
    monthly_input: pd.Series
    aggregation_error: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Smoother
# ──────────────────────────────────────────────────────────────────


@dataclass
class KalmanDailySmoother:
    """Kalman smoother (BLUE form) for daily disaggregation of monthly returns.

    Parameters
    ----------
    phi
        AR(1) persistence of the *daily* idiosyncratic series. Default 0
        (white noise within each month). Must lie in ``(-1, 1)``.
    aggregation
        ``MULTIPLICATIVE`` (default) or ``ADDITIVE``. Affects only how
        returns are interpreted; the linear algebra is the same in
        workspace.
    intercept_distribution
        How to spread monthly ``α`` across the days of each month. Default
        ``'uniform'`` (``α / n_d`` per day); ``'none'`` skips α distribution
        (use when α=0 anyway).
    """

    phi: float = 0.0
    aggregation: AggregationType | str = AggregationType.MULTIPLICATIVE
    intercept_distribution: Literal["uniform", "none"] = "uniform"

    def __post_init__(self) -> None:
        if not (-0.999 <= self.phi <= 0.999):
            raise ValueError(f"phi must lie in (-1, 1), got {self.phi}")
        self.aggregation = (
            AggregationType(self.aggregation)
            if not isinstance(self.aggregation, AggregationType)
            else self.aggregation
        )
        if self.intercept_distribution not in ("uniform", "none"):
            raise ValueError(
                f"intercept_distribution must be 'uniform' or 'none', "
                f"got {self.intercept_distribution!r}"
            )

    def fit(
        self,
        monthly_returns: pd.Series,
        daily_factor_returns: pd.DataFrame,
        factor_betas: dict[str, float],
        *,
        alpha: float = 0.0,
        daily_index: pd.DatetimeIndex | None = None,
    ) -> DailyDisaggregationResult:
        """Disaggregate ``monthly_returns`` to daily.

        Parameters
        ----------
        monthly_returns
            Monthly simple-return series at the strategy's native frequency.
        daily_factor_returns
            Daily factor returns. The DataFrame's columns must include
            every key in ``factor_betas``; columns not present in
            ``factor_betas`` are ignored.
        factor_betas
            Mapping ``{factor_name: β_k}`` from Stage 2.
        alpha
            Monthly intercept from Stage 2. Distributed across days per
            ``intercept_distribution``.
        daily_index
            Optional explicit daily index. Defaults to
            ``daily_factor_returns.index``.

        Returns
        -------
        DailyDisaggregationResult
        """
        m_returns, F_daily, hf_index, factor_names, block_sizes = self._validate(
            monthly_returns, daily_factor_returns, factor_betas, daily_index
        )
        n_m = len(m_returns)
        n_d = len(hf_index)

        # ── Workspace: log-returns ─────────────────────────────────
        if self.aggregation is AggregationType.MULTIPLICATIVE:
            if (m_returns.to_numpy() <= -1.0).any() or (F_daily <= -1.0).any():
                raise ValueError(
                    "multiplicative workspace requires all returns > -1."
                )
            r_m_ws = np.log1p(m_returns.to_numpy(dtype=float))
            F_ws = np.log1p(F_daily)
        else:
            r_m_ws = m_returns.to_numpy(dtype=float)
            F_ws = F_daily

        # ── Daily systematic component (workspace) ─────────────────
        beta_arr = np.array([factor_betas[k] for k in factor_names], dtype=float)
        sys_d_ws = F_ws @ beta_arr
        if alpha != 0.0 and self.intercept_distribution == "uniform":
            # Distribute α/n_d per day, summing to α per month
            alpha_d = np.empty(n_d)
            start = 0
            for b, sz in enumerate(block_sizes):
                alpha_d[start : start + sz] = alpha / sz
                start += sz
            sys_d_ws = sys_d_ws + alpha_d

        # ── Monthly aggregated systematic (workspace) ──────────────
        C = irregular_aggregation_matrix(block_sizes)
        sys_m_ws = C @ sys_d_ws

        # ── Idiosyncratic monthly target (workspace) ───────────────
        idio_m_ws = r_m_ws - sys_m_ws

        # ── Disaggregate idiosyncratic via BLUE / GLS with AR(1) V ─
        V = ar1_covariance(n_d, float(self.phi))
        V_lf = C @ V @ C.T
        # Smoothed idiosyncratic: ε_d = V·C'·V_lf⁻¹·idio_m
        Vct = V @ C.T
        idio_d_ws = Vct @ np.linalg.solve(V_lf, idio_m_ws)

        # ── Daily workspace return = sys + idio ────────────────────
        r_d_ws = sys_d_ws + idio_d_ws

        # ── Convert back to simple returns ─────────────────────────
        if self.aggregation is AggregationType.MULTIPLICATIVE:
            r_d = np.expm1(r_d_ws)
            sys_d = np.expm1(sys_d_ws)
        else:
            r_d = r_d_ws.copy()
            sys_d = sys_d_ws.copy()

        # ── Round-trip check: monthly aggregate of daily ≈ input ──
        # We use the same aggregation as configured, on the actual blocks.
        agg_back = _aggregate_irregular(
            r_d, block_sizes, aggregation=self.aggregation
        )
        agg_err = float(np.max(np.abs(agg_back - m_returns.to_numpy())))

        diagnostics = {
            "method": "kalman_blue",
            "phi": float(self.phi),
            "n_monthly": int(n_m),
            "n_daily": int(n_d),
            "block_sizes": block_sizes.copy(),
            "aggregation": self.aggregation.value,
            "factor_names": factor_names,
            "betas": dict(factor_betas),
            "alpha": float(alpha),
            "aggregation_error": agg_err,
        }

        return DailyDisaggregationResult(
            daily_returns=pd.Series(r_d, index=hf_index, name=monthly_returns.name),
            daily_systematic=pd.Series(sys_d, index=hf_index, name="systematic"),
            daily_idiosyncratic=pd.Series(
                idio_d_ws, index=hf_index, name="idiosyncratic"
            ),
            monthly_input=monthly_returns.copy(),
            aggregation_error=agg_err,
            diagnostics=diagnostics,
        )

    # ── Internals ───────────────────────────────────────────────────

    def _validate(
        self,
        monthly_returns: pd.Series,
        daily_factor_returns: pd.DataFrame,
        factor_betas: dict[str, float],
        daily_index: pd.DatetimeIndex | None,
    ) -> tuple[
        pd.Series, np.ndarray, pd.DatetimeIndex, list[str], np.ndarray
    ]:
        if not isinstance(monthly_returns, pd.Series):
            raise TypeError(
                f"monthly_returns must be a Series, got {type(monthly_returns).__name__}"
            )
        if not isinstance(daily_factor_returns, pd.DataFrame) or daily_factor_returns.empty:
            raise TypeError(
                "daily_factor_returns must be a non-empty DataFrame."
            )
        if monthly_returns.isna().any():
            raise ValueError("monthly_returns contains NaN.")
        if not isinstance(factor_betas, dict) or not factor_betas:
            raise ValueError("factor_betas must be a non-empty dict.")
        missing = [k for k in factor_betas if k not in daily_factor_returns.columns]
        if missing:
            raise KeyError(f"daily_factor_returns missing columns {missing!r}")

        hf_index = (
            daily_index
            if daily_index is not None
            else daily_factor_returns.index
        )
        if not isinstance(hf_index, pd.DatetimeIndex):
            raise TypeError("daily_index must be a DatetimeIndex.")
        if len(hf_index) != len(daily_factor_returns):
            raise ValueError(
                f"daily_index length {len(hf_index)} != daily_factor_returns length "
                f"{len(daily_factor_returns)}"
            )

        # Determine block sizes per month
        block_sizes = block_sizes_for_months(hf_index, monthly_returns.index)
        if int(block_sizes.sum()) != len(hf_index):
            # Defensive: should already be caught by block_sizes_for_months
            raise ValueError(
                f"block sizes sum to {int(block_sizes.sum())} but daily index has "
                f"{len(hf_index)} entries."
            )

        # Subset & order factor columns to match factor_betas dict order
        factor_names = list(factor_betas)
        F_daily = daily_factor_returns[factor_names].reindex(hf_index)
        if F_daily.isna().any().any():
            raise ValueError("daily_factor_returns has NaN after reindex.")
        return monthly_returns, F_daily.to_numpy(dtype=float), hf_index, factor_names, block_sizes


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _aggregate_irregular(
    daily: np.ndarray,
    block_sizes: np.ndarray,
    aggregation: AggregationType,
) -> np.ndarray:
    """Aggregate a 1-D daily series into per-block totals with variable block sizes."""
    if aggregation is AggregationType.MULTIPLICATIVE:
        log_d = np.log1p(daily)
        out = np.empty(len(block_sizes))
        start = 0
        for b, sz in enumerate(block_sizes):
            out[b] = float(np.expm1(log_d[start : start + sz].sum()))
            start += int(sz)
        return out
    out = np.empty(len(block_sizes))
    start = 0
    for b, sz in enumerate(block_sizes):
        out[b] = float(daily[start : start + sz].sum())
        start += int(sz)
    return out


__all__ = ["DailyDisaggregationResult", "KalmanDailySmoother"]
