r"""
Stage 4 Mode B — Monte Carlo daily-path simulation.

Where :mod:`daily.kalman` produces a *single* conditional-expectation path
(used for backtests and point estimates), this module produces *N*
constraint-respecting daily paths used for VaR / CVaR / drawdown
distributions.

Algorithm
---------
For each strategy and each simulation path:

    1. Compute the daily systematic component
       :math:`\mathrm{sys}_d = \beta \cdot \log(1 + F_d) + \alpha / n_d`
       in workspace (log-returns).
    2. Aggregate to monthly: :math:`\mathrm{sys}_m = \sum_{d \in m}
       \mathrm{sys}_d`.
    3. The target idiosyncratic for the month is
       :math:`\varepsilon_m = \log(1 + r_m) - \mathrm{sys}_m`.
    4. Draw N raw daily idiosyncratic shocks from the per-strategy fitted
       residual distribution, scaled to daily frequency.
    5. **Constrain**: subtract each block's mean and add
       :math:`\varepsilon_m / n_d` so that
       :math:`\sum_{d \in m} \varepsilon^{\mathrm{adj}}_d = \varepsilon_m`
       holds exactly.
    6. Daily workspace return = ``sys_d + ε_adj_d``; exponentiate to simple.

Multi-strategy correlation
--------------------------
When more than one strategy is supplied, raw shocks are drawn jointly so
that the cross-strategy correlation matches the supplied
``cross_strategy_correlation`` (typically the post-Stage-2 residual
correlation). The implementation Cholesky-rotates iid Gaussian draws with
matching marginal scale, which preserves linear correlation but not higher-
order tail dependence — the spec accepts this trade-off ("draw from joint
distribution using estimated cross-strategy residual correlation").

Constraint enforcement
----------------------
The block-mean centering trick preserves *both*

    * the deterministic monthly idiosyncratic (the constraint), and
    * the *within-block* sampled deviations from the residual distribution

so the resulting paths have realistic per-day variability while exactly
respecting the multiplicative aggregation. This is the same construction
used by :mod:`disaggregation.multivariate` for cross-sectional rotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.protocols import AggregationType
from ..decomposition.residual_analysis import FittedDistribution
from .calendar import block_sizes_for_months

# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SimulationResult:
    """Output of :meth:`DailySimulator.simulate`.

    Attributes
    ----------
    paths
        Dict ``{strategy_name: np.ndarray}`` of shape ``(n_paths, n_daily)``
        with simple returns per day.
    daily_index
        Shared DatetimeIndex for all paths.
    n_paths
        Number of simulated paths.
    aggregation_error
        Maximum absolute aggregation error across paths and strategies.
    diagnostics
        Free-form mapping.
    """

    paths: dict[str, np.ndarray]
    daily_index: pd.DatetimeIndex
    n_paths: int
    aggregation_error: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────


@dataclass
class DailySimulator:
    """Stage 4 Mode B: Monte Carlo daily simulator with monthly constraint.

    Parameters
    ----------
    distribution_per_strategy
        Mapping ``{strategy_name: FittedDistribution}``. The distribution is
        used to draw *unit-scale* daily shocks; the simulator handles
        scaling so that the within-block variability is plausible.
    cross_strategy_correlation
        Optional ``(n_strats × n_strats)`` correlation matrix indexed by
        strategy names. ``None`` (default) → independent shocks across
        strategies. Pass the residual-correlation matrix from
        :func:`decomposition.residual_analysis.cross_strategy_covariance`
        (after standardising to correlation) for realistic joint behaviour.
    daily_scale_factor
        Factor by which the per-strategy fitted distribution scale is
        multiplied to obtain daily-frequency shocks. Default
        ``1 / sqrt(21)`` (≈ 0.218) reflects the typical 21 business days
        per month under a iid scaling assumption. Override when the daily
        process is non-iid.
    aggregation
        ``MULTIPLICATIVE`` (default) or ``ADDITIVE``.
    """

    distribution_per_strategy: dict[str, FittedDistribution]
    cross_strategy_correlation: pd.DataFrame | None = None
    daily_scale_factor: float = 1.0 / np.sqrt(21.0)
    aggregation: AggregationType | str = AggregationType.MULTIPLICATIVE

    def __post_init__(self) -> None:
        if not isinstance(self.distribution_per_strategy, dict) or not self.distribution_per_strategy:
            raise ValueError("distribution_per_strategy must be a non-empty dict.")
        if self.daily_scale_factor <= 0.0:
            raise ValueError(f"daily_scale_factor must be > 0, got {self.daily_scale_factor}")
        self.aggregation = (
            AggregationType(self.aggregation)
            if not isinstance(self.aggregation, AggregationType)
            else self.aggregation
        )
        if self.cross_strategy_correlation is not None:
            if not isinstance(self.cross_strategy_correlation, pd.DataFrame):
                raise TypeError(
                    "cross_strategy_correlation must be a DataFrame indexed by strategy."
                )
            corr = self.cross_strategy_correlation.to_numpy()
            if corr.shape[0] != corr.shape[1]:
                raise ValueError("cross_strategy_correlation must be square.")
            # Will validate symmetry / PSD lazily when used (Cholesky factors check).

    def simulate(
        self,
        monthly_returns: pd.DataFrame,
        daily_factor_returns: pd.DataFrame,
        factor_betas_per_strategy: dict[str, dict[str, float]],
        *,
        alpha_per_strategy: dict[str, float] | None = None,
        n_paths: int = 1000,
        rng: np.random.Generator | int | None = None,
        daily_index: pd.DatetimeIndex | None = None,
    ) -> SimulationResult:
        """Generate ``n_paths`` constraint-respecting daily paths.

        Parameters
        ----------
        monthly_returns
            DataFrame with one column per strategy, indexed monthly.
        daily_factor_returns
            DataFrame of daily factor returns. Must contain every factor
            referenced in ``factor_betas_per_strategy``.
        factor_betas_per_strategy
            Mapping ``{strategy_name: {factor_name: β_k}}`` from Stage 2.
        alpha_per_strategy
            Optional ``{strategy_name: α}``. Defaults to all-zero α.
        n_paths
            Number of simulation paths.
        rng
            Generator or seed.
        daily_index
            Optional explicit daily index. Defaults to
            ``daily_factor_returns.index``.

        Returns
        -------
        SimulationResult
        """
        if not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        if not isinstance(monthly_returns, pd.DataFrame) or monthly_returns.empty:
            raise TypeError("monthly_returns must be a non-empty DataFrame.")
        if monthly_returns.isna().any().any():
            raise ValueError("monthly_returns contains NaN.")
        if not isinstance(daily_factor_returns, pd.DataFrame) or daily_factor_returns.empty:
            raise TypeError("daily_factor_returns must be a non-empty DataFrame.")
        if daily_factor_returns.isna().any().any():
            raise ValueError("daily_factor_returns contains NaN.")

        strategies = list(monthly_returns.columns)
        missing_dist = [s for s in strategies if s not in self.distribution_per_strategy]
        if missing_dist:
            raise KeyError(
                f"distribution_per_strategy missing entries for {missing_dist!r}"
            )
        missing_betas = [s for s in strategies if s not in factor_betas_per_strategy]
        if missing_betas:
            raise KeyError(
                f"factor_betas_per_strategy missing entries for {missing_betas!r}"
            )
        alpha_per_strategy = alpha_per_strategy or {s: 0.0 for s in strategies}

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

        # Block sizes (business days per month)
        block_sizes = block_sizes_for_months(hf_index, monthly_returns.index)
        n_d = len(hf_index)
        n_m = len(monthly_returns)

        # Workspace conversion
        if self.aggregation is AggregationType.MULTIPLICATIVE:
            if (monthly_returns.to_numpy() <= -1.0).any() or (
                daily_factor_returns.to_numpy() <= -1.0
            ).any():
                raise ValueError("multiplicative workspace requires returns > -1.")
            r_m_ws = np.log1p(monthly_returns.to_numpy(dtype=float))
            F_ws_full = np.log1p(daily_factor_returns.to_numpy(dtype=float))
        else:
            r_m_ws = monthly_returns.to_numpy(dtype=float)
            F_ws_full = daily_factor_returns.to_numpy(dtype=float)
        F_columns = list(daily_factor_returns.columns)

        # Pre-compute per-strategy systematic & idiosyncratic monthly target
        sys_d_per_strategy: dict[str, np.ndarray] = {}
        eps_m_per_strategy: dict[str, np.ndarray] = {}
        for s in strategies:
            betas = factor_betas_per_strategy[s]
            missing_factors = [k for k in betas if k not in F_columns]
            if missing_factors:
                raise KeyError(
                    f"daily_factor_returns missing columns {missing_factors!r} for strategy {s!r}"
                )
            beta_arr = np.array([betas[k] for k in betas], dtype=float)
            F_subset = F_ws_full[:, [F_columns.index(k) for k in betas]]
            sys_d = F_subset @ beta_arr
            alpha = float(alpha_per_strategy.get(s, 0.0))
            if alpha != 0.0:
                alpha_d = np.empty(n_d)
                start = 0
                for b, sz in enumerate(block_sizes):
                    alpha_d[start : start + sz] = alpha / sz
                    start += int(sz)
                sys_d = sys_d + alpha_d
            sys_d_per_strategy[s] = sys_d
            sys_m = _aggregate_blocks(sys_d, block_sizes)
            eps_m_per_strategy[s] = r_m_ws[:, strategies.index(s)] - sys_m

        # Pre-compute per-strategy daily-scaled distributions for sampling
        # Default approach: keep the FittedDistribution but scale draws by daily_scale_factor
        # since the distribution was fit at monthly frequency.

        # Cross-strategy correlation Cholesky factor (if provided)
        L_corr: np.ndarray | None = None
        if self.cross_strategy_correlation is not None and len(strategies) >= 2:
            corr = self.cross_strategy_correlation.reindex(
                index=strategies, columns=strategies
            )
            if corr.isna().any().any():
                raise ValueError(
                    "cross_strategy_correlation does not cover all strategies."
                )
            corr_arr = corr.to_numpy()
            try:
                L_corr = np.linalg.cholesky(corr_arr + 1e-12 * np.eye(len(strategies)))
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"cross_strategy_correlation is not positive definite: {exc}"
                ) from exc

        # Run simulation
        paths: dict[str, np.ndarray] = {s: np.empty((n_paths, n_d)) for s in strategies}
        n_strats = len(strategies)

        # Per-strategy "scale factor" so the daily marginal vol is daily_scale_factor × monthly_scale
        scales = np.array(
            [
                self.distribution_per_strategy[s].params.get("scale", 1.0)
                * self.daily_scale_factor
                for s in strategies
            ]
        )

        for path in range(n_paths):
            # Draw raw daily shocks from the per-strategy distribution.
            # For independent draws we simply call .sample on each. For
            # correlated draws we Cholesky-rotate standardised draws.
            if L_corr is None or n_strats < 2:
                raw_shocks = np.zeros((n_d, n_strats))
                for k, s in enumerate(strategies):
                    dist = self.distribution_per_strategy[s]
                    draws = dist.sample(size=n_d, rng=rng)
                    # Centre and scale to (0, daily_scale_factor·dist.scale)
                    loc = dist.params.get("loc", 0.0)
                    scl = dist.params.get("scale", 1.0)
                    standard = (draws - loc) / scl
                    raw_shocks[:, k] = standard * scales[k]
            else:
                # Standard Gaussian via Cholesky for correlation, then map to per-strategy
                # marginal via inverse-CDF (Gaussian copula). This preserves linear
                # correlation up to the copula-vs-marginal transform.
                z = rng.standard_normal(size=(n_d, n_strats))
                z_corr = z @ L_corr.T
                from scipy import stats

                u = stats.norm.cdf(z_corr)
                # Clamp to keep the inverse-CDF stable
                u = np.clip(u, 1e-9, 1.0 - 1e-9)
                raw_shocks = np.empty((n_d, n_strats))
                for k, s in enumerate(strategies):
                    dist = self.distribution_per_strategy[s]
                    draws = _quantile(dist, u[:, k])
                    loc = dist.params.get("loc", 0.0)
                    scl = dist.params.get("scale", 1.0)
                    standard = (draws - loc) / scl
                    raw_shocks[:, k] = standard * scales[k]

            # Constraint enforcement: per (strategy, block) replace the block
            # mean with eps_m_per_strategy[s][b] / n_d_in_block.
            for k, s in enumerate(strategies):
                shocks = raw_shocks[:, k]
                eps_m = eps_m_per_strategy[s]
                eps_d = _enforce_block_sum(shocks, block_sizes, eps_m)
                # Daily workspace return = sys + idio
                r_d_ws = sys_d_per_strategy[s] + eps_d
                # Convert to simple
                if self.aggregation is AggregationType.MULTIPLICATIVE:
                    paths[s][path] = np.expm1(r_d_ws)
                else:
                    paths[s][path] = r_d_ws

        # Aggregation error: for each path and strategy, max |aggregate(daily) - monthly|
        max_err = 0.0
        for s in strategies:
            for p in range(n_paths):
                back = _aggregate_blocks_returns(
                    paths[s][p], block_sizes, self.aggregation
                )
                err = float(
                    np.max(np.abs(back - monthly_returns[s].to_numpy()))
                )
                max_err = max(max_err, err)

        return SimulationResult(
            paths=paths,
            daily_index=hf_index,
            n_paths=n_paths,
            aggregation_error=max_err,
            diagnostics={
                "n_monthly": int(n_m),
                "n_daily": int(n_d),
                "block_sizes": block_sizes.copy(),
                "aggregation": self.aggregation.value,
                "strategies": strategies,
                "daily_scale_factor": self.daily_scale_factor,
            },
        )


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _aggregate_blocks(daily: np.ndarray, block_sizes: np.ndarray) -> np.ndarray:
    """Sum each block of ``daily`` (workspace, additive)."""
    out = np.empty(len(block_sizes))
    start = 0
    for b, sz in enumerate(block_sizes):
        out[b] = float(daily[start : start + sz].sum())
        start += int(sz)
    return out


def _aggregate_blocks_returns(
    daily: np.ndarray,
    block_sizes: np.ndarray,
    aggregation: AggregationType,
) -> np.ndarray:
    """Aggregate daily *simple* returns into monthly using the configured rule."""
    if aggregation is AggregationType.MULTIPLICATIVE:
        log_d = np.log1p(daily)
        out = np.empty(len(block_sizes))
        start = 0
        for b, sz in enumerate(block_sizes):
            out[b] = float(np.expm1(log_d[start : start + sz].sum()))
            start += int(sz)
        return out
    return _aggregate_blocks(daily, block_sizes)


def _enforce_block_sum(
    shocks: np.ndarray,
    block_sizes: np.ndarray,
    target_block_sums: np.ndarray,
) -> np.ndarray:
    """Adjust shocks so each block sums to the corresponding target.

    Subtracts each block's mean and adds ``target_block_sums[b] / n_d``.
    Preserves within-block deviations exactly.
    """
    out = np.empty_like(shocks)
    start = 0
    for b, sz in enumerate(block_sizes):
        block = shocks[start : start + sz]
        block_mean = float(block.mean())
        target_per_day = float(target_block_sums[b]) / float(sz)
        out[start : start + sz] = block - block_mean + target_per_day
        start += int(sz)
    return out


def _quantile(dist: FittedDistribution, u: np.ndarray) -> np.ndarray:
    """Quantile function (inverse CDF) for a fitted distribution.

    Falls back to inverse-CDF on a fine grid for the Hansen skewed-t —
    cheaper than re-fitting and accurate to a few bps for typical use.
    """
    from scipy import stats

    if dist.name == "normal":
        return stats.norm.ppf(u, loc=dist.params["loc"], scale=dist.params["scale"])
    if dist.name == "student_t":
        return stats.t.ppf(
            u,
            df=dist.params["df"],
            loc=dist.params["loc"],
            scale=dist.params["scale"],
        )
    # Skewed-t: build CDF on a fine grid
    z_grid = np.linspace(-15.0, 15.0, 4001)
    pdf = dist.pdf(dist.params["loc"] + dist.params["scale"] * z_grid)
    cdf = np.cumsum(pdf) * (z_grid[1] - z_grid[0])
    cdf /= cdf[-1]
    z = np.interp(u, cdf, z_grid)
    return dist.params["loc"] + dist.params["scale"] * z


__all__ = ["DailySimulator", "SimulationResult"]
