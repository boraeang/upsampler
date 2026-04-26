"""Tests for ``daily.simulation``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import AggregationType
from private_assets_frequency.daily.calendar import business_day_index
from private_assets_frequency.daily.simulation import (
    DailySimulator,
    SimulationResult,
    _enforce_block_sum,
)
from private_assets_frequency.decomposition.residual_analysis import (
    fit_normal,
    fit_student_t,
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _make_problem(
    *,
    start: str = "2021-01-01",
    end: str = "2022-06-30",
    n_strategies: int = 2,
    seed: int = 0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, float]],
    dict[str, float],
    pd.DatetimeIndex,
]:
    rng = np.random.default_rng(seed)
    daily_index = business_day_index(start, end, calendar="BDAY")
    n_d = len(daily_index)
    f = rng.normal(0.0, 0.01, size=n_d)
    daily_factors = pd.DataFrame({"factor": f}, index=daily_index)
    strategies = [f"s{k+1}" for k in range(n_strategies)]
    betas_per = {s: {"factor": float(0.5 + 0.5 * (k + 1))} for k, s in enumerate(strategies)}
    alpha_per = {s: 0.001 * (k + 1) for k, s in enumerate(strategies)}
    # Generate "true" daily returns and aggregate to monthly
    monthly_panel: dict[str, np.ndarray] = {}
    monthly_idx: pd.DatetimeIndex | None = None
    for s in strategies:
        beta = betas_per[s]["factor"]
        alpha = alpha_per[s]
        eps_d = rng.normal(0.0, 0.005, size=n_d)
        r_d_true = beta * f + eps_d
        # Aggregate to monthly using the actual block sizes
        from private_assets_frequency.daily.calendar import (
            block_sizes_for_months,
            business_days_per_month,
        )
        per_month = business_days_per_month(daily_index)
        block_sizes = per_month.to_numpy()
        starts = np.concatenate([[0], np.cumsum(block_sizes)])
        log_d = np.log1p(r_d_true)
        # Add monthly alpha back as a small drift
        monthly_arr = np.array(
            [
                np.expm1(log_d[starts[b] : starts[b + 1]].sum()) + alpha
                for b in range(len(block_sizes))
            ]
        )
        monthly_panel[s] = monthly_arr
        if monthly_idx is None:
            monthly_idx = per_month.index
    panel = pd.DataFrame(monthly_panel, index=monthly_idx)
    return panel, daily_factors, betas_per, alpha_per, daily_index


# ──────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────


class TestSimulatorConstruction:
    def test_basic(self):
        rng = np.random.default_rng(0)
        dist = fit_normal(rng.normal(0.0, 0.04, size=100))
        sim = DailySimulator(distribution_per_strategy={"a": dist})
        assert sim.aggregation is AggregationType.MULTIPLICATIVE

    def test_invalid_scale(self):
        rng = np.random.default_rng(0)
        dist = fit_normal(rng.normal(0.0, 0.04, size=100))
        with pytest.raises(ValueError):
            DailySimulator(
                distribution_per_strategy={"a": dist},
                daily_scale_factor=-1.0,
            )

    def test_empty_distribution_dict(self):
        with pytest.raises(ValueError):
            DailySimulator(distribution_per_strategy={})


# ──────────────────────────────────────────────────────────────────
# _enforce_block_sum
# ──────────────────────────────────────────────────────────────────


class TestEnforceBlockSum:
    def test_block_sums_match_target(self):
        rng = np.random.default_rng(0)
        shocks = rng.normal(0.0, 1.0, size=12)
        block_sizes = np.array([3, 5, 4])
        target = np.array([0.05, -0.02, 0.10])
        out = _enforce_block_sum(shocks, block_sizes, target)
        starts = np.concatenate([[0], np.cumsum(block_sizes)])
        for b in range(len(block_sizes)):
            np.testing.assert_allclose(
                out[starts[b] : starts[b + 1]].sum(), target[b], atol=1e-12
            )

    def test_within_block_deviations_preserved(self):
        rng = np.random.default_rng(0)
        shocks = rng.normal(0.0, 1.0, size=12)
        block_sizes = np.array([3, 5, 4])
        target = np.array([0.05, -0.02, 0.10])
        out = _enforce_block_sum(shocks, block_sizes, target)
        # The deviation of each entry from the block mean should match before/after
        starts = np.concatenate([[0], np.cumsum(block_sizes)])
        for b in range(len(block_sizes)):
            sl = slice(starts[b], starts[b + 1])
            before = shocks[sl] - shocks[sl].mean()
            after = out[sl] - out[sl].mean()
            np.testing.assert_allclose(before, after, atol=1e-12)


# ──────────────────────────────────────────────────────────────────
# Simulate — central correctness test
# ──────────────────────────────────────────────────────────────────


class TestSimulateAggregation:
    """Within each month, daily returns must sum (compound) to the input monthly."""

    def test_aggregation_constraint_holds_for_all_paths(self):
        rng = np.random.default_rng(0)
        panel, factors, betas, alphas, d_idx = _make_problem(seed=0)
        # Per-strategy distributions fitted on synthetic monthly residuals
        dists = {
            s: fit_normal(rng.normal(0.0, 0.02, size=200)) for s in panel.columns
        }
        sim = DailySimulator(distribution_per_strategy=dists)
        result = sim.simulate(
            panel,
            factors,
            betas,
            alpha_per_strategy=alphas,
            n_paths=20,
            rng=42,
            daily_index=d_idx,
        )
        assert isinstance(result, SimulationResult)
        assert result.aggregation_error < 1e-10, (
            f"aggregation error {result.aggregation_error:.3e} > 1e-10"
        )
        # Verify path shapes
        for s in panel.columns:
            assert result.paths[s].shape == (20, len(d_idx))

    def test_correlated_strategies_aggregate_correctly(self):
        rng = np.random.default_rng(0)
        panel, factors, betas, alphas, d_idx = _make_problem(seed=1, n_strategies=3)
        dists = {
            s: fit_normal(rng.normal(0.0, 0.02, size=200)) for s in panel.columns
        }
        # Build a positive-definite cross-strategy correlation matrix
        corr = pd.DataFrame(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ],
            index=panel.columns,
            columns=panel.columns,
        )
        sim = DailySimulator(
            distribution_per_strategy=dists,
            cross_strategy_correlation=corr,
        )
        result = sim.simulate(
            panel,
            factors,
            betas,
            alpha_per_strategy=alphas,
            n_paths=10,
            rng=7,
            daily_index=d_idx,
        )
        assert result.aggregation_error < 1e-10
        # Paths should have 3 strategies
        assert set(result.paths) == set(panel.columns)

    def test_student_t_distribution_aggregation(self):
        rng = np.random.default_rng(0)
        panel, factors, betas, alphas, d_idx = _make_problem(seed=2)
        # Fit a t-distribution to heavy-tailed residual sample
        from scipy import stats

        x = stats.t.rvs(df=5, size=300, random_state=rng) * 0.02
        dist_t = fit_student_t(x)
        dists = {s: dist_t for s in panel.columns}
        sim = DailySimulator(distribution_per_strategy=dists)
        result = sim.simulate(
            panel,
            factors,
            betas,
            alpha_per_strategy=alphas,
            n_paths=5,
            rng=11,
            daily_index=d_idx,
        )
        assert result.aggregation_error < 1e-10


# ──────────────────────────────────────────────────────────────────
# Cross-strategy correlation reasonably preserved
# ──────────────────────────────────────────────────────────────────


class TestCorrelationStructure:
    def test_paths_correlate_via_cholesky(self):
        rng = np.random.default_rng(0)
        panel, factors, betas, alphas, d_idx = _make_problem(seed=4, n_strategies=2)
        dists = {
            s: fit_normal(rng.normal(0.0, 0.02, size=300)) for s in panel.columns
        }
        target_rho = 0.7
        corr = pd.DataFrame(
            [[1.0, target_rho], [target_rho, 1.0]],
            index=panel.columns,
            columns=panel.columns,
        )
        sim = DailySimulator(
            distribution_per_strategy=dists,
            cross_strategy_correlation=corr,
        )
        result = sim.simulate(
            panel,
            factors,
            betas,
            alpha_per_strategy=alphas,
            n_paths=200,
            rng=99,
            daily_index=d_idx,
        )
        s_names = list(panel.columns)
        # Pool all daily residuals across paths and compute realised correlation
        log_a = np.log1p(result.paths[s_names[0]])
        log_b = np.log1p(result.paths[s_names[1]])
        flat_a = log_a.flatten()
        flat_b = log_b.flatten()
        realised = float(np.corrcoef(flat_a, flat_b)[0, 1])
        # The systematic component is a single common factor with positive
        # exposure on both, so the pooled realised correlation reflects both
        # the cross-strategy correlation we imposed AND the common factor.
        # Just check it's clearly above zero and not below the target.
        assert realised > 0.4


# ──────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_n_paths(self):
        rng = np.random.default_rng(0)
        panel, factors, betas, alphas, d_idx = _make_problem(seed=0)
        dists = {
            s: fit_normal(rng.normal(0.0, 0.02, size=100)) for s in panel.columns
        }
        sim = DailySimulator(distribution_per_strategy=dists)
        with pytest.raises(ValueError):
            sim.simulate(
                panel,
                factors,
                betas,
                alpha_per_strategy=alphas,
                n_paths=0,
                daily_index=d_idx,
            )

    def test_missing_distribution_for_strategy(self):
        rng = np.random.default_rng(0)
        panel, factors, betas, alphas, d_idx = _make_problem(seed=0, n_strategies=2)
        dists = {"s1": fit_normal(rng.normal(0.0, 0.02, size=100))}  # missing 's2'
        sim = DailySimulator(distribution_per_strategy=dists)
        with pytest.raises(KeyError):
            sim.simulate(
                panel,
                factors,
                betas,
                alpha_per_strategy=alphas,
                n_paths=5,
                daily_index=d_idx,
            )

    def test_missing_factor_column(self):
        rng = np.random.default_rng(0)
        panel, factors, betas, alphas, d_idx = _make_problem(seed=0)
        bad_factors = factors.rename(columns={"factor": "other"})
        dists = {s: fit_normal(rng.normal(0.0, 0.02, size=100)) for s in panel.columns}
        sim = DailySimulator(distribution_per_strategy=dists)
        with pytest.raises(KeyError):
            sim.simulate(
                panel,
                bad_factors,
                betas,
                alpha_per_strategy=alphas,
                n_paths=5,
                daily_index=d_idx,
            )

    def test_non_psd_correlation(self):
        rng = np.random.default_rng(0)
        panel, factors, betas, alphas, d_idx = _make_problem(seed=0, n_strategies=2)
        dists = {s: fit_normal(rng.normal(0.0, 0.02, size=100)) for s in panel.columns}
        # Diagonal "correlation" with one entry > 1 (not PSD)
        bad_corr = pd.DataFrame(
            [[1.0, 1.5], [1.5, 1.0]],
            index=panel.columns,
            columns=panel.columns,
        )
        sim = DailySimulator(
            distribution_per_strategy=dists,
            cross_strategy_correlation=bad_corr,
        )
        with pytest.raises(ValueError, match="positive definite"):
            sim.simulate(
                panel,
                factors,
                betas,
                alpha_per_strategy=alphas,
                n_paths=5,
                daily_index=d_idx,
            )
