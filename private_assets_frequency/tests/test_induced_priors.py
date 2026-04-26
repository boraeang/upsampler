"""Tests for ``desmoothing.induced_priors``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import (
    BetaDist,
    InverseGammaPrior,
    NormalPrior,
)
from private_assets_frequency.desmoothing.ar1_bayesian import AR1BayesianSmoother
from private_assets_frequency.desmoothing.induced_priors import (
    InducedPriorsRunner,
    _beta_from_mean_var,
)
from private_assets_frequency.tests.conftest import make_ar1_pe_dataset


# ──────────────────────────────────────────────────────────────────
# Beta-from-moments helper
# ──────────────────────────────────────────────────────────────────


class TestBetaFromMeanVar:
    def test_round_trip(self):
        a, b = _beta_from_mean_var(0.6, 0.04)
        # Closed-form: mean = a/(a+b) = 0.6, var = ab/((a+b)^2 (a+b+1))
        s = a + b
        assert a / s == pytest.approx(0.6, abs=1e-9)
        assert (a * b) / (s * s * (s + 1.0)) == pytest.approx(0.04, abs=1e-9)

    def test_falls_back_when_var_too_large(self):
        # Var > mean*(1-mean) is infeasible
        a, b = _beta_from_mean_var(0.5, 0.5)
        assert (a, b) == (2.0, 2.0)

    def test_falls_back_when_mean_outside_unit(self):
        a, b = _beta_from_mean_var(1.5, 0.1)
        assert (a, b) == (2.0, 2.0)


# ──────────────────────────────────────────────────────────────────
# Runner basics
# ──────────────────────────────────────────────────────────────────


def _initial_priors(beta_mean: float = 1.0) -> dict:
    return {
        "lambda": BetaDist(2.0, 2.0),
        "beta": {"equity_market": NormalPrior(beta_mean, 0.5)},
        "alpha": NormalPrior(0.0, 0.05),
        "sigma_eps": InverseGammaPrior(3.0, 0.02),
    }


class TestInducedPriorsRunner:
    def test_default_construction(self):
        runner = InducedPriorsRunner(smoother_factory=AR1BayesianSmoother)
        assert runner.max_iter == 10
        assert runner.tol == 0.01
        assert runner.shrink_std is False

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            InducedPriorsRunner(
                smoother_factory=AR1BayesianSmoother, max_iter=0
            )
        with pytest.raises(ValueError):
            InducedPriorsRunner(
                smoother_factory=AR1BayesianSmoother, tol=0.0
            )

    def test_dataset_priors_keys_must_match(self):
        runner = InducedPriorsRunner(smoother_factory=AR1BayesianSmoother)
        ds = make_ar1_pe_dataset(rng=0)
        with pytest.raises(ValueError, match="differ"):
            runner.run(
                datasets={"a": (ds.observed_quarterly, ds.factor_quarterly)},
                initial_priors={"b": _initial_priors()},
            )

    def test_empty_datasets(self):
        runner = InducedPriorsRunner(smoother_factory=AR1BayesianSmoother)
        with pytest.raises(ValueError):
            runner.run(datasets={}, initial_priors={})


# ──────────────────────────────────────────────────────────────────
# Cross-sectional shrinkage actually shrinks β
# ──────────────────────────────────────────────────────────────────


class TestInducedPriorsShrinkage:
    def test_priors_shrink_toward_cross_sectional_mean(self):
        # Three peer "PE" strategies, all with the same true β = 1.15.
        # Use deliberately bad initial priors (β centered far from 1.15) to make
        # the cross-sectional update visible. After iteration, the priors should
        # all migrate toward the empirical cross-sectional mean.
        datasets = {
            f"strat_{i}": (
                ds.observed_quarterly,
                ds.factor_quarterly,
            )
            for i, ds in enumerate(
                [make_ar1_pe_dataset(rng=seed) for seed in (1, 2, 3)]
            )
        }
        initial = {name: _initial_priors(beta_mean=0.0) for name in datasets}

        runner = InducedPriorsRunner(
            smoother_factory=lambda: AR1BayesianSmoother(lambda_grid_size=20),
            max_iter=5,
            tol=0.005,
        )
        result = runner.run(datasets=datasets, initial_priors=initial)

        # Each strategy's β prior mean should be in the neighbourhood of 1.15
        for name, priors in result.priors.items():
            updated_mean = priors["beta"]["equity_market"].mean
            assert 0.7 < updated_mean < 1.5, (
                f"{name}: β prior mean {updated_mean:.3f} did not migrate from 0 toward truth"
            )

    def test_history_recorded(self):
        datasets = {
            f"s_{i}": (ds.observed_quarterly, ds.factor_quarterly)
            for i, ds in enumerate([make_ar1_pe_dataset(rng=seed) for seed in (10, 11)])
        }
        initial = {name: _initial_priors() for name in datasets}
        runner = InducedPriorsRunner(
            smoother_factory=lambda: AR1BayesianSmoother(lambda_grid_size=20),
            max_iter=3,
            tol=1e-9,  # never converges in 3 iterations
        )
        result = runner.run(datasets=datasets, initial_priors=initial)
        assert len(result.history) == 3
        assert "lambda_mean" in result.history[0]
        assert "beta_equity_market_mean" in result.history[0]

    def test_converges_when_truth_aligned(self):
        # If priors already point at truth, the iteration should converge fast.
        datasets = {
            f"s_{i}": (ds.observed_quarterly, ds.factor_quarterly)
            for i, ds in enumerate([make_ar1_pe_dataset(rng=seed) for seed in (20, 21, 22)])
        }
        initial = {name: _initial_priors(beta_mean=1.15) for name in datasets}
        runner = InducedPriorsRunner(
            smoother_factory=lambda: AR1BayesianSmoother(lambda_grid_size=20),
            max_iter=10,
            tol=0.02,
        )
        result = runner.run(datasets=datasets, initial_priors=initial)
        assert result.converged
        assert result.n_iterations <= 10
