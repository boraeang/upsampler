"""Tests for ``decomposition.residual_analysis``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.decomposition.residual_analysis import (
    FittedDistribution,
    arch_lm,
    cross_strategy_covariance,
    fit_best_distribution,
    fit_normal,
    fit_skewed_t,
    fit_student_t,
    jarque_bera,
    ljung_box,
)


# ──────────────────────────────────────────────────────────────────
# Diagnostic tests
# ──────────────────────────────────────────────────────────────────


class TestLjungBox:
    def test_white_noise_passes(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=500)
        result = ljung_box(x, lags=(5, 10, 20))
        # White noise should not reject at any lag at α=0.05
        for lag in (5, 10, 20):
            assert result.passes_at(lag, alpha=0.01)

    def test_ar1_fails(self):
        rng = np.random.default_rng(0)
        x = np.zeros(500)
        x[0] = rng.normal()
        for t in range(1, 500):
            x[t] = 0.7 * x[t - 1] + rng.normal()
        result = ljung_box(x, lags=(5,))
        assert not result.passes_at(5, alpha=0.05)

    def test_lag_too_large_filtered(self):
        with pytest.raises(ValueError):
            ljung_box(np.zeros(5), lags=(10, 20))


class TestArchLM:
    def test_homoscedastic_passes(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.04, size=500)
        result = arch_lm(x, nlags=5)
        assert not result.has_arch(alpha=0.05)

    def test_heteroscedastic_detected(self):
        rng = np.random.default_rng(0)
        n = 500
        sigma = 0.02 * np.exp(np.sin(np.linspace(0, 8 * np.pi, n)))
        x = rng.normal(0.0, sigma, size=n)
        result = arch_lm(x, nlags=5)
        assert result.has_arch(alpha=0.05)


class TestJarqueBera:
    def test_normal_passes(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=2000)
        result = jarque_bera(x)
        assert result.passes(alpha=0.05)

    def test_t_distribution_fails(self):
        rng = np.random.default_rng(0)
        from scipy import stats
        x = stats.t.rvs(df=3, size=2000, random_state=rng)
        result = jarque_bera(x)
        # Heavy tails of t(3) → JB rejects normality
        assert not result.passes(alpha=0.01)


# ──────────────────────────────────────────────────────────────────
# Distribution fitting
# ──────────────────────────────────────────────────────────────────


class TestFitNormal:
    def test_recovers_parameters(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.5, 0.2, size=10000)
        f = fit_normal(x)
        assert f.name == "normal"
        assert abs(f.params["loc"] - 0.5) < 0.01
        assert abs(f.params["scale"] - 0.2) < 0.01

    def test_pdf_and_sample(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=2000)
        f = fit_normal(x)
        # PDF positive
        assert f.pdf(0.0) > 0.0
        # Sample preserves moments
        samples = f.sample(size=5000, rng=42)
        assert abs(np.mean(samples) - f.params["loc"]) < 0.05
        assert abs(np.std(samples, ddof=1) - f.params["scale"]) < 0.05


class TestFitStudentT:
    def test_recovers_dof(self):
        rng = np.random.default_rng(0)
        from scipy import stats
        x = stats.t.rvs(df=5, size=10000, random_state=rng) * 0.05
        f = fit_student_t(x)
        assert f.name == "student_t"
        assert 3.0 < f.params["df"] < 8.0  # ~5 with sampling noise


class TestFitSkewedT:
    def test_recovers_normal_when_lambda_zero(self):
        rng = np.random.default_rng(0)
        # Heavy-tailed but symmetric → λ should be near zero
        x = rng.standard_t(df=8, size=4000) * 0.05
        f = fit_skewed_t(x)
        assert f.name == "skewed_t"
        assert abs(f.params["skew"]) < 0.2
        assert f.params["df"] > 2.0

    def test_skewed_data_recovers_skew_sign(self):
        rng = np.random.default_rng(0)
        # Lognormal-style: positively skewed in raw scale
        z = rng.normal(0.0, 1.0, size=4000)
        x = np.exp(0.3 * z) - 1.0  # right-skewed
        f = fit_skewed_t(x)
        # Hansen's λ: positive ⇒ right-skewed
        assert f.params["skew"] > 0.05


class TestFitBest:
    def test_picks_normal_for_gaussian(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=2000)
        best = fit_best_distribution(x, criterion="aic")
        assert best.name == "normal"

    def test_picks_student_t_for_heavy_tails(self):
        rng = np.random.default_rng(0)
        from scipy import stats
        x = stats.t.rvs(df=4, size=4000, random_state=rng)
        best = fit_best_distribution(x, criterion="aic")
        assert best.name in ("student_t", "skewed_t")

    def test_invalid_candidate(self):
        with pytest.raises(ValueError):
            fit_best_distribution(np.zeros(100), candidates=("uniform",))  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────
# Cross-strategy covariance
# ──────────────────────────────────────────────────────────────────


class TestCrossStrategyCovariance:
    def test_empirical(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(0.0, 0.04, size=(200, 3)),
            columns=["a", "b", "c"],
        )
        cov = cross_strategy_covariance(df, method="empirical")
        # Should match pandas' .cov()
        pd.testing.assert_frame_equal(cov, df.cov())

    def test_shrinkage_blends(self):
        rng = np.random.default_rng(0)
        n = 500
        idx = np.arange(n)
        # Two strongly correlated series
        u = rng.normal(0.0, 0.04, size=n)
        df = pd.DataFrame(
            {"a": u + rng.normal(0.0, 0.005, size=n), "b": u + rng.normal(0.0, 0.005, size=n)},
            index=idx,
        )
        emp = cross_strategy_covariance(df, method="empirical")
        shr_full = cross_strategy_covariance(df, method="shrinkage", shrinkage_intensity=1.0)
        # Full shrinkage = diagonal: off-diagonals are zero
        assert shr_full.loc["a", "b"] == pytest.approx(0.0)
        # Diagonals match the empirical
        np.testing.assert_allclose(np.diag(shr_full.to_numpy()), np.diag(emp.to_numpy()))

    def test_invalid_intensity(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(0.0, 1.0, size=(100, 2)), columns=["a", "b"])
        with pytest.raises(ValueError):
            cross_strategy_covariance(df, method="shrinkage", shrinkage_intensity=2.0)
