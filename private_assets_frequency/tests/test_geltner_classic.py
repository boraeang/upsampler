"""Tests for ``private_assets_frequency.desmoothing.geltner_classic``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import DesmoothedResult, SmoothingModel
from private_assets_frequency.desmoothing.geltner_classic import (
    GeltnerClassicSmoother,
    _lag1_autocorr,
)
from private_assets_frequency.tests.conftest import (
    AR1SyntheticDataset,
    make_ar1_pe_dataset,
)
from private_assets_frequency.utils.returns import realised_volatility


# ──────────────────────────────────────────────────────────────────
# Lag-1 autocorrelation helper
# ──────────────────────────────────────────────────────────────────


class TestLag1Autocorr:
    def test_white_noise_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=10_000)
        rho = _lag1_autocorr(x)
        assert abs(rho) < 0.05

    def test_high_correlation_for_smooth_ramp(self):
        # Sample lag-1 autocorrelation of a perfect ramp is (n-2)/n = 0.96 for n=50;
        # the empirical centered-product form gives ~0.94, well above zero.
        x = np.linspace(0.0, 1.0, 50)
        rho = _lag1_autocorr(x)
        assert rho > 0.9

    def test_constant_returns_zero(self):
        x = np.full(100, 0.05)
        rho = _lag1_autocorr(x)
        assert rho == 0.0

    def test_recovers_known_ar1_lambda(self):
        # Generate AR(1) with known λ
        rng = np.random.default_rng(0)
        n = 5000
        lam = 0.7
        e = rng.normal(0.0, 1.0, size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = lam * x[t - 1] + (1.0 - lam) * e[t]
        rho = _lag1_autocorr(x)
        # Asymptotic: ρ_1 = λ
        assert abs(rho - lam) < 0.03

    def test_too_short(self):
        with pytest.raises(ValueError):
            _lag1_autocorr(np.array([0.1]))


# ──────────────────────────────────────────────────────────────────
# GeltnerClassicSmoother
# ──────────────────────────────────────────────────────────────────


def _priors_for_pe():
    """Geltner ignores priors but the protocol allows passing them."""
    return {}


class TestGeltnerClassicBasics:
    def test_implements_protocol(self):
        assert isinstance(GeltnerClassicSmoother(), SmoothingModel)

    def test_default_construction(self):
        smoother = GeltnerClassicSmoother()
        assert smoother.lambda_floor == 0.01
        assert smoother.lambda_ceil == 0.95
        assert smoother.lambda_override is None

    def test_invalid_floor_ceil(self):
        with pytest.raises(ValueError):
            GeltnerClassicSmoother(lambda_floor=0.5, lambda_ceil=0.4)

    def test_invalid_override(self):
        with pytest.raises(ValueError):
            GeltnerClassicSmoother(lambda_override=1.5)


class TestGeltnerDesmoothFunction:
    def test_round_trip_known_lambda(self):
        # Apply smoothing then exact desmoothing — should recover original
        rng = np.random.default_rng(0)
        r = rng.normal(0.01, 0.04, size=80)
        lam = 0.5
        # Forward smoothing
        s = np.empty_like(r)
        s[0] = r[0]
        for t in range(1, len(r)):
            s[t] = (1.0 - lam) * r[t] + lam * s[t - 1]
        # Inverse via desmooth
        r_back = GeltnerClassicSmoother().desmooth(s, np.array([lam]))
        # First obs unchanged (s[0] == r[0]); the rest exact
        np.testing.assert_allclose(r_back, r, atol=1e-12)

    def test_lambda_zero_is_identity(self):
        s = np.array([0.01, 0.02, 0.03, 0.04])
        out = GeltnerClassicSmoother().desmooth(s, np.array([0.0]))
        np.testing.assert_array_equal(out, s)

    @pytest.mark.parametrize("lam", [-0.1, 1.0, 1.5])
    def test_invalid_lambda_rejected(self, lam):
        with pytest.raises(ValueError):
            GeltnerClassicSmoother().desmooth(np.zeros(5), np.array([lam]))

    def test_2d_input_rejected(self):
        with pytest.raises(ValueError):
            GeltnerClassicSmoother().desmooth(np.zeros((4, 2)), np.array([0.5]))


class TestGeltnerFit:
    def test_recovers_lambda_from_synthetic(self, ar1_pe_dataset: AR1SyntheticDataset):
        result = GeltnerClassicSmoother().fit(
            ar1_pe_dataset.observed_quarterly,
            ar1_pe_dataset.factor_quarterly,
            priors=_priors_for_pe(),
        )
        assert isinstance(result, DesmoothedResult)
        lam_hat = result.smoothing_params["lambda"]
        # Geltner-classic uses raw lag-1 autocorr — biased downward for AR(1) at
        # T=120 (asymptotically unbiased but finite-sample bias is non-trivial).
        # We only require it lands in a sensible band, not within 0.1 of truth.
        assert abs(lam_hat - ar1_pe_dataset.lambda_) < 0.25

    def test_desmoothed_vol_higher_than_observed(
        self, ar1_pe_dataset: AR1SyntheticDataset
    ):
        result = GeltnerClassicSmoother().fit(
            ar1_pe_dataset.observed_quarterly,
            ar1_pe_dataset.factor_quarterly,
            priors=_priors_for_pe(),
        )
        sigma_obs = realised_volatility(ar1_pe_dataset.observed_quarterly)
        sigma_des = realised_volatility(result.true_returns)
        assert sigma_des > sigma_obs, (
            f"desmoothed vol {sigma_des:.4f} not greater than observed {sigma_obs:.4f}"
        )

    def test_recovers_true_quarterly_volatility(
        self, ar1_pe_dataset: AR1SyntheticDataset
    ):
        result = GeltnerClassicSmoother().fit(
            ar1_pe_dataset.observed_quarterly,
            ar1_pe_dataset.factor_quarterly,
            priors=_priors_for_pe(),
        )
        sigma_des = realised_volatility(result.true_returns)
        sigma_true = realised_volatility(ar1_pe_dataset.true_quarterly)
        # Geltner under-corrects when ρ̂_1 < λ_true; allow a wider band than the
        # Bayesian recovery which is approximately unbiased.
        assert 0.55 * sigma_true <= sigma_des <= 1.45 * sigma_true, (
            f"desmoothed σ {sigma_des:.4f} out of [0.55, 1.45]·{sigma_true:.4f}"
        )

    def test_factor_regression_reasonable(
        self, ar1_pe_dataset: AR1SyntheticDataset
    ):
        result = GeltnerClassicSmoother().fit(
            ar1_pe_dataset.observed_quarterly,
            ar1_pe_dataset.factor_quarterly,
            priors=_priors_for_pe(),
        )
        beta = result.smoothing_params["beta"]
        # True β = 1.15, recovered should be in a reasonable band
        assert "equity_market" in beta
        assert 0.7 < beta["equity_market"] < 1.6

    def test_lambda_override_skips_estimation(self):
        idx = pd.date_range("2020-03-31", periods=40, freq="Q-DEC")
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0.0, 0.05, size=40), index=idx)
        smoother = GeltnerClassicSmoother(lambda_override=0.5)
        result = smoother.fit(s, pd.DataFrame(index=idx), priors={})
        assert result.smoothing_params["lambda"] == 0.5
        assert result.diagnostics["lambda_overridden"] is True

    def test_extreme_lambda_clipped(self):
        # Construct a series with very high autocorrelation
        rng = np.random.default_rng(0)
        n = 200
        x = np.zeros(n)
        x[0] = rng.normal()
        eps = rng.normal(0.0, 0.1, size=n)
        for t in range(1, n):
            x[t] = 0.99 * x[t - 1] + eps[t]
        s = pd.Series(
            x, index=pd.date_range("2000-03-31", periods=n, freq="Q-DEC")
        )
        smoother = GeltnerClassicSmoother()
        result = smoother.fit(s, pd.DataFrame(index=s.index), priors={})
        # Raw should be > 0.95, clipped to ceiling
        assert result.diagnostics["lambda_raw"] > 0.95
        assert result.smoothing_params["lambda"] == pytest.approx(0.95)
        assert result.diagnostics["lambda_clipped"] is True

    def test_rejects_nan_returns(self):
        idx = pd.date_range("2020-03-31", periods=20, freq="Q-DEC")
        s = pd.Series(np.full(20, 0.01), index=idx)
        s.iloc[5] = np.nan
        with pytest.raises(ValueError):
            GeltnerClassicSmoother().fit(s, pd.DataFrame(index=idx), priors={})

    def test_rejects_too_short(self):
        idx = pd.date_range("2020-03-31", periods=3, freq="Q-DEC")
        s = pd.Series(np.full(3, 0.01), index=idx)
        with pytest.raises(ValueError):
            GeltnerClassicSmoother().fit(s, pd.DataFrame(index=idx), priors={})


class TestGeltnerLogLikelihood:
    def test_finite_at_valid_lambda(self):
        rng = np.random.default_rng(0)
        s = rng.normal(0.0, 0.04, size=40)
        ll = GeltnerClassicSmoother().log_likelihood(np.array([0.5]), s, None)
        assert np.isfinite(ll)

    def test_neg_inf_at_invalid_lambda(self):
        s = np.full(40, 0.0)
        ll = GeltnerClassicSmoother().log_likelihood(np.array([1.0]), s, None)
        assert ll == -np.inf

    def test_log_prior_uniform(self):
        sm = GeltnerClassicSmoother()
        assert sm.log_prior(np.array([0.5]), {}) == 0.0
        assert sm.log_prior(np.array([1.5]), {}) == -np.inf
