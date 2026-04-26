"""Tests for ``private_assets_frequency.desmoothing.ar1_bayesian``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import (
    BetaDist,
    InverseGammaPrior,
    NormalPrior,
)
from private_assets_frequency.core.protocols import DesmoothedResult, SmoothingModel
from private_assets_frequency.desmoothing.ar1_bayesian import AR1BayesianSmoother
from private_assets_frequency.tests.conftest import (
    AR1SyntheticDataset,
    make_ar1_pe_dataset,
)
from private_assets_frequency.utils.returns import realised_volatility


# ──────────────────────────────────────────────────────────────────
# Default priors for PE-style synthetic data
# ──────────────────────────────────────────────────────────────────


def _default_priors(true_beta: float = 1.15) -> dict:
    return {
        "lambda": BetaDist(2.0, 2.0),
        "beta": {"equity_market": NormalPrior(true_beta, 0.5)},
        "alpha": NormalPrior(0.0, 0.05),
        "sigma_eps": InverseGammaPrior(3.0, 0.02),
    }


# ──────────────────────────────────────────────────────────────────
# Construction & protocol
# ──────────────────────────────────────────────────────────────────


class TestAR1Construction:
    def test_implements_protocol(self):
        assert isinstance(AR1BayesianSmoother(), SmoothingModel)

    def test_default_grid(self):
        sm = AR1BayesianSmoother()
        assert sm.lambda_grid_size == 50
        assert sm.lambda_grid.shape == (50,)
        assert sm.lambda_grid[0] == 0.01
        assert sm.lambda_grid[-1] == 0.95

    def test_custom_grid(self):
        sm = AR1BayesianSmoother(lambda_grid_size=20, lambda_lower=0.05, lambda_upper=0.90)
        assert sm.lambda_grid.shape == (20,)
        assert sm.lambda_grid[0] == 0.05
        assert sm.lambda_grid[-1] == 0.90

    def test_invalid_grid_bounds(self):
        with pytest.raises(ValueError):
            AR1BayesianSmoother(lambda_lower=0.5, lambda_upper=0.4)
        with pytest.raises(ValueError):
            AR1BayesianSmoother(lambda_lower=0.0)
        with pytest.raises(ValueError):
            AR1BayesianSmoother(lambda_upper=1.0)

    def test_invalid_grid_size(self):
        with pytest.raises(ValueError):
            AR1BayesianSmoother(lambda_grid_size=3)


# ──────────────────────────────────────────────────────────────────
# Desmooth helper (deterministic)
# ──────────────────────────────────────────────────────────────────


class TestAR1Desmooth:
    def test_round_trip_known_lambda(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.01, 0.04, size=80)
        lam = 0.6
        s = np.empty_like(r)
        s[0] = r[0]
        for t in range(1, len(r)):
            s[t] = (1.0 - lam) * r[t] + lam * s[t - 1]
        r_back = AR1BayesianSmoother().desmooth(s, np.array([lam]))
        np.testing.assert_allclose(r_back, r, atol=1e-12)

    def test_lambda_zero_is_identity(self):
        s = np.array([0.01, 0.02, 0.03, 0.04])
        out = AR1BayesianSmoother().desmooth(s, np.array([0.0]))
        np.testing.assert_array_equal(out, s)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            AR1BayesianSmoother().desmooth(np.zeros(5), np.array([1.0]))


# ──────────────────────────────────────────────────────────────────
# Fit on default synthetic dataset
# ──────────────────────────────────────────────────────────────────


class TestAR1FitOnDefaultSynthetic:
    @pytest.fixture
    def fit_result(self, ar1_pe_dataset: AR1SyntheticDataset):
        sm = AR1BayesianSmoother()
        result = sm.fit(
            ar1_pe_dataset.observed_quarterly,
            ar1_pe_dataset.factor_quarterly,
            _default_priors(),
        )
        return sm, result, ar1_pe_dataset

    def test_returns_desmoothed_result(self, fit_result):
        _, result, _ = fit_result
        assert isinstance(result, DesmoothedResult)
        assert "lambda" in result.smoothing_params
        assert "beta" in result.smoothing_params
        assert "alpha" in result.smoothing_params
        assert "sigma_eps" in result.smoothing_params

    def test_lambda_close_to_truth(self, fit_result):
        _, result, ds = fit_result
        lam_post = result.smoothing_params["lambda"]
        assert abs(lam_post - ds.lambda_) < 0.15

    def test_lambda_truth_in_credible_interval(self, fit_result):
        _, result, ds = fit_result
        ci_lo = result.posterior_summary["lambda"]["ci_05"]
        ci_hi = result.posterior_summary["lambda"]["ci_95"]
        assert ci_lo <= ds.lambda_ <= ci_hi

    def test_beta_close_to_truth(self, fit_result):
        _, result, ds = fit_result
        beta_post = result.smoothing_params["beta"]["equity_market"]
        beta_true = ds.beta["equity_market"]
        assert abs(beta_post - beta_true) < 0.25

    def test_desmoothed_vol_above_observed_vol(self, fit_result):
        _, result, ds = fit_result
        sigma_obs = realised_volatility(ds.observed_quarterly)
        sigma_des = realised_volatility(result.true_returns)
        assert sigma_des > sigma_obs

    def test_density_normalises(self, fit_result):
        sm, result, _ = fit_result
        density = result.posterior_summary["lambda"]["density"]
        from scipy import integrate
        assert float(integrate.trapezoid(density, sm.lambda_grid)) == pytest.approx(1.0, abs=1e-6)

    def test_diagnostics_present(self, fit_result):
        _, result, _ = fit_result
        d = result.diagnostics
        assert d["method"] == "ar1_bayesian"
        assert "kl_post_prior" in d
        assert "lambda_ci_width" in d
        assert "flags" in d
        assert "warnings" in d


# ──────────────────────────────────────────────────────────────────
# Coverage test (the central correctness requirement)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.synthetic
class TestAR1Coverage:
    """The 90% credible interval on λ must contain λ_true on >= 90% of seeds."""

    @pytest.mark.parametrize("lambda_true", [0.4, 0.6, 0.75])
    def test_coverage_at_least_90pct(self, lambda_true: float):
        n_seeds = 50
        successes = 0
        widths: list[float] = []
        for seed in range(n_seeds):
            ds = make_ar1_pe_dataset(
                rng=seed,
                n_years=30,
                lambda_=lambda_true,
            )
            sm = AR1BayesianSmoother()
            result = sm.fit(
                ds.observed_quarterly,
                ds.factor_quarterly,
                _default_priors(),
            )
            ci_lo = result.posterior_summary["lambda"]["ci_05"]
            ci_hi = result.posterior_summary["lambda"]["ci_95"]
            widths.append(ci_hi - ci_lo)
            if ci_lo <= lambda_true <= ci_hi:
                successes += 1
        coverage = successes / n_seeds
        avg_width = float(np.mean(widths))
        assert coverage >= 0.9, (
            f"λ={lambda_true}: coverage {coverage:.2f} below 90% target "
            f"({successes}/{n_seeds}); average CI width {avg_width:.3f}"
        )


# ──────────────────────────────────────────────────────────────────
# Diagnostic warnings
# ──────────────────────────────────────────────────────────────────


class TestAR1Diagnostics:
    def test_short_noisy_series_flags_wide_ci(self):
        """A short, weakly-identified series should trip the ci_too_wide flag."""
        rng = np.random.default_rng(0)
        n = 16  # very short — λ cannot be tightly identified
        idx = pd.date_range("2000-03-31", periods=n, freq="Q-DEC")
        # Pure noise observed series — no AR(1) structure to identify
        s = pd.Series(rng.normal(0.0, 0.05, size=n), index=idx)
        f = pd.DataFrame({"f": rng.normal(0.0, 0.05, size=n)}, index=idx)
        priors = {
            "lambda": BetaDist(2.0, 2.0),
            "beta": {"f": NormalPrior(1.0, 0.5)},
            "alpha": NormalPrior(0.0, 0.05),
            "sigma_eps": InverseGammaPrior(3.0, 0.02),
        }
        result = AR1BayesianSmoother().fit(s, f, priors)
        flags = result.diagnostics["flags"]
        # On a short noisy series the CI should be wide → ci_too_wide trips
        assert flags["ci_too_wide"], (
            f"expected wide-CI flag for short noisy series; "
            f"width={result.diagnostics['lambda_ci_width']:.3f}"
        )
        # And the diagnostic warning list contains the corresponding message
        assert any("poorly identified" in w for w in result.diagnostics["warnings"])


# ──────────────────────────────────────────────────────────────────
# Posterior sampling
# ──────────────────────────────────────────────────────────────────


class TestSamplePosterior:
    def test_sample_after_fit(self, ar1_pe_dataset: AR1SyntheticDataset):
        sm = AR1BayesianSmoother()
        sm.fit(
            ar1_pe_dataset.observed_quarterly,
            ar1_pe_dataset.factor_quarterly,
            _default_priors(),
        )
        samples = sm.sample_posterior(n_samples=200, rng=42)
        assert samples["lambda"].shape == (200,)
        assert samples["beta"].shape == (200, 1)
        assert samples["alpha"].shape == (200,)
        assert samples["sigma_eps"].shape == (200,)
        assert (samples["lambda"] >= 0.01).all()
        assert (samples["lambda"] <= 0.95).all()
        assert (samples["sigma_eps"] > 0.0).all()

    def test_sample_before_fit_raises(self):
        sm = AR1BayesianSmoother()
        with pytest.raises(RuntimeError, match="fit"):
            sm.sample_posterior(n_samples=10, rng=42)

    def test_sample_mean_close_to_posterior_mean(
        self, ar1_pe_dataset: AR1SyntheticDataset
    ):
        sm = AR1BayesianSmoother()
        result = sm.fit(
            ar1_pe_dataset.observed_quarterly,
            ar1_pe_dataset.factor_quarterly,
            _default_priors(),
        )
        samples = sm.sample_posterior(n_samples=2000, rng=0)
        post_mean = result.posterior_summary["lambda"]["mean"]
        sample_mean = float(np.mean(samples["lambda"]))
        assert abs(sample_mean - post_mean) < 0.02


# ──────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────


class TestAR1Validation:
    def test_rejects_non_series_observed(self):
        with pytest.raises(TypeError):
            AR1BayesianSmoother().fit(
                np.zeros(20), pd.DataFrame({"f": np.zeros(20)}), _default_priors()
            )

    def test_rejects_empty_factors(self):
        idx = pd.date_range("2000-03-31", periods=40, freq="Q-DEC")
        s = pd.Series(np.zeros(40), index=idx)
        with pytest.raises(TypeError):
            AR1BayesianSmoother().fit(s, pd.DataFrame(), _default_priors())

    def test_rejects_nan_returns(self, ar1_pe_dataset: AR1SyntheticDataset):
        s = ar1_pe_dataset.observed_quarterly.copy()
        s.iloc[10] = np.nan
        with pytest.raises(ValueError):
            AR1BayesianSmoother().fit(
                s, ar1_pe_dataset.factor_quarterly, _default_priors()
            )

    def test_rejects_short_series(self):
        idx = pd.date_range("2000-03-31", periods=8, freq="Q-DEC")
        s = pd.Series(np.zeros(8), index=idx)
        f = pd.DataFrame({"f": np.zeros(8)}, index=idx)
        priors = {
            "lambda": BetaDist(2.0, 2.0),
            "beta": {"f": NormalPrior(1.0, 0.5)},
            "alpha": NormalPrior(0.0, 0.05),
            "sigma_eps": InverseGammaPrior(3.0, 0.02),
        }
        with pytest.raises(ValueError, match=">= 12"):
            AR1BayesianSmoother().fit(s, f, priors)

    def test_missing_prior_raises(self, ar1_pe_dataset):
        priors = _default_priors()
        del priors["alpha"]
        with pytest.raises(ValueError, match="alpha"):
            AR1BayesianSmoother().fit(
                ar1_pe_dataset.observed_quarterly,
                ar1_pe_dataset.factor_quarterly,
                priors,
            )

    def test_missing_factor_prior_raises(self, ar1_pe_dataset):
        priors = _default_priors()
        priors["beta"] = {}  # no entry for 'equity_market'
        with pytest.raises(ValueError, match="missing entries"):
            AR1BayesianSmoother().fit(
                ar1_pe_dataset.observed_quarterly,
                ar1_pe_dataset.factor_quarterly,
                priors,
            )

    def test_invalid_sigma_prior_alpha(self, ar1_pe_dataset):
        priors = _default_priors()
        priors["sigma_eps"] = InverseGammaPrior(0.5, 0.02)
        with pytest.raises(ValueError, match="alpha"):
            AR1BayesianSmoother().fit(
                ar1_pe_dataset.observed_quarterly,
                ar1_pe_dataset.factor_quarterly,
                priors,
            )


# ──────────────────────────────────────────────────────────────────
# log_likelihood / log_prior surface
# ──────────────────────────────────────────────────────────────────


class TestAR1LogPosteriorSurface:
    def test_log_likelihood_finite(self, ar1_pe_dataset):
        sm = AR1BayesianSmoother()
        ll = sm.log_likelihood(
            np.array([0.5]),
            ar1_pe_dataset.observed_quarterly.to_numpy(),
            ar1_pe_dataset.factor_quarterly.to_numpy(),
        )
        assert np.isfinite(ll)

    def test_log_prior_at_lambda(self):
        sm = AR1BayesianSmoother()
        prior_config = {"lambda": BetaDist(2.0, 2.0)}
        # Beta(2,2) has mode at 0.5; logpdf evaluable
        lp = sm.log_prior(np.array([0.5]), prior_config)
        assert np.isfinite(lp)

    def test_log_prior_outside_support(self):
        sm = AR1BayesianSmoother()
        lp = sm.log_prior(np.array([1.5]), {"lambda": BetaDist(2.0, 2.0)})
        assert lp == -np.inf

    def test_log_prior_missing_raises(self):
        sm = AR1BayesianSmoother()
        with pytest.raises(ValueError):
            sm.log_prior(np.array([0.5]), {})
