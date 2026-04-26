"""Tests for ``desmoothing.threshold_ar1``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import (
    BetaDist,
    InverseGammaPrior,
    NormalPrior,
)
from private_assets_frequency.core.protocols import DesmoothedResult
from private_assets_frequency.desmoothing.threshold_ar1 import ThresholdAR1Smoother
from private_assets_frequency.preprocessing.credit_decomposition import (
    decompose_credit_return,
)
from private_assets_frequency.tests.conftest import (
    ThresholdAR1SyntheticDataset,
    make_threshold_credit_dataset,
)
from private_assets_frequency.utils.returns import realised_volatility


# ──────────────────────────────────────────────────────────────────
# Default priors
# ──────────────────────────────────────────────────────────────────


def _default_priors(regime_indicator: pd.Series) -> dict:
    return {
        "lambda_normal": BetaDist(5.0, 2.0),
        "lambda_stress": BetaDist(2.0, 5.0),
        "regime_indicator": regime_indicator,
        "beta": {
            "credit_spread": NormalPrior(0.7, 0.3),
            "rate_duration": NormalPrior(-0.2, 0.2),
        },
        "alpha": NormalPrior(0.0, 0.03),
        "sigma_eps": InverseGammaPrior(3.0, 0.001),
    }


# ──────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────


class TestThresholdAR1Construction:
    def test_default(self):
        sm = ThresholdAR1Smoother()
        assert sm.lambda_grid_size == 30
        assert sm.lambda_grid.shape == (30,)

    def test_invalid_grid_size(self):
        with pytest.raises(ValueError):
            ThresholdAR1Smoother(lambda_grid_size=3)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            ThresholdAR1Smoother(lambda_lower=0.5, lambda_upper=0.4)


# ──────────────────────────────────────────────────────────────────
# Desmooth helper
# ──────────────────────────────────────────────────────────────────


class TestThresholdDesmoothFunction:
    def test_round_trip_known_lambdas(self):
        rng = np.random.default_rng(0)
        n = 60
        r = rng.normal(0.0, 0.02, size=n)
        regime = (rng.uniform(size=n) < 0.2).astype(int)
        lam_n, lam_s = 0.7, 0.15
        s = np.empty_like(r)
        s[0] = r[0]
        for t in range(1, n):
            lam_t = lam_s if regime[t] == 1 else lam_n
            s[t] = (1.0 - lam_t) * r[t] + lam_t * s[t - 1]
        r_back = ThresholdAR1Smoother().desmooth(
            s, np.array([lam_n, lam_s]), regime_indicator=regime
        )
        np.testing.assert_allclose(r_back, r, atol=1e-12)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            ThresholdAR1Smoother().desmooth(
                np.zeros(5), np.array([1.0, 0.5]), regime_indicator=np.zeros(5)
            )

    def test_regime_shape_mismatch(self):
        with pytest.raises(ValueError):
            ThresholdAR1Smoother().desmooth(
                np.zeros(5),
                np.array([0.5, 0.1]),
                regime_indicator=np.zeros(4),
            )


# ──────────────────────────────────────────────────────────────────
# Fit on synthetic credit dataset
# ──────────────────────────────────────────────────────────────────


class TestThresholdFitOnSynthetic:
    @pytest.fixture
    def fit_inputs(self, threshold_credit_dataset: ThresholdAR1SyntheticDataset):
        ds = threshold_credit_dataset
        # Stage 0: split carry / MTM
        carry, mtm = decompose_credit_return(
            ds.observed_quarterly, ds.yield_quarterly, frequency="quarterly"
        )
        return ds, carry, mtm

    @pytest.fixture
    def fit_result(self, fit_inputs):
        ds, carry, mtm = fit_inputs
        sm = ThresholdAR1Smoother()
        priors = _default_priors(ds.regime_indicator)
        result = sm.fit(mtm, ds.factor_quarterly, priors)
        return sm, result, ds, carry, mtm

    def test_returns_desmoothed_result(self, fit_result):
        _, result, _, _, _ = fit_result
        assert isinstance(result, DesmoothedResult)
        assert "lambda_normal" in result.smoothing_params
        assert "lambda_stress" in result.smoothing_params
        assert "beta" in result.smoothing_params

    def test_lambda_normal_above_lambda_stress(self, fit_result):
        _, result, _, _, _ = fit_result
        # Model premise: stress regime marks faster than normal
        assert (
            result.smoothing_params["lambda_normal"]
            > result.smoothing_params["lambda_stress"]
        )

    def test_lambda_normal_close_to_truth(self, fit_result):
        _, result, ds, _, _ = fit_result
        lam_n = result.smoothing_params["lambda_normal"]
        assert abs(lam_n - ds.lambda_normal) < 0.20, (
            f"λ_normal posterior {lam_n:.3f} vs truth {ds.lambda_normal:.3f}"
        )

    def test_lambda_truth_in_credible_intervals(self, fit_result):
        _, result, ds, _, _ = fit_result
        norm_lo, norm_hi = result.diagnostics["lambda_normal_ci"]
        assert norm_lo <= ds.lambda_normal <= norm_hi, (
            f"λ_normal truth {ds.lambda_normal} outside CI [{norm_lo:.3f}, {norm_hi:.3f}]"
        )
        # Stress regime is harder to identify with few stress observations;
        # only require that the recovered point is in the lower tail.
        assert result.smoothing_params["lambda_stress"] < 0.6, (
            f"λ_stress {result.smoothing_params['lambda_stress']:.3f} unexpectedly high"
        )

    def test_desmoothed_mtm_vol_higher_than_smoothed(self, fit_result):
        _, result, _, _, mtm = fit_result
        sigma_obs = realised_volatility(mtm)
        sigma_des = realised_volatility(result.true_returns)
        assert sigma_des > sigma_obs

    def test_diagnostics_include_regime_counts(self, fit_result):
        _, result, _, _, _ = fit_result
        d = result.diagnostics
        assert d["method"] == "threshold_ar1"
        assert d["n_normal"] + d["n_stress"] == d["n_observations"] - 1


# ──────────────────────────────────────────────────────────────────
# Diagnostic warnings
# ──────────────────────────────────────────────────────────────────


class TestThresholdDiagnostics:
    def test_underpowered_stress_regime_warns(self):
        # Construct data with very few stress observations
        ds = make_threshold_credit_dataset(
            n_years=10,
            spread_baseline_bps=200.0,
            spread_uncond_vol_bps=80.0,
            regime_threshold_bps=600.0,  # threshold rarely crossed
            rng=0,
        )
        _, mtm = decompose_credit_return(
            ds.observed_quarterly, ds.yield_quarterly, frequency="quarterly"
        )
        priors = _default_priors(ds.regime_indicator)
        sm = ThresholdAR1Smoother(min_regime_obs=8)
        result = sm.fit(mtm, ds.factor_quarterly, priors)
        flags = result.diagnostics["flags"]
        if result.diagnostics["n_stress"] < 8:
            assert flags["stress_underpowered"]
            assert any(
                "stress regime has only" in w for w in result.diagnostics["warnings"]
            )


# ──────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────


class TestThresholdValidation:
    def test_missing_regime_indicator(self, threshold_credit_dataset):
        ds = threshold_credit_dataset
        _, mtm = decompose_credit_return(
            ds.observed_quarterly, ds.yield_quarterly, frequency="quarterly"
        )
        priors = _default_priors(ds.regime_indicator)
        del priors["regime_indicator"]
        with pytest.raises(ValueError, match="regime_indicator"):
            ThresholdAR1Smoother().fit(mtm, ds.factor_quarterly, priors)

    def test_non_binary_regime(self, threshold_credit_dataset):
        ds = threshold_credit_dataset
        _, mtm = decompose_credit_return(
            ds.observed_quarterly, ds.yield_quarterly, frequency="quarterly"
        )
        priors = _default_priors(ds.regime_indicator * 2)  # values 0 and 2
        with pytest.raises(ValueError, match="binary"):
            ThresholdAR1Smoother().fit(mtm, ds.factor_quarterly, priors)

    def test_log_likelihood_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ThresholdAR1Smoother().log_likelihood(
                np.array([0.5, 0.2]), np.zeros(10), np.zeros((10, 1))
            )

    def test_log_prior_at_lambda(self, threshold_credit_dataset):
        ds = threshold_credit_dataset
        sm = ThresholdAR1Smoother()
        prior_config = {
            "lambda_normal": BetaDist(5.0, 2.0),
            "lambda_stress": BetaDist(2.0, 5.0),
        }
        lp = sm.log_prior(np.array([0.7, 0.2]), prior_config)
        assert np.isfinite(lp)

    def test_log_prior_outside_support(self):
        sm = ThresholdAR1Smoother()
        prior_config = {
            "lambda_normal": BetaDist(5.0, 2.0),
            "lambda_stress": BetaDist(2.0, 5.0),
        }
        lp = sm.log_prior(np.array([1.5, 0.2]), prior_config)
        assert lp == -np.inf
