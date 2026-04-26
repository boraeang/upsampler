"""Tests for ``private_assets_frequency.core.config``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from private_assets_frequency.core.config import (
    AssetClassConfig,
    BetaDist,
    FallbackPolicy,
    InverseGammaPrior,
    NormalPrior,
)
from private_assets_frequency.core.protocols import (
    Frequency,
    PreprocessingKind,
    Prior,
    SmoothingModelKind,
)


# ──────────────────────────────────────────────────────────────────
# FallbackPolicy enum
# ──────────────────────────────────────────────────────────────────


class TestFallbackPolicy:
    def test_values(self):
        assert FallbackPolicy.STRICT == "strict"
        assert FallbackPolicy.WARN == "warn"
        assert FallbackPolicy.AUTO == "auto"

    def test_construction_from_string(self):
        assert FallbackPolicy("warn") is FallbackPolicy.WARN

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            FallbackPolicy("loose")


# ──────────────────────────────────────────────────────────────────
# NormalPrior
# ──────────────────────────────────────────────────────────────────


class TestNormalPrior:
    def test_basic(self):
        p = NormalPrior(0.0, 1.0)
        assert p.mean == 0.0
        assert p.std == 1.0
        assert p.variance == pytest.approx(1.0)

    def test_logpdf_matches_scipy(self):
        p = NormalPrior(0.5, 0.2)
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            p.logpdf(x), stats.norm.logpdf(x, loc=0.5, scale=0.2)
        )

    def test_sample_deterministic_with_seed(self):
        p = NormalPrior(1.0, 0.3)
        a = p.sample(size=5, rng=42)
        b = p.sample(size=5, rng=42)
        np.testing.assert_array_equal(a, b)

    def test_sample_distribution(self):
        p = NormalPrior(2.0, 0.5)
        samples = p.sample(size=100_000, rng=0)
        assert abs(np.mean(samples) - 2.0) < 0.01
        assert abs(np.std(samples) - 0.5) < 0.01

    def test_runtime_protocol_compliance(self):
        assert isinstance(NormalPrior(0.0, 1.0), Prior)

    @pytest.mark.parametrize("std", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_std_rejected(self, std):
        with pytest.raises(ValueError):
            NormalPrior(0.0, std)

    @pytest.mark.parametrize("mean", [math.inf, -math.inf, math.nan])
    def test_invalid_mean_rejected(self, mean):
        with pytest.raises(ValueError):
            NormalPrior(mean, 1.0)

    def test_immutable(self):
        p = NormalPrior(0.0, 1.0)
        with pytest.raises((AttributeError, TypeError)):
            p.mean = 1.0  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────
# BetaDist
# ──────────────────────────────────────────────────────────────────


class TestBetaDist:
    def test_basic(self):
        b = BetaDist(2.0, 2.0)
        assert b.mean == pytest.approx(0.5)
        assert b.variance == pytest.approx(2 * 2 / (4 * 4 * 5))

    def test_pdf_logpdf_match_scipy(self):
        b = BetaDist(3.0, 5.0)
        x = np.linspace(0.05, 0.95, 7)
        np.testing.assert_allclose(b.pdf(x), stats.beta.pdf(x, 3.0, 5.0))
        np.testing.assert_allclose(b.logpdf(x), stats.beta.logpdf(x, 3.0, 5.0))

    def test_sample_in_unit_interval(self):
        b = BetaDist(1.5, 4.0)
        samples = b.sample(size=10_000, rng=7)
        assert (samples >= 0.0).all() and (samples <= 1.0).all()

    def test_sample_mean_consistent(self):
        b = BetaDist(5.0, 2.0)
        samples = b.sample(size=200_000, rng=0)
        assert abs(np.mean(samples) - b.mean) < 0.005

    @pytest.mark.parametrize("a,bb", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (math.nan, 1.0)])
    def test_invalid_params_rejected(self, a, bb):
        with pytest.raises(ValueError):
            BetaDist(a, bb)


# ──────────────────────────────────────────────────────────────────
# InverseGammaPrior
# ──────────────────────────────────────────────────────────────────


class TestInverseGammaPrior:
    def test_logpdf_matches_scipy(self):
        ig = InverseGammaPrior(3.0, 0.02)
        x = np.array([0.005, 0.01, 0.02, 0.05])
        np.testing.assert_allclose(
            ig.logpdf(x), stats.invgamma.logpdf(x, 3.0, scale=0.02)
        )

    def test_sample_positive(self):
        ig = InverseGammaPrior(3.0, 0.02)
        samples = ig.sample(size=1000, rng=42)
        assert (samples > 0.0).all()

    def test_mean_finite_when_alpha_gt_one(self):
        ig = InverseGammaPrior(3.0, 0.02)
        assert ig.mean == pytest.approx(0.02 / 2.0)

    def test_mean_infinite_when_alpha_le_one(self):
        ig = InverseGammaPrior(1.0, 0.02)
        assert math.isinf(ig.mean)

    @pytest.mark.parametrize("a,b", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0)])
    def test_invalid_params_rejected(self, a, b):
        with pytest.raises(ValueError):
            InverseGammaPrior(a, b)


# ──────────────────────────────────────────────────────────────────
# AssetClassConfig
# ──────────────────────────────────────────────────────────────────


def _ar1_config(**overrides) -> AssetClassConfig:
    base = dict(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        lambda_prior=BetaDist(2.0, 2.0),
        beta_priors={"equity_market": NormalPrior(1.15, 0.5)},
        alpha_prior=NormalPrior(0.0, 0.05),
        sigma_eps_prior=InverseGammaPrior(3.0, 0.02),
        default_factors=("SP500", "Russell2000"),
        public_proxy="Russell 2000 (leveraged)",
    )
    base.update(overrides)
    return AssetClassConfig(**base)


class TestAssetClassConfigBasics:
    def test_minimum_ar1_construction(self):
        cfg = _ar1_config()
        assert cfg.smoothing_model == "ar1_bayesian"
        assert cfg.frequency is Frequency.QUARTERLY
        assert cfg.smoothing_kind is SmoothingModelKind.AR1_BAYESIAN
        assert cfg.default_factors == ("SP500", "Russell2000")

    def test_string_or_enum_inputs_normalised(self):
        cfg = _ar1_config(
            smoothing_model=SmoothingModelKind.AR1_BAYESIAN,
            native_frequency=Frequency.QUARTERLY,
        )
        assert cfg.smoothing_model == "ar1_bayesian"
        assert cfg.native_frequency == "quarterly"

    def test_default_factors_coerced_to_tuple(self):
        cfg = _ar1_config(default_factors=["A", "B"])
        assert cfg.default_factors == ("A", "B")
        assert isinstance(cfg.default_factors, tuple)

    def test_immutable(self):
        cfg = _ar1_config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.asset_class = "x"  # type: ignore[misc]

    def test_with_overrides(self):
        cfg = _ar1_config()
        cfg2 = cfg.with_overrides(public_proxy="Russell 1000")
        assert cfg.public_proxy == "Russell 2000 (leveraged)"
        assert cfg2.public_proxy == "Russell 1000"
        assert cfg2.smoothing_model == cfg.smoothing_model

    def test_preprocessing_none_kept(self):
        cfg = _ar1_config(preprocessing=None)
        assert cfg.preprocessing is None

    def test_preprocessing_string_validated(self):
        cfg = _ar1_config(preprocessing="reporting_lag_adjustment", reporting_lag_months=1)
        assert cfg.preprocessing == "reporting_lag_adjustment"


class TestAssetClassConfigValidation:
    def test_unknown_smoothing_model_rejected(self):
        with pytest.raises(ValueError, match="smoothing_model"):
            _ar1_config(smoothing_model="bogus")

    def test_unknown_frequency_rejected(self):
        with pytest.raises(ValueError, match="native_frequency"):
            _ar1_config(native_frequency="annual")

    def test_unknown_preprocessing_rejected(self):
        with pytest.raises(ValueError, match="preprocessing"):
            _ar1_config(preprocessing="bogus")

    def test_ar1_requires_lambda_prior(self):
        with pytest.raises(ValueError, match="lambda_prior"):
            AssetClassConfig(
                asset_class="private_equity",
                smoothing_model="ar1_bayesian",
                native_frequency="quarterly",
                beta_priors={"x": NormalPrior(1.0, 1.0)},
            )

    def test_threshold_requires_full_set(self):
        with pytest.raises(ValueError, match="lambda_prior_normal"):
            AssetClassConfig(
                asset_class="private_credit",
                smoothing_model="threshold_ar1",
                native_frequency="quarterly",
                beta_priors={"credit_spread": NormalPrior(0.7, 0.3)},
                lambda_prior_stress=BetaDist(2.0, 5.0),
                regime_indicator="HY_OAS",
                regime_threshold=500.0,
            )

    def test_threshold_full_set_ok(self):
        cfg = AssetClassConfig(
            asset_class="private_credit",
            smoothing_model="threshold_ar1",
            native_frequency="quarterly",
            beta_priors={"credit_spread": NormalPrior(0.7, 0.3)},
            alpha_prior=NormalPrior(0.0, 0.03),
            lambda_prior_normal=BetaDist(5.0, 2.0),
            lambda_prior_stress=BetaDist(2.0, 5.0),
            regime_indicator="HY_OAS",
            regime_threshold=500.0,
            preprocessing="carry_mtm_decomposition",
        )
        assert cfg.smoothing_kind is SmoothingModelKind.THRESHOLD_AR1
        assert cfg.regime_threshold == 500.0
        assert cfg.preprocessing == PreprocessingKind.CARRY_MTM_DECOMPOSITION.value

    def test_ma_requires_ma_lags(self):
        with pytest.raises(ValueError, match="ma_lags"):
            AssetClassConfig(
                asset_class="hedge_fund",
                smoothing_model="ma_glm",
                native_frequency="monthly",
                beta_priors={"equity_market": NormalPrior(0.4, 0.2)},
            )

    def test_ma_with_lags_ok(self):
        cfg = AssetClassConfig(
            asset_class="hedge_fund",
            smoothing_model="ma_glm",
            native_frequency="monthly",
            beta_priors={"equity_market": NormalPrior(0.4, 0.2)},
            ma_lags=2,
            theta_prior="ordered_dirichlet",
            preprocessing="reporting_lag_adjustment",
            reporting_lag_months=1,
        )
        assert cfg.ma_lags == 2

    def test_no_smoothing_no_priors_required(self):
        cfg = AssetClassConfig(
            asset_class="hedge_fund",
            smoothing_model="no_smoothing",
            native_frequency="monthly",
        )
        assert cfg.smoothing_model == "no_smoothing"

    def test_bayesian_requires_beta_priors(self):
        with pytest.raises(ValueError, match="beta_priors"):
            AssetClassConfig(
                asset_class="private_equity",
                smoothing_model="ar1_bayesian",
                native_frequency="quarterly",
                lambda_prior=BetaDist(2.0, 2.0),
            )

    def test_negative_reporting_lag_rejected(self):
        with pytest.raises(ValueError, match="reporting_lag_months"):
            _ar1_config(reporting_lag_months=-1)

    def test_reporting_lag_preprocessing_requires_positive_lag(self):
        with pytest.raises(ValueError, match="reporting_lag"):
            AssetClassConfig(
                asset_class="hedge_fund",
                smoothing_model="ma_glm",
                native_frequency="monthly",
                beta_priors={"equity_market": NormalPrior(0.4, 0.2)},
                ma_lags=2,
                preprocessing="reporting_lag_adjustment",
                reporting_lag_months=0,
            )
