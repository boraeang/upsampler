"""Tests for ``decomposition.factor_model``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.decomposition.factor_model import (
    FactorModel,
    FactorRegressionResult,
)
from private_assets_frequency.tests.conftest import AR1SyntheticDataset
from private_assets_frequency.utils.time_series import (
    MONTH_END_FREQ,
    QUARTER_END_FREQ,
)


# ──────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────


class TestFactorModelConstruction:
    def test_default(self):
        m = FactorModel()
        assert m.method == "ols"
        assert m.use_intercept is True

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            FactorModel(method="bogus")  # type: ignore[arg-type]

    def test_wls_requires_halflife(self):
        with pytest.raises(ValueError):
            FactorModel(method="wls_expdecay")
        with pytest.raises(ValueError):
            FactorModel(method="wls_expdecay", halflife=-1.0)


# ──────────────────────────────────────────────────────────────────
# OLS recovery on synthetic data
# ──────────────────────────────────────────────────────────────────


class TestOLSRecovery:
    def test_recovers_known_betas(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2000-01-31", periods=240, freq=MONTH_END_FREQ)
        f1 = rng.normal(0.0, 0.04, size=240)
        f2 = rng.normal(0.0, 0.03, size=240)
        eps = rng.normal(0.0, 0.005, size=240)
        beta_true = {"f1": 1.2, "f2": -0.4}
        alpha_true = 0.002
        r = beta_true["f1"] * f1 + beta_true["f2"] * f2 + alpha_true + eps
        result = FactorModel().fit(
            pd.Series(r, index=idx, name="r"),
            pd.DataFrame({"f1": f1, "f2": f2}, index=idx),
        )
        assert isinstance(result, FactorRegressionResult)
        assert abs(result.betas["f1"] - 1.2) < 0.05
        assert abs(result.betas["f2"] - -0.4) < 0.05
        assert abs(result.alpha - alpha_true) < 0.005
        assert result.r_squared > 0.95
        assert result.sigma_eps == pytest.approx(0.005, abs=0.001)

    def test_predict_returns_systematic_component(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2010-03-31", periods=80, freq=QUARTER_END_FREQ)
        f1 = rng.normal(0.0, 0.05, size=80)
        r = 1.0 * f1 + 0.001 + rng.normal(0.0, 0.01, size=80)
        m = FactorModel()
        result = m.fit(
            pd.Series(r, index=idx, name="r"),
            pd.DataFrame({"f1": f1}, index=idx),
        )
        prediction = result.predict(pd.DataFrame({"f1": f1}, index=idx))
        # E[prediction] ≈ E[r]; correlation with realised r should be high
        assert prediction.corr(pd.Series(r, index=idx)) > 0.95

    def test_residuals_uncorrelated_with_factors(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2010-01-31", periods=120, freq=MONTH_END_FREQ)
        f = rng.normal(0.0, 0.04, size=120)
        r = 0.8 * f + rng.normal(0.0, 0.01, size=120)
        result = FactorModel().fit(
            pd.Series(r, index=idx),
            pd.DataFrame({"f": f}, index=idx),
        )
        # Residuals should be approximately orthogonal to factor
        corr = float(np.corrcoef(result.residuals, f)[0, 1])
        assert abs(corr) < 1e-10


# ──────────────────────────────────────────────────────────────────
# Weighted OLS / exponential decay
# ──────────────────────────────────────────────────────────────────


class TestExpDecayWeighting:
    def test_recent_obs_weight_larger(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2010-01-31", periods=60, freq=MONTH_END_FREQ)
        f = rng.normal(0.0, 0.03, size=60)
        r = 1.0 * f + rng.normal(0.0, 0.01, size=60)
        m = FactorModel(method="wls_expdecay", halflife=12.0)
        result = m.fit(
            pd.Series(r, index=idx, name="r"),
            pd.DataFrame({"f": f}, index=idx),
        )
        w = result.weights
        # Newest weight = 1, oldest weight = 0.5^(59/12) ≈ 0.032
        assert w[-1] == pytest.approx(1.0)
        assert w[0] == pytest.approx(0.5 ** (59.0 / 12.0), rel=1e-12)
        # Weights are monotonically increasing (older → newer)
        assert (np.diff(w) > 0).all()

    def test_recovers_betas_with_weighting(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2010-01-31", periods=120, freq=MONTH_END_FREQ)
        f = rng.normal(0.0, 0.04, size=120)
        r = 0.7 * f + rng.normal(0.0, 0.01, size=120)
        m = FactorModel(method="wls_expdecay", halflife=24.0)
        result = m.fit(
            pd.Series(r, index=idx),
            pd.DataFrame({"f": f}, index=idx),
        )
        assert abs(result.betas["f"] - 0.7) < 0.1


# ──────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────


class TestValidation:
    def test_rejects_non_series(self):
        with pytest.raises(TypeError):
            FactorModel().fit(np.zeros(50), pd.DataFrame({"f": np.zeros(50)}))

    def test_rejects_empty_factors(self):
        idx = pd.date_range("2010-01-31", periods=50, freq=MONTH_END_FREQ)
        with pytest.raises(TypeError):
            FactorModel().fit(pd.Series(np.zeros(50), index=idx), pd.DataFrame())

    def test_rejects_nan_returns(self):
        idx = pd.date_range("2010-01-31", periods=50, freq=MONTH_END_FREQ)
        s = pd.Series(np.zeros(50), index=idx)
        s.iloc[10] = np.nan
        with pytest.raises(ValueError):
            FactorModel().fit(s, pd.DataFrame({"f": np.zeros(50)}, index=idx))

    def test_rejects_misaligned_factors(self):
        idx_a = pd.date_range("2010-01-31", periods=50, freq=MONTH_END_FREQ)
        idx_b = pd.date_range("2015-01-31", periods=50, freq=MONTH_END_FREQ)
        with pytest.raises(ValueError):
            FactorModel().fit(
                pd.Series(np.zeros(50), index=idx_a),
                pd.DataFrame({"f": np.zeros(50)}, index=idx_b),
            )

    def test_rejects_zero_variance_factor(self):
        idx = pd.date_range("2010-01-31", periods=50, freq=MONTH_END_FREQ)
        with pytest.raises(ValueError, match="zero variance"):
            FactorModel().fit(
                pd.Series(np.linspace(0.0, 0.05, 50), index=idx),
                pd.DataFrame({"f": np.ones(50)}, index=idx),
            )

    def test_rejects_too_few_observations(self):
        idx = pd.date_range("2010-01-31", periods=2, freq=MONTH_END_FREQ)
        with pytest.raises(ValueError, match="observations"):
            FactorModel().fit(
                pd.Series([0.01, 0.02], index=idx),
                pd.DataFrame({"f": [0.01, 0.02]}, index=idx),
            )


# ──────────────────────────────────────────────────────────────────
# Integration with AR(1) synthetic dataset
# ──────────────────────────────────────────────────────────────────


def test_recovers_synthetic_quarterly_betas(ar1_pe_dataset: AR1SyntheticDataset):
    """The factor model fit on the *true* (unsmoothed) quarterly returns should recover β."""
    result = FactorModel().fit(
        ar1_pe_dataset.true_quarterly,
        ar1_pe_dataset.factor_quarterly,
    )
    beta_true = ar1_pe_dataset.beta["equity_market"]
    # Slight bias is expected — the synthetic generator compounds monthly factor
    # & monthly returns to quarterly multiplicatively, but the factor model fits
    # in arithmetic (additive) space at quarterly. Allow ~0.10 of slack.
    assert abs(result.betas["equity_market"] - beta_true) < 0.10
    # R² should be high since the systematic component dominates by construction
    assert result.r_squared > 0.5
