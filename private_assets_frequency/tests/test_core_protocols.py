"""Tests for ``private_assets_frequency.core.protocols``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import (
    AggregationType,
    CorrelationMethod,
    DesmoothedResult,
    DisaggregationMethod,
    Frequency,
    PreprocessingKind,
    Prior,
    SmoothingModel,
    SmoothingModelKind,
    StageValidationResult,
    UncertaintyMode,
    ValidationStatus,
)
from private_assets_frequency.utils.time_series import QUARTER_END_FREQ


# ──────────────────────────────────────────────────────────────────
# Enum coverage
# ──────────────────────────────────────────────────────────────────


class TestFrequencyEnum:
    def test_canonical_values(self):
        assert Frequency.QUARTERLY.value == "quarterly"
        assert Frequency.MONTHLY.value == "monthly"
        assert Frequency.WEEKLY.value == "weekly"
        assert Frequency.DAILY.value == "daily"

    @pytest.mark.parametrize(
        "freq, expected",
        [
            (Frequency.QUARTERLY, 4),
            (Frequency.MONTHLY, 12),
            (Frequency.WEEKLY, 52),
            (Frequency.DAILY, 252),
        ],
    )
    def test_periods_per_year(self, freq, expected):
        assert freq.periods_per_year == expected

    def test_string_construction(self):
        assert Frequency("monthly") is Frequency.MONTHLY

    def test_invalid_string_rejected(self):
        with pytest.raises(ValueError):
            Frequency("annual")


class TestStringEnumValues:
    @pytest.mark.parametrize(
        "kind, expected",
        [
            (SmoothingModelKind.AR1_BAYESIAN, "ar1_bayesian"),
            (SmoothingModelKind.MA_GLM, "ma_glm"),
            (SmoothingModelKind.THRESHOLD_AR1, "threshold_ar1"),
            (SmoothingModelKind.NO_SMOOTHING, "no_smoothing"),
            (SmoothingModelKind.GELTNER_CLASSIC, "geltner_classic"),
            (PreprocessingKind.NONE, "none"),
            (PreprocessingKind.CARRY_MTM_DECOMPOSITION, "carry_mtm_decomposition"),
            (PreprocessingKind.REPORTING_LAG_ADJUSTMENT, "reporting_lag_adjustment"),
            (DisaggregationMethod.CHOW_LIN, "chow_lin"),
            (DisaggregationMethod.FERNANDEZ, "fernandez"),
            (DisaggregationMethod.LITTERMAN, "litterman"),
            (AggregationType.ADDITIVE, "additive"),
            (AggregationType.MULTIPLICATIVE, "multiplicative"),
            (UncertaintyMode.POINT, "point"),
            (UncertaintyMode.FULL, "full"),
            (CorrelationMethod.EMPIRICAL, "empirical"),
            (CorrelationMethod.SHRINKAGE, "shrinkage"),
            (ValidationStatus.PASS, "pass"),
            (ValidationStatus.WARN, "warn"),
            (ValidationStatus.FAIL, "fail"),
        ],
    )
    def test_value_is_canonical_string(self, kind, expected):
        assert kind.value == expected
        assert isinstance(kind, str)
        assert kind == expected  # str equality due to (str, Enum) base


# ──────────────────────────────────────────────────────────────────
# DesmoothedResult
# ──────────────────────────────────────────────────────────────────


class TestDesmoothedResult:
    def test_namedtuple_fields(self):
        idx = pd.date_range("2020-03-31", periods=4, freq=QUARTER_END_FREQ)
        s = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
        result = DesmoothedResult(
            true_returns=s,
            smoothing_params={"lambda": 0.6},
            posterior_summary={"lambda": {"mean": 0.6, "std": 0.05}},
            diagnostics={"warnings": []},
        )
        assert result.true_returns is s
        assert result.smoothing_params["lambda"] == 0.6
        assert result.posterior_summary["lambda"]["mean"] == 0.6
        assert result.diagnostics == {"warnings": []}

    def test_unpacking(self):
        idx = pd.date_range("2020-03-31", periods=4, freq=QUARTER_END_FREQ)
        s = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
        result = DesmoothedResult(s, {}, {}, {})
        ret, sp, ps, diag = result
        assert ret is s
        assert sp == {} and ps == {} and diag == {}


# ──────────────────────────────────────────────────────────────────
# StageValidationResult
# ──────────────────────────────────────────────────────────────────


class TestStageValidationResult:
    def test_passed(self):
        result = StageValidationResult(stage="Stage 1", status=ValidationStatus.PASS)
        assert result.passed is True
        assert result.has_warnings is False
        assert result.is_failure is False

    def test_warn(self):
        result = StageValidationResult(
            stage="Stage 1",
            status=ValidationStatus.WARN,
            checks={"vol_increased": False},
            messages=["desmoothed vol < observed vol"],
        )
        assert result.passed is False
        assert result.has_warnings is True
        assert result.is_failure is False
        assert result.checks == {"vol_increased": False}

    def test_fail(self):
        result = StageValidationResult(
            stage="Stage 3 → 4",
            status=ValidationStatus.FAIL,
            messages=["round-trip error 1e-3 > tol 1e-10"],
        )
        assert result.passed is False
        assert result.has_warnings is True
        assert result.is_failure is True

    def test_immutable(self):
        result = StageValidationResult(stage="x", status=ValidationStatus.PASS)
        with pytest.raises((AttributeError, TypeError)):
            result.stage = "y"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────
# SmoothingModel protocol — runtime structural check
# ──────────────────────────────────────────────────────────────────


class _ToyModel:
    """Minimal class implementing the SmoothingModel attribute surface."""

    def fit(self, observed_returns, factor_returns, priors):
        return DesmoothedResult(observed_returns, {}, {}, {})

    def log_likelihood(self, smoothing_params, observed_returns, factor_returns):
        return 0.0

    def log_prior(self, smoothing_params, prior_config):
        return 0.0

    def desmooth(self, observed_returns, smoothing_params):
        return np.asarray(observed_returns)


class _IncompleteModel:
    """Missing ``desmooth``."""

    def fit(self, observed_returns, factor_returns, priors):
        return DesmoothedResult(observed_returns, {}, {}, {})

    def log_likelihood(self, smoothing_params, observed_returns, factor_returns):
        return 0.0

    def log_prior(self, smoothing_params, prior_config):
        return 0.0


class TestSmoothingModelProtocol:
    def test_runtime_isinstance_passes_for_full_impl(self):
        assert isinstance(_ToyModel(), SmoothingModel)

    def test_runtime_isinstance_fails_for_missing_method(self):
        assert not isinstance(_IncompleteModel(), SmoothingModel)


# ──────────────────────────────────────────────────────────────────
# Prior protocol
# ──────────────────────────────────────────────────────────────────


class _ToyPrior:
    mean = 0.5

    def logpdf(self, x):
        return -0.5 * (x - self.mean) ** 2

    def sample(self, size=1, rng=None):
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        return rng.normal(self.mean, 1.0, size=size)


class TestPriorProtocol:
    def test_runtime_isinstance(self):
        p = _ToyPrior()
        assert isinstance(p, Prior)
