"""Tests for ``private_assets_frequency.desmoothing.no_smoothing``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import DesmoothedResult, SmoothingModel
from private_assets_frequency.desmoothing.no_smoothing import NoSmoothing
from private_assets_frequency.utils.time_series import MONTH_END_FREQ


class TestNoSmoothing:
    def test_implements_protocol(self):
        assert isinstance(NoSmoothing(), SmoothingModel)

    def test_fit_returns_observed_unchanged(self):
        idx = pd.date_range("2020-01-31", periods=24, freq=MONTH_END_FREQ)
        s = pd.Series(np.linspace(0.001, 0.024, 24), index=idx, name="r")
        factors = pd.DataFrame({"f": np.zeros(24)}, index=idx)
        result = NoSmoothing().fit(s, factors, priors={})
        assert isinstance(result, DesmoothedResult)
        pd.testing.assert_series_equal(result.true_returns, s)
        # Identity-checking: it's a copy, not the same object
        assert result.true_returns is not s

    def test_smoothing_params_empty(self):
        idx = pd.date_range("2020-01-31", periods=24, freq=MONTH_END_FREQ)
        s = pd.Series(np.zeros(24), index=idx)
        result = NoSmoothing().fit(s, pd.DataFrame(), priors={})
        assert result.smoothing_params == {}
        assert result.posterior_summary == {}
        assert result.diagnostics["method"] == "no_smoothing"
        assert result.diagnostics["n_observations"] == 24

    def test_desmooth_array(self):
        arr = np.array([0.01, 0.02, 0.03])
        out = NoSmoothing().desmooth(arr, np.array([]))
        np.testing.assert_array_equal(out, arr)
        assert out is not arr  # copy, not view

    def test_log_likelihood_zero(self):
        assert NoSmoothing().log_likelihood(np.array([]), np.array([0.01]), np.array([0.0])) == 0.0

    def test_log_prior_zero(self):
        assert NoSmoothing().log_prior(np.array([]), {}) == 0.0

    def test_fit_rejects_non_series(self):
        with pytest.raises(TypeError):
            NoSmoothing().fit(np.array([0.01, 0.02]), pd.DataFrame(), priors={})  # type: ignore[arg-type]
