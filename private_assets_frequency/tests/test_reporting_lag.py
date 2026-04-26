"""Tests for ``preprocessing.reporting_lag``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.preprocessing.reporting_lag import adjust_reporting_lag
from private_assets_frequency.utils.time_series import MONTH_END_FREQ


class TestAdjustReportingLag:
    def test_zero_lag_is_identity(self):
        idx = pd.date_range("2020-01-31", periods=6, freq=MONTH_END_FREQ)
        s = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], index=idx, name="r")
        out = adjust_reporting_lag(s, lag_months=0)
        pd.testing.assert_series_equal(out, s)
        assert out is not s  # copy, not the same object

    def test_lag_one_drops_tail(self):
        idx = pd.date_range("2020-01-31", periods=4, freq=MONTH_END_FREQ)
        s = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx, name="r")
        out = adjust_reporting_lag(s, lag_months=1)
        assert len(out) == 3
        np.testing.assert_array_equal(out.to_numpy(), [0.02, 0.03, 0.04])
        # Index is the leading 3 entries of the original
        pd.testing.assert_index_equal(out.index, idx[:3])

    def test_lag_two(self):
        idx = pd.date_range("2020-01-31", periods=6, freq=MONTH_END_FREQ)
        s = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], index=idx)
        out = adjust_reporting_lag(s, lag_months=2)
        np.testing.assert_array_equal(out.to_numpy(), [0.03, 0.04, 0.05, 0.06])

    def test_drop_tail_false_keeps_nan(self):
        idx = pd.date_range("2020-01-31", periods=4, freq=MONTH_END_FREQ)
        s = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
        out = adjust_reporting_lag(s, lag_months=1, drop_tail=False)
        assert len(out) == 4
        assert out.isna().iloc[-1]
        np.testing.assert_array_equal(out.iloc[:-1].to_numpy(), [0.02, 0.03, 0.04])

    def test_dataframe_input(self):
        idx = pd.date_range("2020-01-31", periods=4, freq=MONTH_END_FREQ)
        df = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.04], "b": [-0.01, 0.0, 0.01, 0.02]}, index=idx)
        out = adjust_reporting_lag(df, lag_months=1)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]
        np.testing.assert_array_equal(out["a"].to_numpy(), [0.02, 0.03, 0.04])

    @pytest.mark.parametrize("bad_lag", [-1, 100])
    def test_invalid_lag(self, bad_lag):
        idx = pd.date_range("2020-01-31", periods=4, freq=MONTH_END_FREQ)
        s = pd.Series(np.zeros(4), index=idx)
        with pytest.raises(ValueError):
            adjust_reporting_lag(s, lag_months=bad_lag)

    def test_lag_must_be_int(self):
        idx = pd.date_range("2020-01-31", periods=4, freq=MONTH_END_FREQ)
        s = pd.Series(np.zeros(4), index=idx)
        with pytest.raises(TypeError):
            adjust_reporting_lag(s, lag_months=1.0)  # type: ignore[arg-type]

    def test_non_datetime_index_rejected(self):
        s = pd.Series([0.01, 0.02, 0.03])
        with pytest.raises(TypeError):
            adjust_reporting_lag(s, lag_months=1)

    def test_non_series_dataframe_rejected(self):
        with pytest.raises(TypeError):
            adjust_reporting_lag(np.array([0.01, 0.02]), lag_months=1)
