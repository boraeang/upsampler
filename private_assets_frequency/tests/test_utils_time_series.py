"""Tests for ``private_assets_frequency.utils.time_series``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import Frequency
from private_assets_frequency.utils.returns import compound_returns
from private_assets_frequency.utils.time_series import (
    MONTH_END_FREQ,
    QUARTER_END_FREQ,
    align_factors_to_returns,
    common_index,
    compound_returns_panel,
    four_offset_annual_streams,
    infer_frequency,
    is_period_end_aligned,
    rolling_compound_window,
    rolling_four_quarter_returns,
    trim_to_index,
)


# ──────────────────────────────────────────────────────────────────
# Frequency inference
# ──────────────────────────────────────────────────────────────────


class TestInferFrequency:
    def test_quarterly(self):
        idx = pd.date_range("2020-03-31", periods=20, freq=QUARTER_END_FREQ)
        assert infer_frequency(idx) is Frequency.QUARTERLY

    def test_monthly(self):
        idx = pd.date_range("2020-01-31", periods=24, freq=MONTH_END_FREQ)
        assert infer_frequency(idx) is Frequency.MONTHLY

    def test_weekly(self):
        idx = pd.date_range("2020-01-05", periods=10, freq="W-SUN")
        assert infer_frequency(idx) is Frequency.WEEKLY

    def test_business_daily(self):
        idx = pd.date_range("2020-01-02", periods=20, freq="B")
        assert infer_frequency(idx) is Frequency.DAILY

    def test_irregular_quarterly_via_median(self):
        # Slight perturbation of quarter-ends; median ~91 days
        base = pd.date_range("2020-03-31", periods=10, freq=QUARTER_END_FREQ)
        perturbed = pd.DatetimeIndex(
            [t + pd.Timedelta(days=i % 3) for i, t in enumerate(base)]
        )
        assert infer_frequency(perturbed) is Frequency.QUARTERLY

    def test_too_few_observations(self):
        idx = pd.date_range("2020-01-31", periods=2, freq=MONTH_END_FREQ)
        with pytest.raises(ValueError, match="at least 3"):
            infer_frequency(idx)

    def test_non_datetime_rejected(self):
        with pytest.raises(TypeError):
            infer_frequency(pd.Index([1, 2, 3]))

    def test_unrecognised_spacing(self):
        # 200-day spacing — not a standard frequency
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01") + pd.Timedelta(days=200 * i) for i in range(5)]
        )
        with pytest.raises(ValueError, match="cannot infer"):
            infer_frequency(idx)


class TestIsPeriodEndAligned:
    def test_quarterly_aligned(self):
        idx = pd.date_range("2020-03-31", periods=10, freq=QUARTER_END_FREQ)
        assert is_period_end_aligned(idx, Frequency.QUARTERLY)

    def test_quarterly_misaligned(self):
        idx = pd.date_range("2020-03-15", periods=10, freq=QUARTER_END_FREQ)  # mid-quarter starts
        # This won't actually be misaligned because date_range with QE snaps; build manually:
        idx = pd.DatetimeIndex(["2020-03-31", "2020-06-15", "2020-09-30"])
        assert not is_period_end_aligned(idx, Frequency.QUARTERLY)

    def test_monthly_aligned(self):
        idx = pd.date_range("2020-01-31", periods=10, freq=MONTH_END_FREQ)
        assert is_period_end_aligned(idx, Frequency.MONTHLY)


# ──────────────────────────────────────────────────────────────────
# Alignment
# ──────────────────────────────────────────────────────────────────


class TestAlignFactorsToReturns:
    def test_basic_inner_join(self):
        idx = pd.date_range("2020-01-31", periods=24, freq=MONTH_END_FREQ)
        factors = pd.DataFrame({"f1": np.arange(24)}, index=idx)
        returns = pd.Series(np.arange(24), index=idx, name="r")
        aligned_f, aligned_r = align_factors_to_returns(factors, returns)
        pd.testing.assert_index_equal(aligned_f.index, idx)
        pd.testing.assert_index_equal(aligned_r.index, idx)

    def test_inner_join_intersects(self):
        f_idx = pd.date_range("2019-01-31", periods=36, freq=MONTH_END_FREQ)
        r_idx = pd.date_range("2020-01-31", periods=24, freq=MONTH_END_FREQ)
        factors = pd.DataFrame({"f1": np.arange(36)}, index=f_idx)
        returns = pd.Series(np.arange(24), index=r_idx)
        aligned_f, aligned_r = align_factors_to_returns(factors, returns)
        # Common is the 2020-01-31 to 2021-12-31 window
        assert len(aligned_f) == 24
        assert aligned_f.index.min() == pd.Timestamp("2020-01-31")

    def test_no_overlap_raises(self):
        f_idx = pd.date_range("2010-01-31", periods=24, freq=MONTH_END_FREQ)
        r_idx = pd.date_range("2020-01-31", periods=24, freq=MONTH_END_FREQ)
        factors = pd.DataFrame({"f1": np.arange(24)}, index=f_idx)
        returns = pd.Series(np.arange(24), index=r_idx)
        with pytest.raises(ValueError, match="share no dates"):
            align_factors_to_returns(factors, returns)

    def test_too_few_overlap_raises(self):
        f_idx = pd.date_range("2020-01-31", periods=24, freq=MONTH_END_FREQ)
        r_idx = pd.date_range("2021-08-31", periods=24, freq=MONTH_END_FREQ)  # only 5 overlap
        factors = pd.DataFrame({"f1": np.arange(24)}, index=f_idx)
        returns = pd.Series(np.arange(24), index=r_idx)
        with pytest.raises(ValueError, match="overlapping observations"):
            align_factors_to_returns(factors, returns)

    def test_non_datetime_index_rejected(self):
        factors = pd.DataFrame({"f1": [1, 2, 3]})
        returns = pd.Series([0.1, 0.2, 0.3])
        with pytest.raises(TypeError):
            align_factors_to_returns(factors, returns)

    def test_invalid_how(self):
        idx = pd.date_range("2020-01-31", periods=24, freq=MONTH_END_FREQ)
        factors = pd.DataFrame({"f1": np.arange(24)}, index=idx)
        returns = pd.Series(np.arange(24), index=idx)
        with pytest.raises(ValueError, match="how"):
            align_factors_to_returns(factors, returns, how="outer")


# ──────────────────────────────────────────────────────────────────
# Rolling windows
# ──────────────────────────────────────────────────────────────────


class TestRollingCompoundWindow:
    def test_first_window_minus_one_are_nan(self):
        idx = pd.date_range("2020-03-31", periods=10, freq=QUARTER_END_FREQ)
        s = pd.Series(np.full(10, 0.01), index=idx)
        out = rolling_compound_window(s, window=4)
        assert out.iloc[:3].isna().all()
        assert not out.iloc[3:].isna().any()

    def test_value_matches_explicit_compound(self):
        idx = pd.date_range("2020-03-31", periods=12, freq=QUARTER_END_FREQ)
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0.01, 0.05, size=12), index=idx, name="pe")
        out = rolling_compound_window(s, window=4)
        for t in range(3, 12):
            block = s.iloc[t - 3 : t + 1].to_numpy()
            expected = float(compound_returns(block))
            assert out.iloc[t] == pytest.approx(expected, rel=1e-12)

    def test_invalid_window(self):
        s = pd.Series([0.01, 0.02, 0.03])
        with pytest.raises(ValueError):
            rolling_compound_window(s, window=0)

    def test_non_series_rejected(self):
        with pytest.raises(TypeError):
            rolling_compound_window(np.array([0.01, 0.02]), window=2)

    def test_rejects_le_neg_one(self):
        s = pd.Series([0.01, -1.0, 0.02])
        with pytest.raises(ValueError):
            rolling_compound_window(s, window=2)


class TestRollingFourQuarterReturns:
    def test_alias_of_window_4(self):
        idx = pd.date_range("2020-03-31", periods=12, freq=QUARTER_END_FREQ)
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0.01, 0.05, size=12), index=idx, name="pe")
        out_alias = rolling_four_quarter_returns(s)
        out_explicit = rolling_compound_window(s, window=4)
        pd.testing.assert_series_equal(out_alias, out_explicit, check_names=False)


class TestFourOffsetAnnualStreams:
    def test_lengths(self):
        idx = pd.date_range("1995-03-31", periods=12, freq=QUARTER_END_FREQ)
        s = pd.Series(np.full(12, 0.01), index=idx)
        streams = four_offset_annual_streams(s)
        assert sorted(streams) == [0, 1, 2, 3]
        # T=12: offsets give n_k = (12 - k) // 4 = [3, 2, 2, 2]
        assert len(streams[0]) == 3
        assert len(streams[1]) == 2
        assert len(streams[2]) == 2
        assert len(streams[3]) == 2

    def test_offset_zero_compounds_correctly(self):
        idx = pd.date_range("1995-03-31", periods=12, freq=QUARTER_END_FREQ)
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0.01, 0.05, size=12), index=idx)
        streams = four_offset_annual_streams(s)
        # First entry of offset 0 should be compound of quarters 0..3
        expected = compound_returns(s.iloc[0:4].to_numpy())
        assert streams[0].iloc[0] == pytest.approx(expected, rel=1e-12)

    def test_offset_two_indexed_at_correct_quarter(self):
        idx = pd.date_range("1995-03-31", periods=12, freq=QUARTER_END_FREQ)
        s = pd.Series(np.linspace(0.01, 0.12, 12), index=idx)
        streams = four_offset_annual_streams(s)
        # offset_2 first compound covers quarters 2..5 → last quarter is index 5
        assert streams[2].index[0] == idx[5]

    def test_returns_series_type(self):
        idx = pd.date_range("1995-03-31", periods=12, freq=QUARTER_END_FREQ)
        s = pd.Series(np.full(12, 0.01), index=idx)
        streams = four_offset_annual_streams(s)
        for k, ser in streams.items():
            assert isinstance(ser, pd.Series)
            assert ser.name == f"offset_{k}"

    def test_too_short(self):
        idx = pd.date_range("1995-03-31", periods=3, freq=QUARTER_END_FREQ)
        s = pd.Series(np.full(3, 0.01), index=idx)
        with pytest.raises(ValueError, match=">= 4"):
            four_offset_annual_streams(s)

    def test_rejects_le_neg_one(self):
        idx = pd.date_range("1995-03-31", periods=8, freq=QUARTER_END_FREQ)
        s = pd.Series([0.01, 0.02, -1.0, 0.04, 0.05, 0.06, 0.07, 0.08], index=idx)
        with pytest.raises(ValueError):
            four_offset_annual_streams(s)


# ──────────────────────────────────────────────────────────────────
# Panel utilities
# ──────────────────────────────────────────────────────────────────


class TestCompoundReturnsPanel:
    def test_returns_series_per_column(self):
        idx = pd.date_range("2020-01-31", periods=12, freq=MONTH_END_FREQ)
        df = pd.DataFrame({"a": np.full(12, 0.01), "b": np.full(12, 0.02)}, index=idx)
        out = compound_returns_panel(df)
        assert isinstance(out, pd.Series)
        assert out["a"] == pytest.approx((1.01) ** 12 - 1.0)
        assert out["b"] == pytest.approx((1.02) ** 12 - 1.0)

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            compound_returns_panel(pd.Series([0.01, 0.02]))


class TestCommonIndex:
    def test_two_series(self):
        a = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        b = pd.Series([4, 5], index=pd.date_range("2020-01-02", periods=2))
        idx = common_index(a, b)
        assert len(idx) == 2

    def test_empty_intersection(self):
        a = pd.Series([1], index=[pd.Timestamp("2020-01-01")])
        b = pd.Series([2], index=[pd.Timestamp("2021-01-01")])
        idx = common_index(a, b)
        assert len(idx) == 0

    def test_no_args(self):
        with pytest.raises(ValueError):
            common_index()

    def test_non_datetime_rejected(self):
        a = pd.Series([1, 2], index=[0, 1])
        with pytest.raises(TypeError):
            common_index(a)


class TestTrimToIndex:
    def test_basic(self):
        idx_full = pd.date_range("2020-01-01", periods=5)
        a = pd.Series(np.arange(5), index=idx_full)
        b = pd.Series(np.arange(5) * 2, index=idx_full)
        common = idx_full[1:4]
        a_t, b_t = trim_to_index([a, b], common)
        assert len(a_t) == 3 and len(b_t) == 3
        np.testing.assert_array_equal(a_t.values, [1, 2, 3])

    def test_missing_values_after_reindex(self):
        a = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        target = pd.date_range("2020-01-02", periods=5)  # extends beyond a's coverage
        with pytest.raises(ValueError):
            trim_to_index([a], target)
