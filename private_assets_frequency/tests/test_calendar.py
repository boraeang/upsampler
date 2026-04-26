"""Tests for ``daily.calendar``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.daily.calendar import (
    block_sizes_for_months,
    business_day_index,
    business_days_per_month,
    irregular_aggregation_matrix,
)


# ──────────────────────────────────────────────────────────────────
# business_day_index
# ──────────────────────────────────────────────────────────────────


class TestBusinessDayIndex:
    def test_bday_calendar_is_mon_fri(self):
        idx = business_day_index("2020-01-01", "2020-01-15", calendar="BDAY")
        weekdays = set(idx.weekday)
        assert weekdays.issubset({0, 1, 2, 3, 4})

    def test_inclusive_bounds(self):
        idx = business_day_index("2020-01-02", "2020-01-08", calendar="BDAY")
        assert idx[0] == pd.Timestamp("2020-01-02")
        assert idx[-1] == pd.Timestamp("2020-01-08")

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="<"):
            business_day_index("2020-01-15", "2020-01-01", calendar="BDAY")

    def test_unsupported_calendar(self):
        with pytest.raises(ValueError):
            business_day_index("2020-01-01", "2020-01-15", calendar="LSE")  # type: ignore[arg-type]

    def test_normalised_to_midnight(self):
        idx = business_day_index("2020-01-02", "2020-01-08", calendar="BDAY")
        for t in idx:
            assert t.hour == 0 and t.minute == 0


# ──────────────────────────────────────────────────────────────────
# business_days_per_month
# ──────────────────────────────────────────────────────────────────


class TestBusinessDaysPerMonth:
    def test_january_2020(self):
        # Jan 2020: 23 weekdays (no holidays in BDAY calendar)
        idx = business_day_index("2020-01-01", "2020-01-31", calendar="BDAY")
        out = business_days_per_month(idx)
        assert len(out) == 1
        assert int(out.iloc[0]) == len(idx) == 23

    def test_full_year(self):
        idx = business_day_index("2020-01-01", "2020-12-31", calendar="BDAY")
        out = business_days_per_month(idx)
        # 12 months
        assert len(out) == 12
        # Total business days = 262 in 2020 (BDAY, no holidays)
        assert int(out.sum()) == len(idx)
        # Each month between 19 and 23 weekdays
        assert out.min() >= 19
        assert out.max() <= 23

    def test_anchor_is_last_business_day(self):
        idx = business_day_index("2020-01-01", "2020-02-28", calendar="BDAY")
        out = business_days_per_month(idx)
        # Last business day of January 2020 is 31 (a Friday)
        # Last business day of February 2020 is 28 (a Friday)
        assert out.index[0] == pd.Timestamp("2020-01-31")
        assert out.index[1] == pd.Timestamp("2020-02-28")

    def test_empty_index_raises(self):
        with pytest.raises(ValueError):
            business_days_per_month(pd.DatetimeIndex([]))

    def test_non_datetime_index_rejected(self):
        with pytest.raises(TypeError):
            business_days_per_month(pd.Index([1, 2, 3]))


# ──────────────────────────────────────────────────────────────────
# block_sizes_for_months
# ──────────────────────────────────────────────────────────────────


class TestBlockSizesForMonths:
    def test_aligns_with_business_days(self):
        idx = business_day_index("2020-01-01", "2020-03-31", calendar="BDAY")
        per_month = business_days_per_month(idx)
        sizes = block_sizes_for_months(idx, per_month.index)
        np.testing.assert_array_equal(sizes, per_month.to_numpy())
        assert int(sizes.sum()) == len(idx)

    def test_extra_daily_outside_months_raises(self):
        idx = business_day_index("2020-01-01", "2020-03-31", calendar="BDAY")
        # Use only the Jan and Feb monthly anchors but supply daily through March
        partial_monthly = pd.DatetimeIndex(
            [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]
        )
        with pytest.raises(ValueError, match="fall inside"):
            block_sizes_for_months(idx, partial_monthly)

    def test_missing_month_raises(self):
        # Daily covers Jan only, monthly index includes Feb
        idx = business_day_index("2020-01-01", "2020-01-31", calendar="BDAY")
        m = pd.DatetimeIndex([pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")])
        with pytest.raises(ValueError):
            block_sizes_for_months(idx, m)

    def test_unsorted_monthly_rejected(self):
        idx = business_day_index("2020-01-01", "2020-02-29", calendar="BDAY")
        m = pd.DatetimeIndex([pd.Timestamp("2020-02-28"), pd.Timestamp("2020-01-31")])
        with pytest.raises(ValueError, match="ascending"):
            block_sizes_for_months(idx, m)


# ──────────────────────────────────────────────────────────────────
# irregular_aggregation_matrix
# ──────────────────────────────────────────────────────────────────


class TestIrregularAggregationMatrix:
    def test_block_structure(self):
        sizes = [3, 5, 4]
        C = irregular_aggregation_matrix(sizes)
        assert C.shape == (3, 12)
        # Each row sums to its block size
        np.testing.assert_array_equal(C.sum(axis=1), sizes)
        # Each column belongs to exactly one block
        np.testing.assert_array_equal(C.sum(axis=0), np.ones(12))
        # Block 0 covers cols 0..2; block 1 cols 3..7; block 2 cols 8..11
        assert (C[0, :3] == 1).all() and (C[0, 3:] == 0).all()
        assert (C[1, 3:8] == 1).all()
        assert (C[2, 8:] == 1).all()

    def test_aggregation_property(self):
        rng = np.random.default_rng(0)
        sizes = [3, 5, 4]
        C = irregular_aggregation_matrix(sizes)
        x = rng.normal(0.0, 0.04, size=12)
        out = C @ x
        # Manual verification
        expected = np.array(
            [x[:3].sum(), x[3:8].sum(), x[8:].sum()]
        )
        np.testing.assert_allclose(out, expected, atol=1e-15)

    def test_invalid_block_sizes(self):
        with pytest.raises(ValueError):
            irregular_aggregation_matrix([3, 0, 4])
        with pytest.raises(ValueError):
            irregular_aggregation_matrix([[1, 2], [3, 4]])
