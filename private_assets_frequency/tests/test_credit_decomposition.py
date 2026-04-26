"""Tests for ``preprocessing.credit_decomposition``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import Frequency
from private_assets_frequency.preprocessing.credit_decomposition import (
    decompose_credit_return,
    reattach_carry,
)
from private_assets_frequency.tests.conftest import (
    ThresholdAR1SyntheticDataset,
    make_threshold_credit_dataset,
)
from private_assets_frequency.utils.time_series import (
    MONTH_END_FREQ,
    QUARTER_END_FREQ,
)


# ──────────────────────────────────────────────────────────────────
# decompose_credit_return
# ──────────────────────────────────────────────────────────────────


class TestDecomposeCreditReturn:
    def test_carry_plus_mtm_equals_total(self):
        idx = pd.date_range("2020-03-31", periods=8, freq=QUARTER_END_FREQ)
        total = pd.Series(np.linspace(0.005, 0.040, 8), index=idx, name="r")
        yld = pd.Series(np.full(8, 0.06), index=idx)
        carry, mtm = decompose_credit_return(total, yld, frequency=Frequency.QUARTERLY)
        np.testing.assert_allclose((carry + mtm).to_numpy(), total.to_numpy(), atol=1e-15)

    def test_quarterly_carry_value(self):
        idx = pd.date_range("2020-03-31", periods=4, freq=QUARTER_END_FREQ)
        total = pd.Series(np.zeros(4), index=idx)
        yld = pd.Series(np.full(4, 0.08), index=idx)
        carry, _ = decompose_credit_return(total, yld, frequency="quarterly")
        # 8% annual yield, quarterly accrual = 2% per quarter
        np.testing.assert_allclose(carry.to_numpy(), 0.02)

    def test_monthly_carry_value(self):
        idx = pd.date_range("2020-01-31", periods=12, freq=MONTH_END_FREQ)
        total = pd.Series(np.zeros(12), index=idx)
        yld = pd.Series(np.full(12, 0.06), index=idx)
        carry, _ = decompose_credit_return(total, yld, frequency=Frequency.MONTHLY)
        np.testing.assert_allclose(carry.to_numpy(), 0.06 / 12.0)

    def test_unknown_frequency_rejected(self):
        idx = pd.date_range("2020-03-31", periods=4, freq=QUARTER_END_FREQ)
        total = pd.Series(np.zeros(4), index=idx)
        yld = pd.Series(np.zeros(4), index=idx)
        with pytest.raises(ValueError):
            decompose_credit_return(total, yld, frequency="annual")

    def test_yield_misalignment_raises(self):
        idx_total = pd.date_range("2020-03-31", periods=4, freq=QUARTER_END_FREQ)
        idx_yld = pd.date_range("2021-03-31", periods=4, freq=QUARTER_END_FREQ)
        total = pd.Series(np.zeros(4), index=idx_total)
        yld = pd.Series(np.full(4, 0.07), index=idx_yld)
        with pytest.raises(ValueError, match="missing"):
            decompose_credit_return(total, yld)

    def test_nan_total_rejected(self):
        idx = pd.date_range("2020-03-31", periods=4, freq=QUARTER_END_FREQ)
        total = pd.Series([0.01, 0.02, np.nan, 0.04], index=idx)
        yld = pd.Series(np.full(4, 0.06), index=idx)
        with pytest.raises(ValueError):
            decompose_credit_return(total, yld)

    def test_round_trip_synthetic(
        self, threshold_credit_dataset: ThresholdAR1SyntheticDataset
    ):
        ds = threshold_credit_dataset
        carry, mtm = decompose_credit_return(
            ds.observed_quarterly,
            ds.yield_quarterly,
            frequency=Frequency.QUARTERLY,
        )
        # The carry should match the conftest's exact construction
        pd.testing.assert_series_equal(
            carry.rename("carry"), ds.carry_quarterly, check_names=False
        )
        # MTM = observed - carry = smoothed_mtm by construction
        pd.testing.assert_series_equal(
            mtm.rename("mtm"), ds.smoothed_mtm_quarterly, check_names=False, atol=1e-12
        )


# ──────────────────────────────────────────────────────────────────
# reattach_carry
# ──────────────────────────────────────────────────────────────────


class TestReattachCarry:
    def test_linear_distribution_quarterly_to_monthly(self):
        # 4 quarters of carry (1% each), spread across 12 months
        q_idx = pd.date_range("2020-03-31", periods=4, freq=QUARTER_END_FREQ)
        m_idx = pd.date_range("2020-01-31", periods=12, freq=MONTH_END_FREQ)
        carry_q = pd.Series(np.full(4, 0.01), index=q_idx)
        mtm_m = pd.Series(np.zeros(12), index=m_idx)  # zero MTM → output is just carry
        out = reattach_carry(mtm_m, carry_q, high_freq_index=m_idx)
        np.testing.assert_allclose(out.to_numpy(), 0.01 / 3.0)
        # Sum within each quarter should equal the quarterly carry
        np.testing.assert_allclose(out.values.reshape(4, 3).sum(axis=1), 0.01)

    def test_preserves_mtm_plus_carry(self):
        q_idx = pd.date_range("2020-03-31", periods=4, freq=QUARTER_END_FREQ)
        m_idx = pd.date_range("2020-01-31", periods=12, freq=MONTH_END_FREQ)
        carry_q = pd.Series([0.02, 0.015, 0.025, 0.018], index=q_idx)
        rng = np.random.default_rng(0)
        mtm_m = pd.Series(rng.normal(0.0, 0.02, size=12), index=m_idx, name="r")
        out = reattach_carry(mtm_m, carry_q, high_freq_index=m_idx)
        # Aggregate carry across each quarter via additive sum (linear distribution
        # gives perfect arithmetic equality even though we don't compound)
        carry_agg = out.values.reshape(4, 3).sum(axis=1) - mtm_m.values.reshape(4, 3).sum(
            axis=1
        )
        np.testing.assert_allclose(carry_agg, carry_q.values, atol=1e-12)

    def test_high_freq_extends_beyond_low_raises(self):
        q_idx = pd.date_range("2020-03-31", periods=2, freq=QUARTER_END_FREQ)
        m_idx = pd.date_range("2020-01-31", periods=12, freq=MONTH_END_FREQ)
        carry_q = pd.Series(np.zeros(2), index=q_idx)
        mtm_m = pd.Series(np.zeros(12), index=m_idx)
        with pytest.raises(ValueError, match="beyond"):
            reattach_carry(mtm_m, carry_q, high_freq_index=m_idx)

    def test_invalid_method(self):
        idx = pd.date_range("2020-03-31", periods=1, freq=QUARTER_END_FREQ)
        with pytest.raises(ValueError):
            reattach_carry(
                pd.Series([0.0], index=idx),
                pd.Series([0.0], index=idx),
                high_freq_index=idx,
                method="bogus",  # type: ignore[arg-type]
            )
