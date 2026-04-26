"""Tests for ``private_assets_frequency.utils.returns``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import AggregationType
from private_assets_frequency.utils.returns import (
    PERIODS_PER_YEAR,
    aggregate_returns,
    aggregation_matrix,
    annualize_return,
    annualize_volatility,
    assert_aggregation_consistent,
    compound_returns,
    from_log_returns,
    realised_volatility,
    to_log_returns,
)
from private_assets_frequency.utils.time_series import MONTH_END_FREQ


# ──────────────────────────────────────────────────────────────────
# log / simple round trips
# ──────────────────────────────────────────────────────────────────


class TestLogReturns:
    @pytest.mark.parametrize("r", [0.0, 0.01, 0.10, -0.05, 0.5])
    def test_round_trip_scalar(self, r):
        arr = np.array([r])
        result = from_log_returns(to_log_returns(arr))
        np.testing.assert_allclose(result, arr, atol=1e-15)

    def test_round_trip_vector(self):
        arr = np.array([-0.10, -0.01, 0.0, 0.01, 0.20, 1.5])
        np.testing.assert_allclose(from_log_returns(to_log_returns(arr)), arr, atol=1e-15)

    def test_pandas_preserved(self):
        idx = pd.date_range("2020-01-01", periods=4)
        s = pd.Series([0.01, 0.02, -0.01, 0.05], index=idx, name="r")
        out = to_log_returns(s)
        assert isinstance(out, pd.Series)
        assert out.name == "r"
        pd.testing.assert_index_equal(out.index, idx)

    def test_dataframe_preserved(self):
        idx = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [-0.01, 0.0, 0.02]}, index=idx)
        out = to_log_returns(df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]

    @pytest.mark.parametrize("bad", [-1.0, -1.5, -10.0])
    def test_rejects_returns_le_neg_one(self, bad):
        with pytest.raises(ValueError, match="≤ -1"):
            to_log_returns(np.array([0.0, bad]))


# ──────────────────────────────────────────────────────────────────
# Compounding
# ──────────────────────────────────────────────────────────────────


class TestCompoundReturns:
    def test_scalar_consistency_simple(self):
        # Two 10% returns compounded → 21%
        arr = np.array([0.1, 0.1])
        assert compound_returns(arr) == pytest.approx(0.21)

    def test_zero_returns_compound_to_zero(self):
        arr = np.zeros(10)
        assert compound_returns(arr) == pytest.approx(0.0)

    def test_compound_matches_explicit_product(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(0.005, 0.04, size=120)
        expected = np.prod(1.0 + arr) - 1.0
        np.testing.assert_allclose(compound_returns(arr), expected, atol=1e-12)

    def test_2d_compounds_per_column(self):
        rng = np.random.default_rng(1)
        arr = rng.normal(0.005, 0.04, size=(60, 3))
        expected = np.prod(1.0 + arr, axis=0) - 1.0
        np.testing.assert_allclose(compound_returns(arr, axis=0), expected, atol=1e-12)

    def test_dataframe_returns_series_indexed_by_columns(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            rng.normal(0.005, 0.04, size=(60, 3)), columns=["a", "b", "c"]
        )
        out = compound_returns(df, axis=0)
        assert isinstance(out, pd.Series)
        assert list(out.index) == ["a", "b", "c"]

    def test_rejects_returns_le_neg_one(self):
        with pytest.raises(ValueError):
            compound_returns(np.array([0.1, -1.0]))


# ──────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────


class TestAggregateReturns:
    def test_multiplicative_round_trip(self):
        rng = np.random.default_rng(0)
        monthly = rng.normal(0.005, 0.04, size=120)
        quarterly = aggregate_returns(monthly, ratio=3)
        # 120 / 3 = 40 quarters
        assert quarterly.shape == (40,)
        # Compound check
        for i in range(40):
            block = monthly[i * 3 : (i + 1) * 3]
            expected = np.prod(1.0 + block) - 1.0
            assert quarterly[i] == pytest.approx(expected, abs=1e-12)

    def test_additive_method(self):
        arr = np.array([0.01, 0.02, 0.03, -0.01, 0.04, 0.05])
        out = aggregate_returns(arr, ratio=3, method="additive")
        np.testing.assert_allclose(out, [0.06, 0.08])

    def test_additive_via_enum(self):
        arr = np.array([0.01, 0.02, 0.03])
        out = aggregate_returns(arr, ratio=3, method=AggregationType.ADDITIVE)
        np.testing.assert_allclose(out, [0.06])

    def test_pandas_series_preserves_index(self):
        idx = pd.date_range("2020-01-31", periods=12, freq=MONTH_END_FREQ)
        s = pd.Series(np.linspace(0.001, 0.012, 12), index=idx, name="r")
        q = aggregate_returns(s, ratio=3)
        assert isinstance(q, pd.Series)
        assert q.name == "r"
        # Each quarterly stamp is the last date of a 3-month block
        expected_index = idx[2::3]
        pd.testing.assert_index_equal(q.index, expected_index)

    def test_pandas_dataframe_preserves_columns(self):
        idx = pd.date_range("2020-01-31", periods=6, freq=MONTH_END_FREQ)
        df = pd.DataFrame({"a": np.full(6, 0.01), "b": np.full(6, 0.02)}, index=idx)
        q = aggregate_returns(df, ratio=3)
        assert isinstance(q, pd.DataFrame)
        assert list(q.columns) == ["a", "b"]
        np.testing.assert_allclose(
            q["a"].to_numpy(), [(1.01) ** 3 - 1.0, (1.01) ** 3 - 1.0]
        )

    def test_length_not_divisible_raises_by_default(self):
        with pytest.raises(ValueError, match="not divisible"):
            aggregate_returns(np.zeros(7), ratio=3)

    def test_trim_leading(self):
        arr = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # length 5, not 3-divisible
        out = aggregate_returns(arr, ratio=3, trim="leading", method="additive")
        np.testing.assert_allclose(out, [0.12])  # 0.03+0.04+0.05

    def test_trim_trailing(self):
        arr = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        out = aggregate_returns(arr, ratio=3, trim="trailing", method="additive")
        np.testing.assert_allclose(out, [0.06])  # 0.01+0.02+0.03

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            aggregate_returns(np.array([0.01]), ratio=0)
        with pytest.raises(ValueError):
            aggregate_returns(np.array([0.01]), ratio=-1)


class TestAggregationMatrix:
    def test_shape_and_block_structure(self):
        C = aggregation_matrix(n_low=5, ratio=3)
        assert C.shape == (5, 15)
        # Row sums equal ratio (each row has `ratio` ones)
        assert (C.sum(axis=1) == 3).all()
        # Column sums equal 1 (each column belongs to exactly one block)
        assert (C.sum(axis=0) == 1).all()

    def test_log_space_aggregation_matches_aggregate_returns(self):
        rng = np.random.default_rng(0)
        monthly = rng.normal(0.005, 0.04, size=24)  # 8 quarters
        # Using C in log-space
        C = aggregation_matrix(n_low=8, ratio=3)
        log_q = C @ np.log1p(monthly)
        from_matrix = np.expm1(log_q)
        from_func = aggregate_returns(monthly, ratio=3)
        np.testing.assert_allclose(from_matrix, from_func, atol=1e-13)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            aggregation_matrix(n_low=0, ratio=3)
        with pytest.raises(ValueError):
            aggregation_matrix(n_low=5, ratio=0)


class TestAssertAggregationConsistent:
    def test_passes_for_correct_pair(self):
        rng = np.random.default_rng(0)
        monthly = rng.normal(0.005, 0.04, size=36)
        quarterly = aggregate_returns(monthly, ratio=3)
        assert_aggregation_consistent(quarterly, monthly, ratio=3)

    def test_raises_when_inconsistent(self):
        rng = np.random.default_rng(0)
        monthly = rng.normal(0.005, 0.04, size=36)
        quarterly = aggregate_returns(monthly, ratio=3)
        # Inject a small perturbation that's larger than tolerance
        perturbed = monthly.copy()
        perturbed[0] += 0.001
        with pytest.raises(AssertionError, match="aggregation inconsistency"):
            assert_aggregation_consistent(quarterly, perturbed, ratio=3)

    def test_machine_precision_round_trip(self):
        # Pivot for the Stage 3 → Stage 4 inter-stage validation
        rng = np.random.default_rng(0)
        monthly = rng.normal(0.005, 0.04, size=120)
        quarterly = aggregate_returns(monthly, ratio=3)
        assert_aggregation_consistent(
            quarterly, monthly, ratio=3, atol=1e-12, rtol=1e-12
        )


# ──────────────────────────────────────────────────────────────────
# Annualisation helpers
# ──────────────────────────────────────────────────────────────────


class TestAnnualizeVolatility:
    def test_monthly_to_annual(self):
        # 4% monthly vol → ~13.86% annual
        assert annualize_volatility(0.04, 12) == pytest.approx(0.04 * np.sqrt(12))

    def test_quarterly_to_annual(self):
        assert annualize_volatility(0.10, 4) == pytest.approx(0.20)

    def test_zero_vol(self):
        assert annualize_volatility(0.0, 12) == 0.0

    @pytest.mark.parametrize("vol,n", [(-0.01, 12), (0.04, 0)])
    def test_invalid(self, vol, n):
        with pytest.raises(ValueError):
            annualize_volatility(vol, n)


class TestAnnualizeReturn:
    def test_compound(self):
        # 1% per month compounded → ~12.68% annual
        assert annualize_return(0.01, 12) == pytest.approx((1.01) ** 12 - 1.0)

    def test_arithmetic(self):
        assert annualize_return(0.01, 12, method="arithmetic") == pytest.approx(0.12)

    def test_negative_return_compounds(self):
        assert annualize_return(-0.01, 12) == pytest.approx((0.99) ** 12 - 1.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            annualize_return(0.01, 12, method="bogus")

    def test_period_return_le_neg_one_rejected(self):
        with pytest.raises(ValueError):
            annualize_return(-1.0, 12)


class TestRealisedVolatility:
    def test_matches_numpy_std(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(0.005, 0.04, size=120)
        assert realised_volatility(arr) == pytest.approx(np.std(arr, ddof=1))

    def test_annualisation_factor(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(0.005, 0.04, size=120)
        s = realised_volatility(arr)
        sa = realised_volatility(arr, annualisation=12)
        assert sa == pytest.approx(s * np.sqrt(12))

    def test_pandas_input(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(0.005, 0.04, size=60)
        s = pd.Series(arr)
        assert realised_volatility(s) == pytest.approx(realised_volatility(arr))

    def test_2d_rejected(self):
        with pytest.raises(ValueError):
            realised_volatility(np.zeros((10, 2)))


class TestPeriodsPerYear:
    def test_constants(self):
        assert PERIODS_PER_YEAR["quarterly"] == 4
        assert PERIODS_PER_YEAR["monthly"] == 12
        assert PERIODS_PER_YEAR["weekly"] == 52
        assert PERIODS_PER_YEAR["daily"] == 252


# ──────────────────────────────────────────────────────────────────
# Integration check: PE-magnitude additive vs multiplicative drift
# ──────────────────────────────────────────────────────────────────


def test_additive_vs_multiplicative_drift_for_pe_magnitude():
    """Spec Critical Implementation Note 3: additive vs multiplicative drift is not trivial.

    For PE-magnitude *quarterly* returns (a few % per quarter, 8 quarters of cover),
    the additive aggregate underestimates the multiplicative aggregate by tens of bps.
    """
    rng = np.random.default_rng(42)
    quarterly = rng.normal(0.04, 0.06, size=8)  # 8 quarters at PE-magnitude vol
    add = aggregate_returns(quarterly, ratio=8, method="additive").item()
    mul = aggregate_returns(quarterly, ratio=8, method="multiplicative").item()
    drift = mul - add  # multiplicative > additive when geometric mean > 0
    # Multiplicative compound differs from sum by enough bps to matter for risk numbers
    assert abs(drift) > 1e-3, f"drift {drift:.6f} unexpectedly small"
