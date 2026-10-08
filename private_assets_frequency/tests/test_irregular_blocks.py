"""Temporal aggregation with *variable* block sizes.

Monthly→daily disaggregation cannot use a uniform ratio: real months hold
anywhere from ~19 to ~23 business days. These tests pin down the irregular-block
building blocks used by Stage 3 — the aggregation matrix, return compounding,
Chow-Lin itself and the Stage 3→4 round-trip gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import (
    AggregationType,
    DisaggregationMethod,
)
from private_assets_frequency.disaggregation.aggregation import (
    aggregation_matrix,
    irregular_aggregation_matrix,
)
from private_assets_frequency.disaggregation.chow_lin import ChowLinDisaggregator
from private_assets_frequency.utils.returns import (
    aggregate_returns,
    aggregate_returns_by_blocks,
)
from private_assets_frequency.utils.time_series import MONTH_END_FREQ
from private_assets_frequency.validation.stage_validation import (
    validate_stage_3_to_4,
)

# Two years of realistic business-day counts per month (19..23 days).
_SIZES = np.array([22, 20, 21, 22, 21, 20, 23, 21, 21, 22, 20, 22,
                   21, 19, 21, 22, 22, 20, 23, 21, 21, 23, 20, 21])


def _irregular_inputs(seed: int = 1) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_hf = int(_SIZES.sum())
    daily_index = pd.bdate_range("2023-01-02", periods=n_hf)
    indicators = pd.DataFrame(
        {"equity": rng.normal(0.0004, 0.01, n_hf)}, index=daily_index
    )
    monthly_index = pd.date_range("2023-01-31", periods=len(_SIZES), freq=MONTH_END_FREQ)
    low = pd.Series(rng.normal(0.006, 0.035, len(_SIZES)), index=monthly_index)
    return low, indicators


# ──────────────────────────────────────────────────────────────────
# Aggregation matrix
# ──────────────────────────────────────────────────────────────────


def test_irregular_matrix_equals_uniform_matrix_for_equal_blocks():
    np.testing.assert_array_equal(
        irregular_aggregation_matrix([3, 3, 3, 3]), aggregation_matrix(4, 3)
    )


def test_irregular_matrix_assigns_each_column_to_exactly_one_block():
    C = irregular_aggregation_matrix([2, 5, 1])
    assert C.shape == (3, 8)
    np.testing.assert_array_equal(C.sum(axis=1), [2, 5, 1])
    np.testing.assert_array_equal(C.sum(axis=0), np.ones(8))


@pytest.mark.parametrize("bad", [[], [3, 0], [[1, 2], [3, 4]]])
def test_irregular_matrix_rejects_invalid_block_sizes(bad):
    with pytest.raises(ValueError):
        irregular_aggregation_matrix(bad)


# ──────────────────────────────────────────────────────────────────
# Return aggregation
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", list(AggregationType))
def test_by_blocks_reduces_to_uniform_aggregation(method):
    r = np.random.default_rng(0).normal(0.0, 0.02, 12)
    np.testing.assert_allclose(
        aggregate_returns_by_blocks(r, [3, 3, 3, 3], method=method),
        aggregate_returns(r, 3, method=method),
    )


def test_by_blocks_compounds_each_block_separately():
    r = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    out = aggregate_returns_by_blocks(r, [2, 3])
    np.testing.assert_allclose(out, [1.01 * 1.02 - 1.0, 0.99 * 1.03 - 1.0])


def test_by_blocks_additive_sums_each_block():
    r = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    out = aggregate_returns_by_blocks(r, [2, 3], method="additive")
    np.testing.assert_allclose(out, [0.03, 0.02])


def test_by_blocks_labels_pandas_output_at_block_end():
    idx = pd.bdate_range("2024-01-29", periods=5)
    out = aggregate_returns_by_blocks(pd.Series(0.01, index=idx), [2, 3])
    assert list(out.index) == [idx[1], idx[4]]


def test_by_blocks_rejects_length_mismatch():
    with pytest.raises(ValueError, match="block_sizes sum"):
        aggregate_returns_by_blocks(np.zeros(5), [2, 2])


def test_by_blocks_rejects_total_loss_in_multiplicative_mode():
    with pytest.raises(ValueError):
        aggregate_returns_by_blocks(np.array([-1.0, 0.0]), [1, 1])


# ──────────────────────────────────────────────────────────────────
# Chow-Lin with irregular blocks
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", list(DisaggregationMethod))
@pytest.mark.parametrize("aggregation", list(AggregationType))
def test_chow_lin_round_trip_with_irregular_blocks(method, aggregation):
    low, indicators = _irregular_inputs()
    result = ChowLinDisaggregator(method=method, aggregation=aggregation).fit(
        low, indicators, block_sizes=_SIZES
    )
    assert len(result.high_frequency) == int(_SIZES.sum())
    back = aggregate_returns_by_blocks(
        result.high_frequency, _SIZES, method=aggregation
    )
    np.testing.assert_allclose(back.to_numpy(), low.to_numpy(), atol=1e-10)
    assert result.diagnostics["block_sizes"] == _SIZES.tolist()
    assert result.diagnostics["ratio"] is None


def test_chow_lin_requires_exactly_one_of_ratio_or_block_sizes():
    low, indicators = _irregular_inputs()
    disagg = ChowLinDisaggregator()
    with pytest.raises(ValueError, match="exactly one"):
        disagg.fit(low, indicators)
    with pytest.raises(ValueError, match="exactly one"):
        disagg.fit(low, indicators, ratio=21, block_sizes=_SIZES)


def test_chow_lin_rejects_block_sizes_of_wrong_length():
    low, indicators = _irregular_inputs()
    with pytest.raises(ValueError, match="block_sizes"):
        ChowLinDisaggregator().fit(low, indicators, block_sizes=_SIZES[:-1])


def test_chow_lin_rejects_indicator_rows_not_matching_block_total():
    low, indicators = _irregular_inputs()
    with pytest.raises(ValueError, match="rows"):
        ChowLinDisaggregator().fit(low, indicators.iloc[:-1], block_sizes=_SIZES)


# ──────────────────────────────────────────────────────────────────
# Stage 3 → 4 gate
# ──────────────────────────────────────────────────────────────────


def test_stage_3_to_4_gate_accepts_irregular_blocks():
    low, indicators = _irregular_inputs()
    result = ChowLinDisaggregator().fit(low, indicators, block_sizes=_SIZES)
    check = validate_stage_3_to_4(
        result.low_frequency_input, result.high_frequency, block_sizes=_SIZES
    )
    assert check.passed
    assert check.metadata["block_sizes"] == _SIZES.tolist()
    assert check.metadata["ratio"] is None


def test_stage_3_to_4_gate_requires_exactly_one_of_ratio_or_block_sizes():
    low, indicators = _irregular_inputs()
    result = ChowLinDisaggregator().fit(low, indicators, block_sizes=_SIZES)
    with pytest.raises(ValueError, match="exactly one"):
        validate_stage_3_to_4(result.low_frequency_input, result.high_frequency)
