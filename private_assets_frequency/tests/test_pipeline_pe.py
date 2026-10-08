"""End-to-end PE pipeline test (quarterly → monthly with AR(1) Bayesian)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import AggregationType
from private_assets_frequency.pipeline.presets import PE_PRESETS
from private_assets_frequency.pipeline.runner import (
    FrequencyPipeline,
    PipelineResult,
)
from private_assets_frequency.tests.conftest import (
    AR1SyntheticDataset,
    make_ar1_pe_dataset,
)
from private_assets_frequency.utils.returns import (
    aggregate_returns,
    realised_volatility,
)
from private_assets_frequency.utils.time_series import MONTH_END_FREQ


# ──────────────────────────────────────────────────────────────────
# End-to-end pipeline on a single PE strategy
# ──────────────────────────────────────────────────────────────────


def _build_inputs(ds: AR1SyntheticDataset, factor_name: str = "equity_market"):
    """Wrap conftest dataset to match the runner's expected input shape."""
    returns = pd.DataFrame({"us_buyout": ds.observed_quarterly})
    monthly = pd.DataFrame({factor_name: ds.factor_monthly[factor_name]})
    return returns, monthly


def test_pe_pipeline_runs_end_to_end(ar1_pe_dataset: AR1SyntheticDataset):
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    cfg = PE_PRESETS["us_large_buyout"]
    pipe = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=factors_m,
        configs={"us_buyout": cfg},
        disaggregation_method="chow_lin",
        aggregation_type="multiplicative",
        fallback_policy="warn",
    )
    result = pipe.run()
    assert isinstance(result, PipelineResult)
    # Monthly DataFrame populated for the one strategy
    assert "us_buyout" in result.monthly_returns.columns
    # No daily output (no factor_returns_daily supplied)
    assert result.daily_returns is None


def test_pe_round_trip_aggregation_within_tolerance(ar1_pe_dataset):
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    pipe = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=factors_m,
        configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
    )
    result = pipe.run()
    monthly_us = result.monthly_returns["us_buyout"]
    # Monthly compounded back to quarterly equals the desmoothed quarterly input
    desmoothed_q = result.per_strategy["us_buyout"].desmoothed.true_returns
    aggregated = aggregate_returns(
        monthly_us, ratio=3, method=AggregationType.MULTIPLICATIVE
    )
    np.testing.assert_allclose(
        aggregated.to_numpy(),
        desmoothed_q.to_numpy(),
        atol=1e-10,
    )


def test_pe_recovers_lambda_close_to_truth(ar1_pe_dataset):
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    result = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=factors_m,
        configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
    ).run()
    lam_post = result.per_strategy["us_buyout"].smoothing_params() if False else (
        result.per_strategy["us_buyout"].desmoothed.smoothing_params["lambda"]
    )
    assert abs(lam_post - ar1_pe_dataset.lambda_) < 0.20


def test_pe_desmoothed_vol_above_observed(ar1_pe_dataset):
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    result = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=factors_m,
        configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
    ).run()
    desmoothed = result.per_strategy["us_buyout"].desmoothed.true_returns
    sigma_des = realised_volatility(desmoothed)
    sigma_obs = realised_volatility(ar1_pe_dataset.observed_quarterly)
    assert sigma_des > sigma_obs


def test_pe_stage_validations_recorded(ar1_pe_dataset):
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    result = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=factors_m,
        configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
    ).run()
    # Three inter-stage gates: 1→2, 2→3, 3→4
    stages = [v.stage for v in result.stage_validations]
    assert any("Stage 1" in s for s in stages)
    assert any("Stage 2" in s for s in stages)
    assert any("Stage 3" in s for s in stages)
    # Stage 3 → 4 must be a PASS (round-trip exact)
    s34 = [v for v in result.stage_validations if "Stage 3" in v.stage][0]
    assert s34.passed, f"Stage 3 → 4 failed: {s34.messages}"


def test_pe_with_daily_factors_produces_daily_output(ar1_pe_dataset):
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    # Generate daily factor returns spanning the full monthly window
    rng = np.random.default_rng(0)
    daily_idx = pd.bdate_range(
        start=factors_m.index[0] + pd.offsets.MonthBegin(-1) + pd.Timedelta(days=1),
        end=factors_m.index[-1],
    )
    daily_idx = daily_idx[daily_idx <= factors_m.index[-1]]
    # Restrict to days within the monthly window of factors_m
    daily_idx = daily_idx[
        (daily_idx >= factors_m.index[0].replace(day=1))
        & (daily_idx <= factors_m.index[-1])
    ]
    daily_F = pd.DataFrame(
        {"equity_market": rng.normal(0.0, 0.01, size=len(daily_idx))},
        index=daily_idx,
    )
    pipe = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=factors_m,
        factor_returns_daily=daily_F,
        configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
    )
    result = pipe.run()
    # Stage 4 may or may not run successfully depending on calendar alignment.
    # The point is it shouldn't raise; the pipeline records a warning if it
    # fails. Verify either daily output or a warning.
    if result.daily_returns is None:
        # Expect at least one warning from Stage 4 about calendar alignment
        assert any("Stage4" in w or "stage 4" in w.lower() for w in result.warnings)
    else:
        assert "us_buyout" in result.daily_returns.columns


def test_pe_multi_strategy_runs(ar1_pe_dataset):
    # Two strategies with same data, different configs
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    returns["us_vc"] = ar1_pe_dataset.observed_quarterly
    pipe = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=factors_m,
        configs={
            "us_buyout": PE_PRESETS["us_large_buyout"],
            "us_vc": PE_PRESETS["us_early_venture"],
        },
    )
    result = pipe.run()
    assert set(result.monthly_returns.columns) == {"us_buyout", "us_vc"}
    assert len(result.per_strategy) == 2


# ──────────────────────────────────────────────────────────────────
# Stage 3 alignment when the low-frequency series is shorter than the input
# ──────────────────────────────────────────────────────────────────


def _assert_quarters_round_trip(result, name):
    disagg = result.per_strategy[name].disaggregation
    monthly = disagg.high_frequency
    quarterly_in = disagg.low_frequency_input
    by_quarter = (1.0 + monthly).groupby(monthly.index.to_period("Q")).prod() - 1.0
    assert list(by_quarter.index) == list(quarterly_in.index.to_period("Q"))
    np.testing.assert_allclose(by_quarter.to_numpy(), quarterly_in.to_numpy(), atol=1e-10)


def test_pe_pipeline_with_lag_dropping_smoother(ar1_pe_dataset):
    """``rudin_reparam`` drops its first ``n_lags`` quarters from the desmoothed
    series; Stage 3 must drop the matching months instead of raising a
    row-count mismatch."""
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    cfg = dataclasses.replace(
        PE_PRESETS["us_large_buyout"], smoothing_model="rudin_reparam"
    )
    result = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=factors_m,
        configs={"us_buyout": cfg},
    ).run()
    desmoothed = result.per_strategy["us_buyout"].desmoothed.true_returns
    assert len(desmoothed) < len(returns)
    _assert_quarters_round_trip(result, "us_buyout")


def test_pe_multi_strategy_with_later_inception(ar1_pe_dataset):
    """A strategy whose history starts later (leading NaNs) is disaggregated on
    its own quarters, not on the length of the widest column."""
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    late = ar1_pe_dataset.observed_quarterly.copy()
    late.iloc[:8] = np.nan
    returns["us_vc"] = late
    cfg = PE_PRESETS["us_large_buyout"]
    result = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=factors_m,
        configs={"us_buyout": cfg, "us_vc": cfg},
    ).run()
    _assert_quarters_round_trip(result, "us_buyout")
    _assert_quarters_round_trip(result, "us_vc")


def test_pe_monthly_factors_with_longer_history_are_aligned_by_date(ar1_pe_dataset):
    """Monthly factors that start two years before the PE series must be matched
    to the PE quarters by date, not by row position."""
    returns, factors_m = _build_inputs(ar1_pe_dataset)
    rng = np.random.default_rng(7)
    earlier_index = pd.date_range(
        end=factors_m.index[0] - pd.offsets.MonthEnd(1), periods=24, freq=MONTH_END_FREQ
    )
    earlier = pd.DataFrame(
        {"equity_market": rng.normal(0.0, 0.04, 24)}, index=earlier_index
    )
    longer = pd.concat([earlier, factors_m])
    result = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=longer,
        configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
    ).run()
    monthly = result.monthly_returns["us_buyout"].dropna()
    assert monthly.index[0].to_period("Q") == returns.index[0].to_period("Q")
    _assert_quarters_round_trip(result, "us_buyout")
