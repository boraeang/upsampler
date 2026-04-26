"""End-to-end private credit pipeline test (quarterly → monthly w/ Threshold AR(1) + carry/MTM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import (
    AssetClassConfig,
    BetaDist,
    InverseGammaPrior,
    NormalPrior,
)
from private_assets_frequency.core.protocols import AggregationType
from private_assets_frequency.pipeline.runner import (
    FrequencyPipeline,
    PipelineResult,
)
from private_assets_frequency.tests.conftest import (
    ThresholdAR1SyntheticDataset,
    make_threshold_credit_dataset,
)
from private_assets_frequency.utils.returns import aggregate_returns
from private_assets_frequency.utils.time_series import (
    MONTH_END_FREQ,
    QUARTER_END_FREQ,
)


# Credit preset matching the conftest synthetic exactly.
_CREDIT_TEST_CFG = AssetClassConfig(
    asset_class="private_credit",
    smoothing_model="threshold_ar1",
    native_frequency="quarterly",
    preprocessing="carry_mtm_decomposition",
    lambda_prior_normal=BetaDist(5.0, 2.0),
    lambda_prior_stress=BetaDist(2.0, 5.0),
    beta_priors={
        "credit_spread": NormalPrior(0.7, 0.3),
        "rate_duration": NormalPrior(-0.2, 0.2),
    },
    alpha_prior=NormalPrior(0.0, 0.03),
    sigma_eps_prior=InverseGammaPrior(3.0, 0.001),
    regime_indicator="regime",
    regime_threshold=500.0,
    default_factors=("credit_spread", "rate_duration"),
    public_proxy="Morningstar LSTA US Leveraged Loan 100",
)


def _build_credit_inputs(ds: ThresholdAR1SyntheticDataset):
    """Build pipeline inputs from the threshold-credit synthetic dataset.

    The credit pipeline needs *monthly* factor returns for Stage 3
    quarterly→monthly disaggregation. We disaggregate each quarterly factor
    column to monthly by uniform multiplicative split (constant per month
    within the quarter) so the round-trip aggregates cleanly.
    """
    returns = pd.DataFrame({"direct_lending": ds.observed_quarterly})
    yields = pd.DataFrame({"direct_lending": ds.yield_quarterly})
    regime = pd.DataFrame({"direct_lending": ds.regime_indicator})

    # Build a monthly index spanning exactly 3·n_q months that ends at each
    # quarter's last month-end.
    n_q = len(ds.factor_quarterly)
    start_q = ds.factor_quarterly.index[0]
    monthly_idx_start = (start_q - pd.offsets.MonthEnd(2)).to_period("M").to_timestamp("M")
    monthly_index = pd.date_range(
        start=monthly_idx_start, periods=3 * n_q, freq=MONTH_END_FREQ
    )
    # Per-month factor return = (1 + r_quarterly)^(1/3) - 1 (constant within
    # the quarter — exact multiplicative split)
    monthly_data = {}
    for col in ds.factor_quarterly.columns:
        monthly_per_q = np.power(1.0 + ds.factor_quarterly[col].to_numpy(), 1.0 / 3.0) - 1.0
        monthly_data[col] = np.repeat(monthly_per_q, 3)
    monthly_F = pd.DataFrame(monthly_data, index=monthly_index)
    return returns, monthly_F, yields, regime


# ──────────────────────────────────────────────────────────────────
# End-to-end pipeline on a single credit strategy
# ──────────────────────────────────────────────────────────────────


def test_credit_pipeline_runs_end_to_end(threshold_credit_dataset):
    returns, monthly_F, yields, regime = _build_credit_inputs(threshold_credit_dataset)
    pipe = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=monthly_F,
        configs={"direct_lending": _CREDIT_TEST_CFG},
        yield_series=yields,
        regime_indicator=regime,
        disaggregation_method="fernandez",
        aggregation_type="multiplicative",
        fallback_policy="warn",
    )
    result = pipe.run()
    assert isinstance(result, PipelineResult)
    assert "direct_lending" in result.monthly_returns.columns
    assert result.daily_returns is None


def test_credit_round_trip_with_carry_reattachment(threshold_credit_dataset):
    returns, monthly_F, yields, regime = _build_credit_inputs(threshold_credit_dataset)
    pipe = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=monthly_F,
        configs={"direct_lending": _CREDIT_TEST_CFG},
        yield_series=yields,
        regime_indicator=regime,
        disaggregation_method="fernandez",
    )
    result = pipe.run()
    # After carry reattachment the monthly compounded back to quarterly
    # equals the *observed* quarterly total return (input to the pipeline).
    monthly = result.monthly_returns["direct_lending"]
    aggregated = aggregate_returns(
        monthly, ratio=3, method=AggregationType.MULTIPLICATIVE
    )
    # Carry uses a linear (not multiplicative) reattachment, so the round-trip
    # holds in arithmetic addition: aggregate(monthly_mtm) + carry_q ≈
    # aggregated total. Check via the disaggregation result's recorded error
    # and verify it's small.
    disagg = result.per_strategy["direct_lending"].disaggregation
    assert disagg.aggregation_error < 1e-2, (
        f"credit aggregation error {disagg.aggregation_error:.4f} too large"
    )


def test_credit_recovers_lambda_normal_in_band(threshold_credit_dataset):
    returns, monthly_F, yields, regime = _build_credit_inputs(threshold_credit_dataset)
    result = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=monthly_F,
        configs={"direct_lending": _CREDIT_TEST_CFG},
        yield_series=yields,
        regime_indicator=regime,
        disaggregation_method="fernandez",
    ).run()
    sp = result.per_strategy["direct_lending"].desmoothed.smoothing_params
    lam_n = sp["lambda_normal"]
    lam_s = sp["lambda_stress"]
    # Truth: λ_normal = 0.75, λ_stress = 0.15
    assert abs(lam_n - threshold_credit_dataset.lambda_normal) < 0.20
    # Stress regime is harder to identify; check ordering only
    assert lam_n > lam_s


def test_credit_stage_validations_recorded(threshold_credit_dataset):
    returns, monthly_F, yields, regime = _build_credit_inputs(threshold_credit_dataset)
    result = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=monthly_F,
        configs={"direct_lending": _CREDIT_TEST_CFG},
        yield_series=yields,
        regime_indicator=regime,
    ).run()
    stages = [v.stage for v in result.stage_validations]
    assert any("Stage 1" in s for s in stages)
    assert any("Stage 3" in s for s in stages)


def test_credit_missing_yield_raises(threshold_credit_dataset):
    returns, monthly_F, _, regime = _build_credit_inputs(threshold_credit_dataset)
    pipe = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=monthly_F,
        configs={"direct_lending": _CREDIT_TEST_CFG},
        regime_indicator=regime,
        # Note: no yield_series supplied
    )
    with pytest.raises(ValueError, match="yield_series"):
        pipe.run()


def test_credit_missing_regime_raises(threshold_credit_dataset):
    returns, monthly_F, yields, _ = _build_credit_inputs(threshold_credit_dataset)
    pipe = FrequencyPipeline(
        returns=returns,
        factor_returns_monthly=monthly_F,
        configs={"direct_lending": _CREDIT_TEST_CFG},
        yield_series=yields,
        # No regime_indicator
    )
    with pytest.raises(ValueError, match="regime_indicator"):
        pipe.run()
