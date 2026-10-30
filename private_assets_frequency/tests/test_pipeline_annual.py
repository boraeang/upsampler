"""Annual-native strategies: desmoothing at annual frequency, Stage 3 annual→monthly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import AssetClassConfig
from private_assets_frequency.core.protocols import AggregationType, Frequency
from private_assets_frequency.desmoothing.okunev_white import (
    _infer_periods_per_year as ow_periods_per_year,
)
from private_assets_frequency.desmoothing.rudin_reparam import (
    _infer_periods_per_year as rudin_periods_per_year,
)
from private_assets_frequency.pipeline.presets import PE_PRESETS
from private_assets_frequency.pipeline.runner import (
    FrequencyPipeline,
    _annual_block_positions,
)
from private_assets_frequency.preprocessing.credit_decomposition import (
    decompose_credit_return,
)
from private_assets_frequency.utils.returns import (
    aggregate_returns,
    aggregate_returns_by_blocks,
)
from private_assets_frequency.utils.time_series import (
    MONTH_END_FREQ,
    YEAR_END_FREQ,
    infer_frequency,
    is_period_end_aligned,
)

ANNUAL_CFG = PE_PRESETS["us_large_buyout"].with_overrides(native_frequency="annual")


def _annual_dataset(
    seed: int = 0,
    *,
    n_years: int = 30,
    first_month: str = "1990-01-31",
    lam: float = 0.5,
    pad_months: tuple[int, int] = (0, 0),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Monthly truth → compounded annual → AR(1)-smoothed annual.

    ``first_month`` is the first month of the first year, so the year-end is the
    month before it (January start → December year-end, July → June).
    ``pad_months`` extends the factor file before and after the series.

    Returns ``(observed_annual, monthly_factors, true_monthly)``.
    """
    rng = np.random.default_rng(seed)
    lead, trail = pad_months
    start = pd.Timestamp(first_month) - pd.offsets.MonthEnd(lead)
    months = pd.date_range(start, periods=lead + 12 * n_years + trail, freq=MONTH_END_FREQ)
    f = rng.normal(0.007, 0.045, len(months))
    r_m = 0.001 + 1.1 * f + rng.normal(0.0, 0.02, len(months))
    factors = pd.DataFrame({"equity_market": f}, index=months)
    true_m = pd.Series(r_m, index=months).iloc[lead: lead + 12 * n_years]
    true_a = aggregate_returns_by_blocks(true_m, [12] * n_years)
    s = np.empty(n_years)
    s[0] = true_a.iloc[0]
    for t in range(1, n_years):
        s[t] = lam * s[t - 1] + (1.0 - lam) * true_a.iloc[t]
    observed = pd.DataFrame({"annual_asset": s}, index=true_a.index)
    return observed, factors, true_m


def _run(observed, factors, cfg=ANNUAL_CFG, **kwargs):
    return FrequencyPipeline(
        returns=observed,
        factor_returns_monthly=factors,
        configs={"annual_asset": cfg},
        **kwargs,
    ).run()


# ──────────────────────────────────────────────────────────────────
# Frequency plumbing
# ──────────────────────────────────────────────────────────────────


class TestAnnualFrequency:
    def test_enum_and_periods(self):
        assert Frequency("annual") is Frequency.ANNUAL
        assert Frequency.ANNUAL.periods_per_year == 1

    def test_config_accepts_annual(self):
        assert ANNUAL_CFG.frequency is Frequency.ANNUAL

    def test_reporting_lag_rejected_for_annual(self):
        with pytest.raises(ValueError, match="read as years"):
            AssetClassConfig(
                asset_class="x",
                smoothing_model="no_smoothing",
                native_frequency="annual",
                preprocessing="reporting_lag_adjustment",
                reporting_lag_months=3,
            )

    @pytest.mark.parametrize("month", [12, 6, 3])
    def test_infer_frequency_any_year_end(self, month):
        idx = pd.date_range(f"2000-{month:02d}-01", periods=10, freq="12MS")
        idx = idx + pd.offsets.MonthEnd(0)
        assert infer_frequency(idx) is Frequency.ANNUAL

    def test_infer_frequency_spacing_fallback(self):
        # Irregular reporting dates (not period-ends) fall back on spacing.
        idx = pd.DatetimeIndex(["2001-12-28", "2002-12-31", "2003-12-30", "2004-12-29"])
        assert infer_frequency(idx) is Frequency.ANNUAL

    def test_period_end_aligned_allows_fiscal_years(self):
        june = pd.date_range("2000-06-30", periods=5, freq="12MS") + pd.offsets.MonthEnd(0)
        assert is_period_end_aligned(june, Frequency.ANNUAL)
        assert not is_period_end_aligned(
            pd.DatetimeIndex(["2000-06-15", "2001-06-15"]), Frequency.ANNUAL
        )

    @pytest.mark.parametrize("fn", [rudin_periods_per_year, ow_periods_per_year])
    def test_desmoother_period_inference(self, fn):
        dec = pd.date_range("2000-12-31", periods=6, freq=YEAR_END_FREQ)
        assert fn(dec) == 1
        # Business year-ends ('BA-DEC' / 'BYE-DEC') must not read as daily.
        assert fn(pd.date_range("2000-12-29", periods=6, freq="B" + YEAR_END_FREQ)) == 1

    def test_credit_carry_is_one_year_of_yield(self):
        idx = pd.date_range("2000-12-31", periods=3, freq=YEAR_END_FREQ)
        carry, mtm = decompose_credit_return(
            pd.Series([0.08, 0.05, 0.09], index=idx),
            pd.Series(0.07, index=idx),
            frequency="annual",
        )
        np.testing.assert_allclose(carry.to_numpy(), 0.07)
        np.testing.assert_allclose((carry + mtm).to_numpy(), [0.08, 0.05, 0.09])


class TestAnnualBlocks:
    def test_june_fiscal_year_maps_july_to_june(self):
        native = pd.DatetimeIndex(["2001-06-30", "2002-06-30"])
        months = pd.date_range("2000-06-30", "2002-07-31", freq=MONTH_END_FREQ)
        pos = pd.Series(_annual_block_positions(months, native), index=months)
        assert pos["2000-06-30"] == -1  # belongs to FY2000
        assert pos["2000-07-31"] == 0
        assert pos["2001-06-30"] == 0
        assert pos["2001-07-31"] == 1
        assert pos["2002-06-30"] == 1
        assert pos["2002-07-31"] == -1
        assert (pos == 0).sum() == 12 and (pos == 1).sum() == 12

    def test_gap_year_leaves_its_months_unassigned(self):
        native = pd.DatetimeIndex(["2001-12-31", "2003-12-31"])
        months = pd.date_range("2001-01-31", "2003-12-31", freq=MONTH_END_FREQ)
        pos = _annual_block_positions(months, native)
        assert (pos == -1).sum() == 12

    def test_quarterly_sampled_rolling_annual_is_refused(self):
        native = pd.date_range("2000-03-31", periods=8, freq="3MS") + pd.offsets.MonthEnd(0)
        with pytest.raises(ValueError, match="at least 12 months apart"):
            _annual_block_positions(native, native)


# ──────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────


class TestAnnualPipeline:
    @pytest.mark.parametrize("aggregation", ["multiplicative", "additive"])
    def test_round_trip_to_desmoothed_annual(self, aggregation):
        observed, factors, _ = _annual_dataset()
        res = _run(observed, factors, aggregation_type=aggregation)
        monthly = res.monthly_returns["annual_asset"]
        assert len(monthly) == 12 * len(observed)
        back = aggregate_returns(monthly, ratio=12, method=AggregationType(aggregation))
        desm = res.per_strategy["annual_asset"].desmoothed.true_returns
        np.testing.assert_allclose(back.to_numpy(), desm.to_numpy(), atol=1e-10)
        assert back.index.equals(observed.index)

    def test_recovers_lambda(self):
        observed, factors, _ = _annual_dataset(lam=0.5)
        res = _run(observed, factors)
        lam = res.per_strategy["annual_asset"].desmoothed.smoothing_params["lambda"]
        assert abs(lam - 0.5) < 0.15

    def test_june_fiscal_year_with_longer_factor_file(self):
        # Series closes in June; the factor file starts 5 months early and runs
        # 7 months late, so a positional slice would misalign every year.
        observed, factors, true_m = _annual_dataset(
            first_month="1990-07-31", pad_months=(5, 7)
        )
        assert (observed.index.month == 6).all()
        res = _run(observed, factors)
        monthly = res.monthly_returns["annual_asset"]
        assert monthly.index.equals(true_m.index)
        back = aggregate_returns(monthly, ratio=12)
        np.testing.assert_allclose(
            back.to_numpy(),
            res.per_strategy["annual_asset"].desmoothed.true_returns.to_numpy(),
            atol=1e-10,
        )

    def test_missing_factor_month_is_reported(self):
        observed, factors, _ = _annual_dataset()
        factors = factors.drop(factors.index[40])
        with pytest.raises(ValueError, match="exactly 12 months for every year"):
            _run(observed, factors)

    def test_leading_nan_years_are_dropped(self):
        observed, factors, _ = _annual_dataset()
        observed.iloc[:3] = np.nan
        res = _run(observed, factors)
        monthly = res.per_strategy["annual_asset"].disaggregation.high_frequency
        assert len(monthly) == 12 * (len(observed) - 3)
        assert monthly.index[0] == pd.Timestamp(observed.index[3]) - pd.offsets.MonthEnd(11)

    def test_rho_floored_at_zero(self):
        observed, factors, _ = _annual_dataset()
        disagg = _run(observed, factors).per_strategy["annual_asset"].disaggregation
        assert disagg.rho >= 0.0
        assert disagg.diagnostics["rho_bounds"][0] == 0.0

    def test_monthly_path_tracks_truth(self):
        # Without the ρ floor the 12:1 Chow-Lin fit drifts to ρ≈-0.9 and the
        # monthly vol ratio reaches ~2x on some seeds; with it, it stays near 1.
        vol_ratios, corrs = [], []
        for seed in range(6):
            observed, factors, true_m = _annual_dataset(seed)
            monthly = _run(observed, factors).monthly_returns["annual_asset"]
            vol_ratios.append(monthly.std() / true_m.std())
            corrs.append(np.corrcoef(monthly, true_m)[0, 1])
        assert max(vol_ratios) < 1.35 and min(vol_ratios) > 0.75
        assert np.median(corrs) > 0.85

    def test_quarterly_rho_grid_unchanged(self):
        observed, factors, _ = _annual_dataset()
        pipe = FrequencyPipeline(
            returns=observed,
            factor_returns_monthly=factors,
            configs={"annual_asset": ANNUAL_CFG},
        )
        assert pipe._make_disaggregator(PE_PRESETS["us_large_buyout"]).rho_lower == -0.99

    @pytest.mark.parametrize("model", ["no_smoothing", "geltner_classic", "ma_glm"])
    def test_other_smoothers_run(self, model):
        observed, factors, _ = _annual_dataset()
        overrides: dict = {"smoothing_model": model}
        if model == "ma_glm":
            overrides.update(ma_lags=1, theta_prior="ordered_dirichlet")
        cfg = ANNUAL_CFG.with_overrides(**overrides)
        res = _run(observed, factors, cfg=cfg)
        strat = res.per_strategy["annual_asset"]
        back = aggregate_returns(strat.disaggregation.high_frequency, ratio=12)
        np.testing.assert_allclose(
            back.to_numpy(),
            strat.desmoothed.true_returns.dropna().to_numpy(),
            atol=1e-10,
        )

    def test_uncertainty_bands(self):
        observed, factors, _ = _annual_dataset()
        res = _run(
            observed, factors, uncertainty_mode="full",
            uncertainty_n_samples=10, uncertainty_seed=0,
        )
        bands = res.uncertainty_bands
        assert bands is not None
        assert bands.monthly[50].shape == (12 * len(observed), 1)
        assert (bands.monthly[5] <= bands.monthly[95] + 1e-12).all().all()

    def test_provisional_year_flags_its_twelve_months(self):
        observed, factors, _ = _annual_dataset(first_month="1990-07-31")
        mask = pd.DataFrame(False, index=observed.index, columns=observed.columns)
        mask.iloc[-1] = True
        observed.attrs["provisional_mask"] = mask
        res = _run(observed, factors, allow_provisional=True)
        diag = res.per_strategy["annual_asset"].disaggregation.diagnostics
        flags = diag["is_provisional"]
        assert flags.sum() == 12
        assert flags[flags].index[0] == observed.index[-2] + pd.offsets.MonthEnd(1)
        assert diag["provisional_quarters"] == [str(observed.index[-1].to_period("M"))]

    def test_daily_extension(self):
        observed, factors, _ = _annual_dataset(n_years=15)
        rng = np.random.default_rng(1)
        days = pd.bdate_range(
            factors.index[0].replace(day=1), factors.index[-1]
        )
        daily = pd.DataFrame(
            {"equity_market": rng.normal(0.0, 0.01, len(days))}, index=days
        )
        res = _run(observed, factors, factor_returns_daily=daily)
        assert res.daily_returns is not None
        assert "annual_asset" in res.daily_returns.columns
