r"""
Tests for ``nowcast/integration.py`` and the ``allow_provisional`` pipeline flag.

Covers module-spec test 10: ``extend_with_nowcasts`` →
``FrequencyPipeline(allow_provisional=True)`` runs end to end, provisional rows
are flagged in every output, the **desmoothing parameters are identical** with and
without provisional rows, and ``reconcile`` overwrites them correctly.

"Identical" is asserted exactly — ``==`` on the floats, not ``approx``. The point
of the guarantee is that provisional rows cannot influence an estimate *at all*;
a tolerance would let a small influence through and call it success.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import (
    AssetClassConfig,
    BetaDist,
    InverseGammaPrior,
    NormalPrior,
)
from private_assets_frequency.nowcast import build_information_set
from private_assets_frequency.nowcast.base import make_nowcast_result
from private_assets_frequency.nowcast.integration import (
    EXTENDED_COLUMNS,
    ProvisionalDataWarning,
    extend_with_nowcasts,
    provisional_markers,
    provisional_quarters,
    reconcile,
    reconciliation_log,
    to_pipeline_returns,
)
from private_assets_frequency.nowcast.models import SmoothingRegressionNowcaster
from private_assets_frequency.pipeline.runner import FrequencyPipeline

AS_OF = pd.Timestamp("2026-10-07")
Q2 = pd.Period("2026Q2", freq="Q")
Q3 = pd.Period("2026Q3", freq="Q")


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pe_config() -> AssetClassConfig:
    """An AR(1) Bayesian PE config whose single factor matches the fixtures."""
    return AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        lambda_prior=BetaDist(2, 2),
        beta_priors={"equity_market": NormalPrior(1.15, 0.5)},
        alpha_prior=NormalPrior(0.0, 0.05),
        sigma_eps_prior=InverseGammaPrior(3, 0.02),
        default_factors=("equity_market",),
    )


@pytest.fixture(scope="module")
def nowcast_bundle(motivating_dataset):
    """Published series plus two provisional quarters, as the module would produce."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = build_information_set(
            motivating_dataset.reported,
            motivating_dataset.daily_factors,
            AS_OF,
            vintages=motivating_dataset.vintages,
        )
        model = SmoothingRegressionNowcaster(n_lags=1).fit(info)
        nowcasts = model.predict(
            info,
            info.unpublished_quarters(),
            n_draws=500,
            rng=np.random.default_rng(0),
        )
    return info, nowcasts


@pytest.fixture(scope="module")
def extended(nowcast_bundle) -> pd.DataFrame:
    info, nowcasts = nowcast_bundle
    return extend_with_nowcasts(info.reported, nowcasts)


def _monthly_factors(dataset, extended_frame) -> pd.DataFrame:
    """Monthly factors trimmed to exactly ``3 * n_quarters`` rows."""
    needed = 3 * len(extended_frame)
    return dataset.monthly_factors.iloc[:needed]


def _run(returns, monthly, config, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FrequencyPipeline(
            returns=returns,
            factor_returns_monthly=monthly,
            configs={"pe": config},
            **kwargs,
        ).run()


# ──────────────────────────────────────────────────────────────────
# extend_with_nowcasts
# ──────────────────────────────────────────────────────────────────


class TestExtendWithNowcasts:
    def test_schema_and_ordering(self, extended):
        assert list(extended.columns) == list(EXTENDED_COLUMNS)
        assert isinstance(extended.index, pd.PeriodIndex)
        assert extended.index.freqstr.startswith("Q")
        assert extended.index.is_monotonic_increasing

    def test_provisional_rows_are_appended_and_flagged(self, extended, nowcast_bundle):
        info, nowcasts = nowcast_bundle
        assert list(provisional_quarters(extended)) == [Q2, Q3]
        assert extended.index[-1] == Q3
        published = extended.loc[~extended["is_provisional"]]
        assert published.index[-1] == info.last_published_quarter
        for result in nowcasts:
            row = extended.loc[result.quarter]
            assert row["value"] == result.point
            assert row["nowcast_model"] == result.model
            assert row["as_of"] == result.as_of
            assert row["lower_80"] == result.interval_80[0]
            assert row["upper_80"] == result.interval_80[1]

    def test_published_rows_carry_no_nowcast_metadata(self, extended):
        published = extended.loc[~extended["is_provisional"]]
        assert published["nowcast_model"].isna().all()
        assert published["as_of"].isna().all()
        assert published["lower_80"].isna().all()

    def test_published_values_are_untouched(self, extended, nowcast_bundle):
        info, _ = nowcast_bundle
        published = extended.loc[~extended["is_provisional"], "value"]
        np.testing.assert_array_equal(
            published.to_numpy(), info.reported.to_numpy()
        )

    def test_horizons_ride_along_for_the_reconciliation_log(self, extended):
        """Not part of the six-column contract, but the error pool needs it."""
        horizons = extended.attrs["nowcast_horizons"]
        assert horizons == {Q2: 1, Q3: 2}

    def test_target_true_is_refused(self, nowcast_bundle):
        """The unsmoothed return is a different quantity on a different scale."""
        info, nowcasts = nowcast_bundle
        bad = make_nowcast_result(
            quarter=Q2,
            horizon=1,
            as_of=AS_OF,
            model="smoothing_regression",
            target="true",
            point=0.05,
            draws=np.linspace(0.0, 0.1, 100),
            interval_method="empirical_idiosyncratic",
        )
        with pytest.raises(ValueError, match="target='true'"):
            extend_with_nowcasts(info.reported, [bad])

    def test_already_published_quarter_is_refused(self, nowcast_bundle):
        info, nowcasts = nowcast_bundle
        stale = make_nowcast_result(
            quarter=pd.Period("2026Q1", freq="Q"),
            horizon=1,
            as_of=AS_OF,
            model="m",
            target="reported",
            point=0.01,
            draws=np.linspace(0.0, 0.02, 50),
            interval_method="empirical_recursive",
        )
        with pytest.raises(ValueError, match="at or before the last published"):
            extend_with_nowcasts(info.reported, [stale])

    def test_duplicate_quarters_keep_the_latest_as_of(self, nowcast_bundle):
        info, nowcasts = nowcast_bundle
        earlier = make_nowcast_result(
            quarter=Q2,
            horizon=2,
            as_of=pd.Timestamp("2026-07-01"),
            model="m",
            target="reported",
            point=0.01,
            draws=np.linspace(0.0, 0.02, 50),
            interval_method="empirical_recursive",
        )
        later = make_nowcast_result(
            quarter=Q2,
            horizon=1,
            as_of=pd.Timestamp("2026-10-07"),
            model="m",
            target="reported",
            point=0.09,
            draws=np.linspace(0.08, 0.10, 50),
            interval_method="empirical_recursive",
        )
        with pytest.warns(ProvisionalDataWarning, match="more than one nowcast"):
            frame = extend_with_nowcasts(info.reported, [later, earlier])
        assert frame.loc[Q2, "value"] == 0.09
        assert frame.loc[Q2, "as_of"] == pd.Timestamp("2026-10-07")
        assert frame.attrs["nowcast_horizons"][Q2] == 1

    def test_duplicate_quarters_can_raise_instead(self, nowcast_bundle):
        info, _ = nowcast_bundle
        pair = [
            make_nowcast_result(
                quarter=Q2,
                horizon=h,
                as_of=as_of,
                model="m",
                target="reported",
                point=0.01,
                draws=np.linspace(0.0, 0.02, 50),
                interval_method="empirical_recursive",
            )
            for h, as_of in ((2, pd.Timestamp("2026-07-01")), (1, AS_OF))
        ]
        with pytest.raises(ValueError, match="several nowcasts cover"):
            extend_with_nowcasts(info.reported, pair, on_duplicate="error")

    def test_non_contiguous_provisional_quarters_are_refused(self, nowcast_bundle):
        """A gap would misalign every downstream high-frequency period."""
        info, _ = nowcast_bundle
        gapped = make_nowcast_result(
            quarter=Q3,
            horizon=2,
            as_of=AS_OF,
            model="m",
            target="reported",
            point=0.02,
            draws=np.linspace(0.0, 0.04, 50),
            interval_method="empirical_recursive",
        )
        with pytest.raises(ValueError, match="not a\n?\\s*contiguous run"):
            extend_with_nowcasts(info.reported, [gapped])
        frame = extend_with_nowcasts(
            info.reported, [gapped], require_contiguous=False
        )
        assert Q2 not in frame.index

    def test_no_nowcasts_gives_the_published_series_back(self, nowcast_bundle):
        info, _ = nowcast_bundle
        frame = extend_with_nowcasts(info.reported, [])
        assert not frame["is_provisional"].any()
        assert len(frame) == info.n_published

    def test_input_validation(self, nowcast_bundle):
        info, nowcasts = nowcast_bundle
        with pytest.raises(TypeError, match="reported must be a Series"):
            extend_with_nowcasts(info.reported.to_frame(), nowcasts)
        with pytest.raises(TypeError, match="must contain NowcastResult"):
            extend_with_nowcasts(info.reported, [{"quarter": Q2}])
        with pytest.raises(ValueError, match="reported is empty"):
            extend_with_nowcasts(
                pd.Series(dtype=float, index=pd.PeriodIndex([], freq="Q")), []
            )
        duped = pd.concat([info.reported, info.reported.iloc[[0]]])
        with pytest.raises(ValueError, match="duplicate quarters"):
            extend_with_nowcasts(duped, nowcasts)


# ──────────────────────────────────────────────────────────────────
# to_pipeline_returns
# ──────────────────────────────────────────────────────────────────


class TestToPipelineReturns:
    def test_mask_is_attached_and_aligned(self, extended):
        returns = to_pipeline_returns({"pe": extended})
        mask = returns.attrs["provisional_mask"]
        assert list(returns.columns) == ["pe"]
        assert mask.shape == returns.shape
        assert mask.index.equals(returns.index)
        assert int(mask["pe"].sum()) == 2
        assert bool(mask["pe"].iloc[-1]) and bool(mask["pe"].iloc[-2])
        assert not bool(mask["pe"].iloc[-3])

    def test_index_is_quarter_end_timestamps_by_default(self, extended):
        returns = to_pipeline_returns({"pe": extended})
        assert isinstance(returns.index, pd.DatetimeIndex)
        assert returns.index[-1] == pd.Timestamp("2026-09-30")
        assert (returns.index.day > 27).all()

    def test_period_index_mode(self, extended):
        returns = to_pipeline_returns({"pe": extended}, index="period")
        assert isinstance(returns.index, pd.PeriodIndex)
        assert returns.index[-1] == Q3

    def test_single_frame_needs_a_name(self, extended):
        with pytest.raises(ValueError, match="needs `name`"):
            to_pipeline_returns(extended)
        returns = to_pipeline_returns(extended, name="pe")
        assert list(returns.columns) == ["pe"]

    def test_multiple_strategies(self, extended):
        returns = to_pipeline_returns({"a": extended, "b": extended})
        assert list(returns.columns) == ["a", "b"]
        assert returns.attrs["provisional_mask"].shape == (len(extended), 2)

    def test_misaligned_strategies_are_refused(self, extended):
        with pytest.raises(ValueError, match="needs one shared low-frequency index"):
            to_pipeline_returns({"a": extended, "b": extended.iloc[:-1]})

    def test_non_trailing_provisional_block_is_refused(self, extended):
        broken = extended.copy()
        broken.iloc[5, broken.columns.get_loc("is_provisional")] = True
        with pytest.raises(ValueError, match="contiguous trailing block"):
            to_pipeline_returns({"pe": broken})

    def test_input_validation(self, extended):
        with pytest.raises(ValueError, match="empty mapping"):
            to_pipeline_returns({})
        with pytest.raises(TypeError, match="must be an extended frame"):
            to_pipeline_returns(["not a frame"])
        with pytest.raises(KeyError, match="missing columns"):
            to_pipeline_returns({"pe": extended.drop(columns="lower_80")})
        with pytest.raises(ValueError, match="index must be"):
            to_pipeline_returns({"pe": extended}, index="weekly")


# ──────────────────────────────────────────────────────────────────
# reconcile
# ──────────────────────────────────────────────────────────────────


class TestReconcile:
    def test_overwrites_provisional_rows_with_the_official_print(
        self, extended, motivating_dataset
    ):
        official = motivating_dataset.first_print.loc[[Q2, Q3]]
        result = reconcile(extended, official)
        assert list(provisional_quarters(result)) == []
        assert result.loc[Q2, "value"] == pytest.approx(float(official[Q2]))
        assert result.loc[Q3, "value"] == pytest.approx(float(official[Q3]))
        assert not bool(result.loc[Q2, "is_provisional"])

    def test_keeps_the_nowcast_metadata_so_the_log_is_reconstructable(
        self, extended, motivating_dataset
    ):
        official = motivating_dataset.first_print.loc[[Q2]]
        result = reconcile(extended, official)
        assert result.loc[Q2, "nowcast_model"] == "smoothing_regression"
        assert result.loc[Q2, "as_of"] == AS_OF
        assert result.loc[Q2, "lower_80"] == extended.loc[Q2, "lower_80"]

    def test_log_records_the_error_and_coverage(self, extended, motivating_dataset):
        official = motivating_dataset.first_print.loc[[Q2, Q3]]
        log = reconciliation_log(reconcile(extended, official))
        assert list(log["quarter"]) == [Q2, Q3]
        assert list(log["horizon"]) == [1, 2]
        for _, row in log.iterrows():
            assert row["nowcast"] == extended.loc[row["quarter"], "value"]
            assert row["actual"] == pytest.approx(float(official[row["quarter"]]))
            assert row["error"] == pytest.approx(row["actual"] - row["nowcast"])
            assert row["covered_80"] == (
                row["lower_80"] <= row["actual"] <= row["upper_80"]
            )

    def test_log_accumulates_across_calls(self, extended, motivating_dataset):
        """Reconciling quarter by quarter builds the realised error history."""
        first = reconcile(extended, motivating_dataset.first_print.loc[[Q2]])
        assert len(reconciliation_log(first)) == 1
        second = reconcile(first, motivating_dataset.first_print.loc[[Q3]])
        log = reconciliation_log(second)
        assert len(log) == 2
        assert list(log["quarter"]) == [Q2, Q3]
        assert list(provisional_quarters(second)) == []

    def test_errors_feed_a_residual_pool(self, extended, motivating_dataset):
        """The documented downstream use: a pool from realised nowcast errors."""
        from private_assets_frequency.nowcast.uncertainty import (
            build_residual_pool,
        )

        official = motivating_dataset.first_print.loc[[Q2, Q3]]
        log = reconciliation_log(reconcile(extended, official))
        by_horizon = {
            int(h): group["error"].to_numpy()
            for h, group in log.groupby("horizon")
        }
        with pytest.warns(Warning):
            pool = build_residual_pool(
                by_horizon,
                np.full(60, 0.005),
                n_params=3,
                horizons=(1, 2),
                label="from_log",
            )
        # Only one realised error per horizon so far, so the pool correctly falls
        # back rather than pretending a single error is a distribution.
        assert pool.n_oos == {1: 1, 2: 1}
        assert pool.is_fallback(1) and pool.is_fallback(2)

    def test_new_quarters_are_appended_as_published(self, extended):
        future = pd.Series(
            [0.02], index=pd.PeriodIndex([pd.Period("2026Q4", freq="Q")], freq="Q")
        )
        result = reconcile(extended, future)
        assert result.index[-1] == pd.Period("2026Q4", freq="Q")
        assert not bool(result.loc[pd.Period("2026Q4", freq="Q"), "is_provisional"])
        # The provisional rows are untouched: nothing was published for them.
        assert list(provisional_quarters(result)) == [Q2, Q3]

    def test_a_revision_to_a_published_row_warns_and_is_not_logged(self, extended):
        """A revision is not a reconciliation — the nowcast was never scored on it."""
        quarter = pd.Period("2026Q1", freq="Q")
        revised = pd.Series(
            [float(extended.loc[quarter, "value"]) + 0.01],
            index=pd.PeriodIndex([quarter], freq="Q"),
        )
        with pytest.warns(ProvisionalDataWarning, match="arrived\n?\\s*with a different"):
            result = reconcile(extended, revised)
        assert result.loc[quarter, "value"] == pytest.approx(float(revised[quarter]))
        assert reconciliation_log(result).empty

    def test_reconciling_twice_is_idempotent(self, extended, motivating_dataset):
        official = motivating_dataset.first_print.loc[[Q2, Q3]]
        once = reconcile(extended, official)
        twice = reconcile(once, official)
        pd.testing.assert_frame_equal(once, twice)
        assert len(reconciliation_log(twice)) == 2

    def test_empty_log_before_any_reconciliation(self, extended):
        log = reconciliation_log(extended)
        assert log.empty
        assert "error" in log.columns

    def test_input_validation(self, extended):
        with pytest.raises(TypeError, match="new_reported must be a Series"):
            reconcile(extended, pd.DataFrame())
        with pytest.raises(KeyError, match="missing columns"):
            reconcile(extended.drop(columns="as_of"), pd.Series(dtype=float))


# ──────────────────────────────────────────────────────────────────
# Spec test 10 — end to end through the pipeline
# ──────────────────────────────────────────────────────────────────


class TestPipelineIntegration:
    @pytest.fixture(scope="class")
    def runs(self, extended, motivating_dataset, pe_config):
        """The provisional run and the published-only run, for comparison."""
        monthly = _monthly_factors(motivating_dataset, extended)
        provisional_returns = to_pipeline_returns({"pe": extended})
        published = extended.loc[~extended["is_provisional"]]
        published_returns = to_pipeline_returns({"pe": published})
        with_prov = _run(
            provisional_returns, monthly, pe_config, allow_provisional=True
        )
        without = _run(
            published_returns,
            monthly.iloc[: 3 * len(published)],
            pe_config,
        )
        return with_prov, without

    def test_runs_end_to_end(self, runs):
        with_prov, _ = runs
        assert "pe" in with_prov.monthly_returns.columns
        assert np.isfinite(with_prov.monthly_returns["pe"].to_numpy()).all()

    def test_desmoothing_parameters_are_identical(self, runs):
        r"""The guarantee, asserted exactly: ``==``, not ``approx``.

        Provisional rows must not influence an estimate *at all*. A tolerance
        would let a small influence through and report it as success.
        """
        with_prov, without = runs
        a = with_prov.per_strategy["pe"].desmoothed.smoothing_params
        b = without.per_strategy["pe"].desmoothed.smoothing_params
        assert a["lambda"] == b["lambda"]
        assert a["beta"] == b["beta"]
        assert a["alpha"] == b["alpha"]
        assert a["sigma_eps"] == b["sigma_eps"]

    def test_factor_model_parameters_are_identical(self, runs):
        with_prov, without = runs
        a = with_prov.per_strategy["pe"].factor_regression
        b = without.per_strategy["pe"].factor_regression
        assert a.betas == b.betas
        assert a.alpha == b.alpha
        assert a.sigma_eps == b.sigma_eps
        assert a.n_obs == b.n_obs

    def test_the_desmoothed_published_series_is_identical(self, runs):
        with_prov, without = runs
        pd.testing.assert_series_equal(
            with_prov.per_strategy["pe"].desmoothed.true_returns,
            without.per_strategy["pe"].desmoothed.true_returns,
            check_exact=True,
        )

    def test_the_estimation_window_stops_at_the_last_published_quarter(self, runs):
        with_prov, _ = runs
        diagnostics = with_prov.per_strategy["pe"].disaggregation.diagnostics
        assert diagnostics["estimation_window"] == ("2001Q1", "2026Q1")
        assert diagnostics["provisional_quarters"] == ["2026Q2", "2026Q3"]

    def test_the_monthly_output_extends_by_three_months_per_quarter(self, runs):
        with_prov, without = runs
        assert len(with_prov.monthly_returns) == len(without.monthly_returns) + 6
        assert with_prov.monthly_returns.index[-1] > without.monthly_returns.index[-1]

    def test_provisional_rows_are_flagged_in_the_outputs(self, runs):
        with_prov, _ = runs
        assert with_prov.diagnostics["allow_provisional"] is True
        block = with_prov.diagnostics["is_provisional"]
        assert block["enabled"] is True
        assert block["pe"]["n_provisional_quarters"] == 2
        assert block["pe"]["n_provisional_high_frequency"] == 6

        markers = provisional_markers(with_prov)
        monthly = markers["pe"]["monthly"]
        assert monthly.index.equals(with_prov.per_strategy["pe"].disaggregation.high_frequency.index)
        assert int(monthly.sum()) == 6
        assert bool(monthly.iloc[-1])
        assert not bool(monthly.iloc[-7])

    def test_provisional_markers_refuse_a_published_only_run(self, runs):
        _, without = runs
        with pytest.raises(ValueError, match="allow_provisional=False"):
            provisional_markers(without)

    def test_aggregation_constraint_holds_on_provisional_quarters_too(self, runs):
        """The upsampled months must compound back to the provisional nowcast."""
        with_prov, _ = runs
        strat = with_prov.per_strategy["pe"]
        monthly = strat.disaggregation.high_frequency
        low = strat.disaggregation.low_frequency_input
        quarters = pd.PeriodIndex(monthly.index, freq="Q")
        for quarter in (Q2, Q3):
            block = monthly.loc[quarters == quarter]
            assert len(block) == 3
            compounded = float(np.expm1(np.log1p(block.to_numpy()).sum()))
            target = float(low.loc[low.index[pd.PeriodIndex(low.index, freq="Q") == quarter][0]])
            assert compounded == pytest.approx(target, abs=1e-10)

    def test_daily_output_is_flagged_too(
        self, extended, motivating_dataset, pe_config
    ):
        monthly = _monthly_factors(motivating_dataset, extended)
        daily = motivating_dataset.daily_factors.loc[:"2026-09-30"]
        result = _run(
            to_pipeline_returns({"pe": extended}),
            monthly,
            pe_config,
            factor_returns_daily=daily,
            allow_provisional=True,
        )
        if result.per_strategy["pe"].daily is None:
            pytest.skip("Stage 4 did not produce a daily series for this fixture")
        markers = provisional_markers(result)
        assert "daily" in markers["pe"]
        assert int(markers["pe"]["daily"].sum()) > 0
        assert bool(markers["pe"]["daily"].iloc[-1])

    def test_reconciling_then_rerunning_matches_the_published_path(
        self, extended, motivating_dataset, pe_config
    ):
        """Once the prints land, the provisional machinery leaves no trace."""
        official = motivating_dataset.first_print.loc[[Q2, Q3]]
        reconciled = reconcile(extended, official)
        assert list(provisional_quarters(reconciled)) == []
        monthly = _monthly_factors(motivating_dataset, reconciled)
        returns = to_pipeline_returns({"pe": reconciled})
        assert returns.attrs["provisional_mask"].to_numpy().sum() == 0
        # No provisional rows left, so allow_provisional=True is a no-op ...
        with_flag = _run(returns, monthly, pe_config, allow_provisional=True)
        # ... and must agree exactly with the ordinary path.
        without_flag = _run(
            to_pipeline_returns({"pe": reconciled})
            .pipe(lambda f: f.drop(columns=[]).assign())
            .pipe(lambda f: f),
            monthly,
            pe_config,
        )
        a = with_flag.per_strategy["pe"].desmoothed.smoothing_params
        b = without_flag.per_strategy["pe"].desmoothed.smoothing_params
        assert a["lambda"] == b["lambda"]
        pd.testing.assert_frame_equal(
            with_flag.monthly_returns, without_flag.monthly_returns, check_exact=True
        )


class TestPipelineProvisionalGuards:
    def test_allow_provisional_without_a_mask_raises(
        self, extended, motivating_dataset, pe_config
    ):
        """It must not guess which rows are nowcasts — it would estimate on them."""
        monthly = _monthly_factors(motivating_dataset, extended)
        bare = pd.DataFrame({"pe": extended["value"].to_numpy()})
        bare.index = pd.PeriodIndex(extended.index, freq="Q").to_timestamp(
            how="end"
        ).normalize()
        with pytest.raises(ValueError, match="provisional_mask'\\] is absent"):
            FrequencyPipeline(
                returns=bare,
                factor_returns_monthly=monthly,
                configs={"pe": pe_config},
                allow_provisional=True,
            )

    def test_a_mask_without_the_flag_warns_loudly(
        self, extended, motivating_dataset, pe_config
    ):
        """Silently estimating on nowcasts is the exact failure to prevent."""
        monthly = _monthly_factors(motivating_dataset, extended)
        returns = to_pipeline_returns({"pe": extended})
        with pytest.warns(UserWarning, match="WILL enter the desmoothing"):
            FrequencyPipeline(
                returns=returns,
                factor_returns_monthly=monthly,
                configs={"pe": pe_config},
                allow_provisional=False,
            )

    def test_flag_defaults_to_false(self, extended, motivating_dataset, pe_config):
        monthly = _monthly_factors(motivating_dataset, extended)
        published = extended.loc[~extended["is_provisional"]]
        pipeline = FrequencyPipeline(
            returns=to_pipeline_returns({"pe": published}),
            factor_returns_monthly=monthly.iloc[: 3 * len(published)],
            configs={"pe": pe_config},
        )
        assert pipeline.allow_provisional is False
        result = pipeline.run()
        assert result.diagnostics["allow_provisional"] is False
        assert result.diagnostics["is_provisional"] == {"enabled": False}

    def test_all_rows_provisional_is_refused(
        self, extended, motivating_dataset, pe_config
    ):
        monthly = _monthly_factors(motivating_dataset, extended)
        returns = to_pipeline_returns({"pe": extended})
        returns.attrs["provisional_mask"] = pd.DataFrame(
            True, index=returns.index, columns=returns.columns
        )
        with pytest.raises(ValueError, match="nothing to estimate from"):
            FrequencyPipeline(
                returns=returns,
                factor_returns_monthly=monthly,
                configs={"pe": pe_config},
                allow_provisional=True,
            ).run()

    def test_non_trailing_provisional_rows_are_refused(
        self, extended, motivating_dataset, pe_config
    ):
        monthly = _monthly_factors(motivating_dataset, extended)
        returns = to_pipeline_returns({"pe": extended})
        mask = returns.attrs["provisional_mask"].copy()
        mask.iloc[-1, 0] = False  # a published row after a provisional one
        returns.attrs["provisional_mask"] = mask
        with pytest.raises(ValueError, match="trailing block"):
            FrequencyPipeline(
                returns=returns,
                factor_returns_monthly=monthly,
                configs={"pe": pe_config},
                allow_provisional=True,
            ).run()


# ──────────────────────────────────────────────────────────────────
# The spec's motivating scenario, end to end
# ──────────────────────────────────────────────────────────────────


class TestMotivatingScenario:
    r"""The module spec's opening paragraph, as one executable acceptance test.

    > On 7 October 2026 the Q2-2026 private equity index return has not been
    > published, Q3-2026 has just closed, and public market data is available
    > through today. The module must produce nowcasts for both missing quarters,
    > with prediction intervals, and hand them to the rest of the pipeline as
    > clearly flagged provisional observations.

    Every other test checks one layer. This one walks the whole path — information
    set, model, intervals, extension, pipeline, reconciliation — so a regression
    that only shows up at a seam cannot hide.
    """

    def test_the_whole_path(self, motivating_dataset, pe_config):
        # ── 1. what is knowable on 7 October 2026 ──────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = build_information_set(
                motivating_dataset.reported,
                motivating_dataset.daily_factors,
                AS_OF,
                vintages=motivating_dataset.vintages,
            )
        assert info.as_of == pd.Timestamp("2026-10-07")
        assert info.last_published_quarter == pd.Period("2026Q1", freq="Q")
        assert info.unpublished_quarters() == [Q2, Q3]
        assert info.horizon_of(Q2) == 1 and info.horizon_of(Q3) == 2
        # "public market data is available through today", and no further
        assert info.public_factors.index[-1] <= AS_OF
        assert info.public_factors.index[-1] >= AS_OF - pd.Timedelta(days=5)
        # Q3 has just closed, so its factor aggregate is complete; Q4 has not.
        assert bool(info.factor_quarter_complete[Q3])
        assert not bool(info.factor_quarter_complete[pd.Period("2026Q4", freq="Q")])

        # ── 2. nowcasts for both missing quarters, with intervals ──
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SmoothingRegressionNowcaster(n_lags=1).fit(info)
            results = model.predict(
                info, [Q2, Q3], n_draws=4000, rng=np.random.default_rng(20261007)
            )
        assert [r.quarter for r in results] == [Q2, Q3]
        assert [r.horizon for r in results] == [1, 2]
        for result in results:
            assert np.isfinite(result.point)
            lo80, hi80 = result.interval_80
            lo95, hi95 = result.interval_95
            assert lo80 < result.point < hi80
            assert lo95 <= lo80 and hi95 >= hi80
            assert result.as_of == AS_OF
            assert result.target == "reported"
        # The h=2 interval must be wider: it conditions on a nowcast, not a print.
        assert results[1].draw_std > results[0].draw_std

        # ── 3. handed on as clearly flagged provisional observations ──
        extended = extend_with_nowcasts(info.reported, results)
        assert list(provisional_quarters(extended)) == [Q2, Q3]
        assert not extended.loc[: pd.Period("2026Q1", freq="Q"), "is_provisional"].any()
        assert (
            extended.loc[[Q2, Q3], "nowcast_model"] == "smoothing_regression"
        ).all()

        # ── 4. the pipeline extends monthly output to the present ──
        monthly = _monthly_factors(motivating_dataset, extended)
        result = _run(
            to_pipeline_returns({"pe": extended}),
            monthly,
            pe_config,
            allow_provisional=True,
        )
        markers = provisional_markers(result)["pe"]["monthly"]
        assert int(markers.sum()) == 6  # two quarters, three months each
        assert markers.index[-1] >= pd.Timestamp("2026-09-30")
        # ... and the estimates never saw the nowcasts.
        assert result.per_strategy["pe"].disaggregation.diagnostics[
            "estimation_window"
        ] == ("2001Q1", "2026Q1")

        # ── 5. and when the prints land, they are scored ───────────
        official = motivating_dataset.first_print.loc[[Q2, Q3]]
        reconciled = reconcile(extended, official)
        log = reconciliation_log(reconciled)
        assert list(provisional_quarters(reconciled)) == []
        assert list(log["quarter"]) == [Q2, Q3]
        assert list(log["horizon"]) == [1, 2]
        assert log["error"].notna().all()
        # The nowcasts should be in the right ballpark: errors well inside the
        # spread of the reported series itself, or the module is not working.
        reported_sd = float(info.reported.std())
        assert (log["error"].abs() < reported_sd).all(), log.to_string()
