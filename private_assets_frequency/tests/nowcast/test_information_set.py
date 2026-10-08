"""
Tests for ``nowcast/information_set.py`` — the leakage and vintage tests.

Test 1 of the module spec (the leakage test) is the acceptance criterion for
the whole package. Here it is enforced at the level of the information set: the
object a model receives must be *identical* whether or not post-``as_of`` data
exists or is perturbed. ``test_models.py`` then closes the loop by asserting the
same thing about predictions, bit for bit.

Test 2 (the vintage test) checks that the value seen for a quarter at ``as_of``
is the latest release at or before ``as_of``, and that unpublished quarters are
assigned horizons 1..H.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import (
    AssetClassConfig,
    BetaDist,
    FallbackPolicy,
    NormalPrior,
)
from private_assets_frequency.nowcast.information_set import (
    DEFAULT_PUBLICATION_LAG_DAYS,
    InformationSet,
    NowcastDataModeWarning,
    build_information_set,
    first_print_values,
    latest_values,
    quarterly_from_high_frequency,
    resolve_factor_columns,
)

AS_OF = pd.Timestamp("2026-10-07")


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _build(ds, **kwargs) -> InformationSet:
    """Build a vintage-mode information set from a synthetic dataset."""
    params = dict(
        reported=ds.reported,
        public_factors=ds.daily_factors,
        as_of=AS_OF,
        vintages=ds.vintages,
    )
    params.update(kwargs)
    return build_information_set(**params)


def _assert_info_identical(a: InformationSet, b: InformationSet) -> None:
    """Assert two information sets carry exactly the same data.

    Uses exact equality, not ``allclose``: a single float of difference means a
    post-``as_of`` observation reached the object.
    """
    assert a.as_of == b.as_of
    assert a.data_mode == b.data_mode
    assert a.factor_columns == b.factor_columns
    pd.testing.assert_series_equal(a.reported, b.reported, check_exact=True)
    pd.testing.assert_series_equal(a.release_dates, b.release_dates, check_exact=True)
    pd.testing.assert_series_equal(a.n_releases, b.n_releases, check_exact=True)
    pd.testing.assert_frame_equal(
        a.public_factors, b.public_factors, check_exact=True
    )
    pd.testing.assert_frame_equal(
        a.factors_quarterly, b.factors_quarterly, check_exact=True
    )
    pd.testing.assert_series_equal(
        a.factor_quarter_complete, b.factor_quarter_complete, check_exact=True
    )
    if a.vintages is None:
        assert b.vintages is None
    else:
        pd.testing.assert_frame_equal(a.vintages, b.vintages, check_exact=True)
    if a.early_signals is None:
        assert b.early_signals is None
    else:
        pd.testing.assert_frame_equal(
            a.early_signals, b.early_signals, check_exact=True
        )


# ──────────────────────────────────────────────────────────────────
# The motivating case
# ──────────────────────────────────────────────────────────────────


class TestMotivatingCase:
    """7 October 2026: 2026Q2 and 2026Q3 are closed and unpublished."""

    def test_last_published_is_q1(self, motivating_dataset):
        info = _build(motivating_dataset)
        assert info.last_published_quarter == pd.Period("2026Q1", freq="Q")

    def test_two_unpublished_quarters_at_horizons_one_and_two(
        self, motivating_dataset
    ):
        info = _build(motivating_dataset)
        unpub = info.unpublished_quarters()
        assert unpub == [pd.Period("2026Q2", freq="Q"), pd.Period("2026Q3", freq="Q")]
        assert info.horizon_of("2026Q2") == 1
        assert info.horizon_of("2026Q3") == 2

    def test_open_quarter_excluded_by_default(self, motivating_dataset):
        info = _build(motivating_dataset)
        assert info.last_closed_quarter == pd.Period("2026Q3", freq="Q")
        assert pd.Period("2026Q4", freq="Q") not in info.unpublished_quarters()
        with_open = info.unpublished_quarters(include_open_quarter=True)
        assert with_open[-1] == pd.Period("2026Q4", freq="Q")

    def test_q3_factor_data_is_complete_q4_is_not(self, motivating_dataset):
        info = _build(motivating_dataset)
        assert bool(info.factor_quarter_complete[pd.Period("2026Q3", freq="Q")])
        assert not bool(info.factor_quarter_complete[pd.Period("2026Q4", freq="Q")])
        # The open quarter is covered only to 7 October — one week of 92 days.
        cov = info.factor_quarter_coverage[pd.Period("2026Q4", freq="Q")]
        assert 0.0 < cov < 0.15

    def test_published_quarter_is_not_a_nowcast_target(self, motivating_dataset):
        info = _build(motivating_dataset)
        with pytest.raises(ValueError, match="already published"):
            info.horizon_of("2026Q1")

    def test_factor_row_refuses_incomplete_quarter_by_default(
        self, motivating_dataset
    ):
        info = _build(motivating_dataset)
        with pytest.raises(ValueError, match="incomplete"):
            info.factor_row("2026Q4")
        partial = info.factor_row("2026Q4", require_complete=False)
        assert np.isfinite(partial).all()


# ──────────────────────────────────────────────────────────────────
# Test 2 — vintages
# ──────────────────────────────────────────────────────────────────


class TestVintageSelection:
    """The value seen at ``as_of`` is the latest release at or before it."""

    @pytest.mark.parametrize(
        "as_of",
        ["2026-07-10", "2026-10-07", "2026-10-09", "2027-01-08", "2027-02-01"],
    )
    def test_value_is_latest_release_at_or_before_as_of(
        self, motivating_dataset, as_of
    ):
        as_of_ts = pd.Timestamp(as_of)
        info = _build(motivating_dataset, as_of=as_of_ts)
        v = motivating_dataset.vintages

        for quarter, value in info.reported.items():
            visible = v.loc[
                (v["quarter"] == quarter) & (v["release_date"] <= as_of_ts)
            ]
            assert not visible.empty, f"{quarter} published with no visible release"
            expected = visible.sort_values("release_date")["value"].iloc[-1]
            assert value == pytest.approx(expected, abs=0.0, rel=0.0)

    def test_quarters_with_no_visible_release_are_absent(self, motivating_dataset):
        info = _build(motivating_dataset)
        v = motivating_dataset.vintages
        unseen = set(v.loc[v["release_date"] > AS_OF, "quarter"]) - set(
            v.loc[v["release_date"] <= AS_OF, "quarter"]
        )
        assert unseen, "fixture should leave at least one quarter unpublished"
        for quarter in unseen:
            assert quarter not in info.reported.index

    def test_n_releases_counts_only_visible_releases(self, motivating_dataset):
        info = _build(motivating_dataset)
        q1 = pd.Period("2026Q1", freq="Q")
        # At 2026-10-07 only the 2026-07-09 print of 2026Q1 has landed.
        assert info.n_releases[q1] == 1
        later = _build(motivating_dataset, as_of=pd.Timestamp("2026-10-09"))
        assert later.n_releases[q1] == 2
        assert later.reported[q1] != info.reported[q1]

    def test_release_dates_match_the_chosen_vintage(self, motivating_dataset):
        info = _build(motivating_dataset)
        v = motivating_dataset.vintages
        for quarter, release in info.release_dates.items():
            visible = v.loc[
                (v["quarter"] == quarter) & (v["release_date"] <= AS_OF)
            ]
            assert release == visible["release_date"].max()

    def test_a_revision_changes_the_lagged_predictor(self, motivating_dataset):
        """The point of vintage mode: ``s_{Q-1}`` depends on the vintage in force."""
        early = _build(motivating_dataset, as_of=pd.Timestamp("2026-10-07"))
        late = _build(motivating_dataset, as_of=pd.Timestamp("2027-02-01"))
        q1 = pd.Period("2026Q1", freq="Q")
        assert early.reported[q1] != late.reported[q1]
        # ... and the final revised series is not what the early view saw.
        final = motivating_dataset.reported[q1]
        assert early.reported[q1] != final

    def test_evaluation_target_defaults_to_first_print_in_vintage_mode(
        self, motivating_dataset
    ):
        info = _build(motivating_dataset)
        assert info.evaluation_target == "first_print"
        assert info.data_mode == "vintage"

    def test_truncated_vintages_contain_no_future_releases(self, motivating_dataset):
        info = _build(motivating_dataset)
        assert info.vintages is not None
        assert (info.vintages["release_date"] <= AS_OF).all()


class TestVintageValidation:
    def test_missing_columns(self, motivating_dataset):
        bad = motivating_dataset.vintages.drop(columns="value")
        with pytest.raises(KeyError, match="missing required columns"):
            _build(motivating_dataset, vintages=bad)

    def test_release_before_quarter_end_is_rejected(self, motivating_dataset):
        bad = motivating_dataset.vintages.copy()
        bad.loc[0, "release_date"] = pd.Timestamp("2001-01-15")
        with pytest.raises(ValueError, match="dated before the end of the quarter"):
            _build(motivating_dataset, vintages=bad)

    def test_nan_value_is_rejected(self, motivating_dataset):
        bad = motivating_dataset.vintages.copy()
        bad.loc[5, "value"] = np.nan
        with pytest.raises(ValueError, match="contains NaN"):
            _build(motivating_dataset, vintages=bad)

    def test_no_visible_release_raises(self, motivating_dataset):
        with pytest.raises(ValueError, match="no reported value is published"):
            _build(motivating_dataset, as_of=pd.Timestamp("2001-02-01"))

    def test_first_print_and_latest_differ(self, motivating_dataset):
        fp = first_print_values(motivating_dataset.vintages)
        lv = latest_values(motivating_dataset.vintages)
        pd.testing.assert_series_equal(
            lv, motivating_dataset.reported.rename("latest"), check_exact=False
        )
        pd.testing.assert_series_equal(
            fp, motivating_dataset.first_print.rename("first_print")
        )
        assert (fp != lv).any()


# ──────────────────────────────────────────────────────────────────
# Lag-rule mode
# ──────────────────────────────────────────────────────────────────


class TestLagRuleMode:
    def test_warns_about_pseudo_real_time(self, motivating_dataset):
        with pytest.warns(NowcastDataModeWarning, match="pseudo-real-time"):
            info = build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
            )
        assert info.data_mode == "lag_rule"
        assert info.publication_lag == pd.Timedelta(
            days=DEFAULT_PUBLICATION_LAG_DAYS
        )
        assert info.evaluation_target == "latest"

    def test_strict_policy_refuses_lag_rule_mode(self, motivating_dataset):
        with pytest.raises(NowcastDataModeWarning, match="pseudo-real-time"):
            build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                fallback_policy=FallbackPolicy.STRICT,
            )

    def test_known_set_follows_the_lag_rule_exactly(self, motivating_dataset):
        lag_days = 100
        with pytest.warns(NowcastDataModeWarning):
            info = build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                publication_lag=lag_days,
            )
        for quarter in motivating_dataset.reported.index:
            due = quarter.end_time.normalize() + pd.Timedelta(days=lag_days)
            # `end_time` carries a nanosecond-resolution time of day; normalising
            # matches how the builder derives publication dates from it.
            expected_known = quarter.end_time + pd.Timedelta(days=lag_days) <= AS_OF
            assert (quarter in info.reported.index) == expected_known, (
                f"{quarter} due {due.date()}"
            )

    def test_lag_rule_uses_final_revised_values(self, motivating_dataset):
        with pytest.warns(NowcastDataModeWarning):
            info = build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
            )
        q1 = pd.Period("2026Q1", freq="Q")
        # This is the flattering bit the warning is about: the value in hand is
        # the fully revised one, which in real time did not exist yet.
        assert info.reported[q1] == motivating_dataset.reported[q1]

    def test_zero_lag_publishes_every_closed_quarter(self, motivating_dataset):
        with pytest.warns(NowcastDataModeWarning):
            info = build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                publication_lag=0,
            )
        assert info.last_published_quarter == pd.Period("2026Q3", freq="Q")
        assert info.unpublished_quarters() == []

    def test_first_print_target_requires_vintages(self, motivating_dataset):
        with pytest.raises(ValueError, match="needs a vintage table"):
            build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                evaluation_target="first_print",
            )

    def test_lag_rule_without_reported_raises(self, motivating_dataset):
        with pytest.raises(ValueError, match="requires `reported`"):
            build_information_set(
                reported=None,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
            )

    def test_negative_lag_rejected(self, motivating_dataset):
        with pytest.raises(ValueError, match="must be >= 0 days"):
            build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                publication_lag=-5,
            )


class TestVintageBackfill:
    """A short vintage table should not silently cost years of training data."""

    def test_backfill_recovers_pre_vintage_history(self, motivating_dataset):
        cut = pd.Period("2015Q1", freq="Q")
        short = motivating_dataset.vintages.loc[
            motivating_dataset.vintages["quarter"] >= cut
        ].reset_index(drop=True)
        with pytest.warns(NowcastDataModeWarning, match="backfilled"):
            info = build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                vintages=short,
            )
        assert info.reported.index[0] == motivating_dataset.reported.index[0]
        assert len(info.diagnostics["backfilled_quarters"]) > 50
        # Backfilled rows carry the final revised value under the lag rule.
        q = pd.Period("2010Q2", freq="Q")
        assert info.reported[q] == motivating_dataset.reported[q]
        # In-vintage rows carry the vintage in force, not the revised value.
        q1 = pd.Period("2026Q1", freq="Q")
        assert info.reported[q1] != motivating_dataset.reported[q1]

    def test_backfill_off_keeps_only_vintage_quarters(self, motivating_dataset):
        cut = pd.Period("2015Q1", freq="Q")
        short = motivating_dataset.vintages.loc[
            motivating_dataset.vintages["quarter"] >= cut
        ].reset_index(drop=True)
        info = build_information_set(
            reported=motivating_dataset.reported,
            public_factors=motivating_dataset.daily_factors,
            as_of=AS_OF,
            vintages=short,
            vintage_backfill=False,
        )
        assert info.reported.index[0] == cut
        assert info.diagnostics["backfilled_quarters"] == []


# ──────────────────────────────────────────────────────────────────
# Test 1 — leakage
# ──────────────────────────────────────────────────────────────────


class TestNoLeakage:
    """Nothing dated after ``as_of`` may touch the information set."""

    def test_perturbing_future_factor_rows_changes_nothing(
        self, motivating_dataset
    ):
        base = _build(motivating_dataset)
        perturbed = motivating_dataset.daily_factors.copy()
        future = perturbed.index > AS_OF
        assert future.sum() > 100, "fixture must have post-as_of factor data"
        perturbed.loc[future] += 0.25
        other = _build(motivating_dataset, public_factors=perturbed)
        _assert_info_identical(base, other)

    def test_truncating_future_factor_rows_changes_nothing(
        self, motivating_dataset
    ):
        """Deleting the future entirely must be indistinguishable from keeping it."""
        base = _build(motivating_dataset)
        truncated = motivating_dataset.daily_factors.loc[
            motivating_dataset.daily_factors.index <= AS_OF
        ]
        other = _build(motivating_dataset, public_factors=truncated)
        _assert_info_identical(base, other)

    def test_perturbing_future_vintage_rows_changes_nothing(
        self, motivating_dataset
    ):
        base = _build(motivating_dataset)
        perturbed = motivating_dataset.vintages.copy()
        future = perturbed["release_date"] > AS_OF
        assert future.sum() > 5
        perturbed.loc[future, "value"] += 0.5
        other = _build(motivating_dataset, vintages=perturbed)
        _assert_info_identical(base, other)

    def test_dropping_future_vintage_rows_changes_nothing(self, motivating_dataset):
        base = _build(motivating_dataset)
        visible_only = motivating_dataset.vintages.loc[
            motivating_dataset.vintages["release_date"] <= AS_OF
        ].reset_index(drop=True)
        other = _build(motivating_dataset, vintages=visible_only)
        _assert_info_identical(base, other)

    def test_perturbing_unpublished_reported_values_changes_nothing(
        self, motivating_dataset
    ):
        """In lag-rule mode, a quarter past its reference end but pre-publication."""
        perturbed = motivating_dataset.reported.copy()
        perturbed.loc[pd.Period("2026Q2", freq="Q")] += 1.0
        perturbed.loc[pd.Period("2026Q3", freq="Q")] += 1.0
        with pytest.warns(NowcastDataModeWarning):
            base = build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
            )
        with pytest.warns(NowcastDataModeWarning):
            other = build_information_set(
                reported=perturbed,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
            )
        _assert_info_identical(base, other)

    def test_perturbing_future_early_signals_changes_nothing(
        self, early_signal_dataset
    ):
        as_of = pd.Timestamp("2020-07-30")
        kw = dict(
            reported=early_signal_dataset.reported,
            public_factors=early_signal_dataset.daily_factors,
            as_of=as_of,
            vintages=early_signal_dataset.vintages,
        )
        base = build_information_set(
            early_signals=early_signal_dataset.early_signals, **kw
        )
        perturbed = early_signal_dataset.early_signals.copy()
        perturbed.loc[perturbed.index > as_of] += 10.0
        other = build_information_set(early_signals=perturbed, **kw)
        _assert_info_identical(base, other)
        assert base.early_signals is not None
        assert (base.early_signals.index <= as_of).all()

    def test_mutating_inputs_after_build_changes_nothing(self, motivating_dataset):
        """The information set holds copies, not views."""
        factors = motivating_dataset.daily_factors.copy()
        reported = motivating_dataset.reported.copy()
        vintages = motivating_dataset.vintages.copy()
        info = build_information_set(
            reported=reported,
            public_factors=factors,
            as_of=AS_OF,
            vintages=vintages,
        )
        snapshot = (
            info.reported.copy(),
            info.public_factors.copy(),
            info.factors_quarterly.copy(),
        )
        factors.iloc[:, :] = 99.0
        reported.iloc[:] = 99.0
        vintages["value"] = 99.0
        pd.testing.assert_series_equal(info.reported, snapshot[0], check_exact=True)
        pd.testing.assert_frame_equal(
            info.public_factors, snapshot[1], check_exact=True
        )
        pd.testing.assert_frame_equal(
            info.factors_quarterly, snapshot[2], check_exact=True
        )

    def test_no_public_factor_observation_after_as_of(self, motivating_dataset):
        info = _build(motivating_dataset)
        assert (info.public_factors.index <= AS_OF).all()
        assert info.public_factors.index[-1] <= AS_OF
        assert info.diagnostics["last_factor_observation"] <= AS_OF


# ──────────────────────────────────────────────────────────────────
# Quarterly compounding
# ──────────────────────────────────────────────────────────────────


class TestQuarterlyCompounding:
    def test_matches_product_of_one_plus_r(self, motivating_dataset):
        daily = motivating_dataset.daily_factors.loc[:AS_OF]
        quarterly, n_obs, coverage = quarterly_from_high_frequency(daily)
        q = pd.Period("2026Q1", freq="Q")
        block = daily.loc[(daily.index >= q.start_time) & (daily.index <= q.end_time)]
        expected = (1.0 + block["equity_market"]).prod() - 1.0
        assert quarterly.loc[q, "equity_market"] == pytest.approx(expected, rel=1e-12)
        assert n_obs[q] == len(block)
        assert coverage[q] == pytest.approx(1.0, abs=0.05)

    def test_monthly_input_gives_the_same_quarterly_returns(
        self, motivating_dataset
    ):
        daily = motivating_dataset.daily_factors.loc[:"2026-06-30"]
        monthly = motivating_dataset.monthly_factors.loc[:"2026-06-30"]
        from_daily, _, _ = quarterly_from_high_frequency(daily)
        from_monthly, _, _ = quarterly_from_high_frequency(monthly)
        pd.testing.assert_frame_equal(
            from_daily, from_monthly, check_exact=False, rtol=1e-12
        )

    def test_rejects_return_at_or_below_minus_one(self, motivating_dataset):
        bad = motivating_dataset.daily_factors.copy()
        bad.iloc[10, 0] = -1.0
        with pytest.raises(ValueError, match="cannot compound"):
            quarterly_from_high_frequency(bad)

    def test_empty_frame_returns_empty_quarterly(self):
        empty = pd.DataFrame(
            columns=["a"], index=pd.DatetimeIndex([], name=None), dtype=float
        )
        quarterly, n_obs, coverage = quarterly_from_high_frequency(empty)
        assert quarterly.empty and n_obs.empty and coverage.empty


# ──────────────────────────────────────────────────────────────────
# Design frame
# ──────────────────────────────────────────────────────────────────


class TestTrainingFrame:
    def test_columns_and_lag_alignment(self, motivating_dataset):
        info = _build(motivating_dataset)
        frame = info.training_frame(n_lags=2)
        assert list(frame.columns) == ["s", "equity_market", "s_lag1", "s_lag2"]
        q = frame.index[10]
        assert frame.loc[q, "s_lag1"] == info.reported[q - 1]
        assert frame.loc[q, "s_lag2"] == info.reported[q - 2]
        assert frame.loc[q, "equity_market"] == info.factors_quarterly.loc[
            q, "equity_market"
        ]

    def test_loses_n_lags_rows(self, motivating_dataset):
        info = _build(motivating_dataset)
        assert len(info.training_frame(n_lags=0)) == info.n_published
        assert len(info.training_frame(n_lags=1)) == info.n_published - 1
        assert len(info.training_frame(n_lags=2)) == info.n_published - 2

    def test_contains_no_unpublished_quarter(self, motivating_dataset):
        info = _build(motivating_dataset)
        frame = info.training_frame(n_lags=1)
        for quarter in info.unpublished_quarters():
            assert quarter not in frame.index

    def test_negative_lags_rejected(self, motivating_dataset):
        info = _build(motivating_dataset)
        with pytest.raises(ValueError, match="n_lags must be >= 0"):
            info.training_frame(n_lags=-1)


# ──────────────────────────────────────────────────────────────────
# Factor column resolution
# ──────────────────────────────────────────────────────────────────


def _config(**overrides) -> AssetClassConfig:
    params = dict(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        lambda_prior=BetaDist(2, 2),
        beta_priors={"equity_market": NormalPrior(1.15, 0.5)},
        alpha_prior=NormalPrior(0.0, 0.05),
        default_factors=("SP500", "Russell2000"),
    )
    params.update(overrides)
    return AssetClassConfig(**params)


class TestResolveFactorColumns:
    """``beta_priors`` keys win, per the resolution in the shipped code."""

    def test_beta_prior_keys_take_precedence(self):
        cfg = _config()
        cols = resolve_factor_columns(cfg, ["equity_market", "SP500", "noise"])
        assert cols == ["equity_market"]

    def test_falls_back_to_default_factors(self):
        cfg = _config()
        cols = resolve_factor_columns(cfg, ["SP500", "Russell2000", "noise"])
        assert cols == ["SP500", "Russell2000"]

    def test_requested_overrides_everything(self):
        cfg = _config()
        cols = resolve_factor_columns(cfg, ["equity_market", "SP500"], ["SP500"])
        assert cols == ["SP500"]

    def test_requested_missing_column_raises(self):
        with pytest.raises(KeyError, match="not in the factor panel"):
            resolve_factor_columns(_config(), ["SP500"], ["NOPE"])

    def test_no_match_raises(self):
        with pytest.raises(KeyError, match="match the factor panel"):
            resolve_factor_columns(_config(), ["unrelated"])

    def test_config_none_requires_requested(self):
        with pytest.raises(KeyError, match="needs either a config"):
            resolve_factor_columns(None, ["SP500"])

    def test_builder_uses_config(self, linear_dataset_two_factor):
        cfg = _config(
            beta_priors={"credit_proxy": NormalPrior(0.35, 0.2)},
        )
        info = build_information_set(
            reported=linear_dataset_two_factor.reported,
            public_factors=linear_dataset_two_factor.daily_factors,
            as_of=pd.Timestamp("2024-06-30"),
            vintages=linear_dataset_two_factor.vintages,
            config=cfg,
        )
        assert info.factor_columns == ("credit_proxy",)


# ──────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_duplicate_quarters_rejected(self, motivating_dataset):
        bad = pd.concat(
            [motivating_dataset.reported, motivating_dataset.reported.iloc[[0]]]
        )
        with pytest.raises(ValueError, match="duplicate quarters"):
            build_information_set(
                reported=bad,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                vintages=motivating_dataset.vintages,
            )

    def test_datetime_indexed_reported_is_accepted(self, motivating_dataset):
        as_dates = motivating_dataset.reported.copy()
        as_dates.index = pd.PeriodIndex(as_dates.index, freq="Q").to_timestamp(
            how="end"
        )
        info = build_information_set(
            reported=as_dates,
            public_factors=motivating_dataset.daily_factors,
            as_of=AS_OF,
            vintages=motivating_dataset.vintages,
        )
        assert info.last_published_quarter == pd.Period("2026Q1", freq="Q")

    def test_string_quarter_labels_accepted(self, motivating_dataset):
        as_str = motivating_dataset.reported.copy()
        as_str.index = pd.Index([str(q) for q in as_str.index])
        info = build_information_set(
            reported=as_str,
            public_factors=motivating_dataset.daily_factors,
            as_of=AS_OF,
            vintages=motivating_dataset.vintages,
        )
        assert info.last_published_quarter == pd.Period("2026Q1", freq="Q")

    def test_non_datetime_factor_index_rejected(self, motivating_dataset):
        bad = motivating_dataset.daily_factors.reset_index(drop=True)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            build_information_set(
                reported=motivating_dataset.reported,
                public_factors=bad,
                as_of=AS_OF,
                vintages=motivating_dataset.vintages,
            )

    def test_no_factor_data_before_as_of_raises(self, motivating_dataset):
        """A factor panel that starts after ``as_of`` leaves nothing to compound."""
        late_factors = motivating_dataset.daily_factors.loc["2020-01-01":]
        with pytest.raises(ValueError, match="no public factor observation"):
            build_information_set(
                reported=motivating_dataset.reported,
                public_factors=late_factors,
                as_of=pd.Timestamp("2010-06-30"),
                vintages=motivating_dataset.vintages,
            )

    def test_empty_reported_set_raises(self, motivating_dataset):
        with pytest.raises(ValueError, match="no reported value is published"):
            build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=pd.Timestamp("1990-01-01"),
                vintages=motivating_dataset.vintages,
            )

    def test_missing_factor_column_raises(self, motivating_dataset):
        with pytest.raises(KeyError, match="missing columns"):
            build_information_set(
                reported=motivating_dataset.reported,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                vintages=motivating_dataset.vintages,
                factor_columns=["nope"],
            )

    def test_max_horizon_must_be_positive(self, motivating_dataset):
        with pytest.raises(ValueError, match="max_horizon must be >= 1"):
            _build(motivating_dataset, max_horizon=0)

    def test_max_horizon_truncates_the_target_list(self, motivating_dataset):
        info = _build(motivating_dataset, max_horizon=1)
        assert info.unpublished_quarters() == [pd.Period("2026Q2", freq="Q")]

    def test_interior_gap_warns(self, motivating_dataset):
        gapped = motivating_dataset.vintages.loc[
            motivating_dataset.vintages["quarter"] != pd.Period("2010Q2", freq="Q")
        ].reset_index(drop=True)
        with pytest.warns(NowcastDataModeWarning, match="interior gaps"):
            build_information_set(
                reported=None,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                vintages=gapped,
            )

    def test_diagnostics_record_provenance(self, motivating_dataset):
        info = _build(motivating_dataset)
        diag = info.diagnostics
        assert diag["data_mode"] == "vintage"
        assert diag["evaluation_target"] == "first_print"
        assert diag["last_published_quarter"] == "2026Q1"
        assert diag["last_closed_quarter"] == "2026Q3"
        assert diag["n_published"] == info.n_published
        assert diag["factor_columns"] == ("equity_market",)
        assert diag["fallback_policy"] == FallbackPolicy.WARN.value
