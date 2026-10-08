"""
Tests for ``nowcast/evaluation.py``.

The harness is the thing that decides whether a reported result is believable, so
most of these tests are about its guardrails rather than its arithmetic: that
nested selection is *verified* and not merely assumed, that a model which fails
everywhere cannot vanish silently from the report, that overlapping
within-quarter rows are not counted as independent observations, and that the
Diebold-Mariano sign convention and small-sample correction are what they claim.
"""

from __future__ import annotations

import os
import tempfile
import warnings

# A writable config dir, set before matplotlib is imported: the default lives
# under the user's home, which is not writable in every environment, and
# matplotlib warns at import time when it has to fall back.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))

import matplotlib  # noqa: E402
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless; must precede any pyplot import

from private_assets_frequency.core.config import FallbackPolicy  # noqa: E402
from private_assets_frequency.nowcast.evaluation import (  # noqa: E402
    DEFAULT_SUBPERIODS,
    EvaluationData,
    EvaluationResult,
    EvaluationWarning,
    NestedSelectionError,
    _assert_nested_selection,
    _selection_quarters,
    diebold_mariano,
    newey_west_variance,
    plot_error_by_horizon,
    plot_nowcast_evolution,
    plot_nowcast_vs_realised,
    rolling_origin_evaluation,
)
from private_assets_frequency.nowcast.models import (  # noqa: E402
    NaiveCarryForward,
    SignatureNowcaster,
    SmoothingRegressionNowcaster,
)

from .conftest import (  # noqa: E402
    make_linear_smoothing_dataset,
    make_path_dependent_dataset,
)

# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def linear_data():
    """Linear DGP with no revisions — the benchmark should win on this.

    160 quarters rather than ~100 so the evaluation window below yields ~100
    scored points per horizon. Coverage measured on 44 points has a standard
    error of ~3pp at the 95 % level, which is too coarse to distinguish
    calibration from noise.
    """
    dataset = make_linear_smoothing_dataset(
        n_quarters=160, revision_sd=0.0, seed=4242
    )
    return dataset, EvaluationData(
        reported=dataset.reported,
        public_factors=dataset.daily_factors,
        vintages=dataset.vintages,
    )


@pytest.fixture(scope="module")
def path_data():
    """Path-dependent DGP — the signature model should win on this."""
    dataset = make_path_dependent_dataset(n_quarters=120, seed=20261008)
    return dataset, EvaluationData(
        reported=dataset.reported,
        public_factors=dataset.daily_factors,
        vintages=dataset.vintages,
    )


@pytest.fixture(scope="module")
def report(linear_data) -> EvaluationResult:
    """One evaluation run reused across many assertions, for speed."""
    _, data = linear_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rolling_origin_evaluation(
            models={
                "naive": NaiveCarryForward,
                "smoothing_regression": lambda: SmoothingRegressionNowcaster(
                    n_lags=1
                ),
            },
            data=data,
            start="2016Q1",
            end="2045Q4",
            n_draws=400,
        )


# ──────────────────────────────────────────────────────────────────
# HAC variance
# ──────────────────────────────────────────────────────────────────


class TestNeweyWestVariance:
    def test_zero_lags_is_the_iid_variance_of_the_mean(self):
        x = np.random.default_rng(0).standard_normal(500)
        assert newey_west_variance(x, 0) == pytest.approx(
            x.var(ddof=0) / x.size, rel=1e-12
        )

    def test_bartlett_weights_keep_it_non_negative(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            x = rng.standard_normal(60)
            for lags in (0, 1, 3, 8):
                assert newey_west_variance(x, lags) >= 0.0

    def test_positive_autocorrelation_inflates_the_variance(self):
        rng = np.random.default_rng(2)
        noise = rng.standard_normal(4000)
        ar1 = np.empty(4000)
        ar1[0] = noise[0]
        for t in range(1, 4000):
            ar1[t] = 0.7 * ar1[t - 1] + noise[t]
        assert newey_west_variance(ar1, 10) > 2.0 * newey_west_variance(ar1, 0)

    def test_degenerate_inputs(self):
        assert newey_west_variance(np.array([1.0]), 3) == 0.0
        assert newey_west_variance(np.array([]), 0) == 0.0
        with pytest.raises(ValueError, match="lags must be >= 0"):
            newey_west_variance(np.arange(10.0), -1)


# ──────────────────────────────────────────────────────────────────
# Diebold-Mariano
# ──────────────────────────────────────────────────────────────────


class TestDieboldMariano:
    def test_negative_statistic_favours_the_first_model(self):
        rng = np.random.default_rng(0)
        better = 0.5 * rng.standard_normal(80)
        worse = 2.0 * rng.standard_normal(80)
        test = diebold_mariano(better, worse)
        assert test["statistic"] < 0
        assert test["favours"] == "a"
        assert test["p_value"] < 0.01

    def test_sign_flips_when_the_arguments_swap(self):
        rng = np.random.default_rng(1)
        a = 0.5 * rng.standard_normal(80)
        b = 2.0 * rng.standard_normal(80)
        forward = diebold_mariano(a, b)
        backward = diebold_mariano(b, a)
        assert forward["statistic"] == pytest.approx(-backward["statistic"])
        assert forward["p_value"] == pytest.approx(backward["p_value"])
        assert backward["favours"] == "b"

    def test_identical_errors_give_no_evidence(self):
        rng = np.random.default_rng(2)
        errors = rng.standard_normal(60)
        test = diebold_mariano(errors, errors.copy())
        assert test["mean_loss_differential"] == pytest.approx(0.0)
        assert not np.isfinite(test["statistic"]), (
            "a zero loss differential has zero variance; the statistic is "
            "undefined rather than zero"
        )
        assert test["favours"] == "tie"

    def test_hln_correction_shrinks_the_statistic(self):
        """Harvey-Leybourne-Newbold: the uncorrected test over-rejects at n~70."""
        rng = np.random.default_rng(3)
        a = 0.8 * rng.standard_normal(70)
        b = rng.standard_normal(70)
        corrected = diebold_mariano(a, b, horizon=2, small_sample_correction=True)
        raw = diebold_mariano(a, b, horizon=2, small_sample_correction=False)
        assert abs(corrected["statistic"]) < abs(raw["statistic"])
        assert corrected["hln_correction"] < 1.0
        assert corrected["distribution"].startswith("t(")
        assert raw["distribution"] == "normal"
        assert corrected["p_value"] > raw["p_value"]

    def test_default_lags_respect_the_horizon(self):
        rng = np.random.default_rng(4)
        a, b = rng.standard_normal(60), rng.standard_normal(60)
        assert diebold_mariano(a, b, horizon=1)["lags"] == 3
        assert diebold_mariano(a, b, horizon=8)["lags"] == 7
        assert diebold_mariano(a, b, lags=0)["lags"] == 0

    def test_absolute_loss(self):
        rng = np.random.default_rng(5)
        a = 0.5 * rng.standard_normal(80)
        b = 2.0 * rng.standard_normal(80)
        test = diebold_mariano(a, b, loss="absolute")
        assert test["loss"] == "absolute"
        assert test["statistic"] < 0

    def test_non_finite_pairs_are_dropped(self):
        rng = np.random.default_rng(6)
        a = rng.standard_normal(40)
        b = rng.standard_normal(40)
        a[5] = np.nan
        b[9] = np.inf
        assert diebold_mariano(a, b)["n"] == 38

    def test_input_validation(self):
        with pytest.raises(ValueError, match="same length"):
            diebold_mariano(np.zeros(10), np.zeros(11))
        with pytest.raises(ValueError, match="at least 3 paired"):
            diebold_mariano(np.zeros(2), np.zeros(2))
        with pytest.raises(ValueError, match="loss must be"):
            diebold_mariano(np.arange(10.0), np.arange(10.0) + 1, loss="huber")


# ──────────────────────────────────────────────────────────────────
# Nested selection — CHECKPOINT 4
# ──────────────────────────────────────────────────────────────────


class _FakeModel:
    """A model that reports whatever selection record a test wants to inject."""

    def __init__(self, selection_index, training_quarters):
        self.fit_diagnostics = {
            "selection": {
                "selected_by": "inner_rolling_origin",
                "detail": {"cfg": {"selection_index": list(selection_index)}},
            },
            "training_quarters": training_quarters,
        }


class TestNestedSelectionVerification:
    """The harness checks the property rather than trusting it."""

    @staticmethod
    def _info(data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return data.information_set(pd.Timestamp("2026-10-07"))

    def test_reads_a_real_model_selection_index(self, linear_data):
        _, data = linear_data
        info = self._info(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SmoothingRegressionNowcaster().fit(info)
        quarters = _selection_quarters(model)
        assert quarters, "the benchmark does select, so this must not be empty"
        published = {str(q) for q in pd.PeriodIndex(info.reported.index, freq="Q")}
        assert quarters <= published

    def test_a_model_without_selection_reports_nothing(self, linear_data):
        _, data = linear_data
        info = self._info(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = NaiveCarryForward().fit(info)
        assert _selection_quarters(model) == set()

    def test_unfitted_model_reports_nothing(self):
        assert _selection_quarters(SmoothingRegressionNowcaster()) == set()

    @pytest.mark.parametrize(
        "factory",
        [
            NaiveCarryForward,
            lambda: SmoothingRegressionNowcaster(),
            lambda: SignatureNowcaster(alpha=1e-3),
        ],
    )
    def test_real_models_pass_the_check(self, linear_data, factory):
        _, data = linear_data
        info = self._info(data)
        targets = info.unpublished_quarters()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = factory().fit(info)
        record = _assert_nested_selection(
            model, info, targets, model_name="test"
        )
        assert record["verified"]

    def test_raises_when_selection_saw_an_evaluation_quarter(self, linear_data):
        """The failure the whole harness exists to catch."""
        _, data = linear_data
        info = self._info(data)
        targets = info.unpublished_quarters()
        leaky = _FakeModel(
            selection_index=[str(targets[0])],
            training_quarters=("2001Q2", str(info.last_published_quarter)),
        )
        with pytest.raises(NestedSelectionError, match="outer evaluation targets"):
            _assert_nested_selection(leaky, info, targets, model_name="leaky")

    def test_raises_when_selection_saw_an_unpublished_quarter(self, linear_data):
        _, data = linear_data
        info = self._info(data)
        targets = info.unpublished_quarters()
        leaky = _FakeModel(
            selection_index=["2040Q1"],
            training_quarters=("2001Q2", str(info.last_published_quarter)),
        )
        with pytest.raises(NestedSelectionError, match="not published"):
            _assert_nested_selection(leaky, info, targets, model_name="leaky")

    def test_raises_when_training_ran_past_the_last_published_quarter(
        self, linear_data
    ):
        """Caught separately: the selection index would not reveal this."""
        _, data = linear_data
        info = self._info(data)
        targets = info.unpublished_quarters()
        leaky = _FakeModel(
            selection_index=[], training_quarters=("2001Q2", "2030Q4")
        )
        with pytest.raises(NestedSelectionError, match="trained through 2030Q4"):
            _assert_nested_selection(leaky, info, targets, model_name="leaky")

    def test_harness_runs_the_check_at_every_date(self, report):
        diagnostics = report.diagnostics
        assert diagnostics["nested_selection_verified"]
        expected = diagnostics["n_as_of_used"] * len(diagnostics["models"])
        assert diagnostics["nested_selection_checks"] >= expected - len(
            diagnostics["models"]
        )

    def test_nested_selection_error_is_never_swallowed(self, linear_data):
        """Even under ``warn``, a leak must stop the run rather than be recorded."""
        _, data = linear_data

        class _LeakyNowcaster(SmoothingRegressionNowcaster):
            def fit(self, info):
                super().fit(info)
                # Forge a selection record naming a quarter beyond the set.
                self._state.diagnostics["training_quarters"] = ("2001Q2", "2099Q4")
                return self

        with pytest.raises(NestedSelectionError):
            rolling_origin_evaluation(
                models={"leaky": _LeakyNowcaster},
                data=data,
                start="2026Q1",
                end="2026Q4",
                as_of_offsets_days=(0,),
                n_draws=50,
                fallback_policy=FallbackPolicy.WARN,
            )


# ──────────────────────────────────────────────────────────────────
# EvaluationData
# ──────────────────────────────────────────────────────────────────


class TestEvaluationData:
    def test_realisations_default_to_first_print_in_vintage_mode(self, linear_data):
        dataset, data = linear_data
        realised = data.realisations()
        pd.testing.assert_series_equal(
            realised, dataset.first_print.rename("first_print")
        )

    def test_latest_target_differs_from_first_print(self):
        dataset = make_linear_smoothing_dataset(n_quarters=60, revision_sd=0.01)
        first = EvaluationData(
            reported=dataset.reported,
            public_factors=dataset.daily_factors,
            vintages=dataset.vintages,
            evaluation_target="first_print",
        ).realisations()
        latest = EvaluationData(
            reported=dataset.reported,
            public_factors=dataset.daily_factors,
            vintages=dataset.vintages,
            evaluation_target="latest",
        ).realisations()
        assert not np.allclose(first.to_numpy(), latest.to_numpy())
        np.testing.assert_allclose(
            latest.to_numpy(), dataset.reported.to_numpy(), atol=1e-12
        )

    def test_lag_rule_mode_scores_against_the_revised_series(self):
        dataset = make_linear_smoothing_dataset(n_quarters=60)
        data = EvaluationData(
            reported=dataset.reported, public_factors=dataset.daily_factors
        )
        np.testing.assert_allclose(
            data.realisations().to_numpy(), dataset.reported.to_numpy()
        )

    def test_information_set_is_built_from_the_dataset(self, linear_data):
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = data.information_set(pd.Timestamp("2026-10-07"))
        assert info.as_of == pd.Timestamp("2026-10-07")
        assert info.data_mode == "vintage"
        assert (info.public_factors.index <= info.as_of).all()


# ──────────────────────────────────────────────────────────────────
# The harness
# ──────────────────────────────────────────────────────────────────


class TestRollingOriginEvaluation:
    def test_row_schema(self, report):
        expected = {
            "model",
            "quarter",
            "horizon",
            "as_of",
            "offset_days",
            "point",
            "actual",
            "error",
            "s_last",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
            "covered_80",
            "covered_95",
            "interval_method",
            "n_train",
            "data_mode",
        }
        assert expected <= set(report.rows.columns)
        assert (report.rows["error"] == report.rows["actual"] - report.rows["point"]).all()
        assert report.rows["horizon"].isin([1, 2]).all()

    def test_metrics_schema(self, report):
        block = report.full_sample()
        for column in (
            "n",
            "rmse",
            "mae",
            "mean_error",
            "directional_hit_rate",
            "directional_coverage",
            "sign_hit_rate",
            "coverage_80",
            "coverage_95",
            "mean_width_80",
        ):
            assert column in block.columns
        assert set(block["model"]) == {"naive", "smoothing_regression"}
        assert set(block["horizon"]) == {1, 2}

    def test_a_class_a_factory_and_an_instance_all_work(self, linear_data):
        """Regression: a nowcaster *class* has a ``fit`` attribute, so an attribute
        check misreads it as an instance and silently drops the model."""
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={
                    "as_class": NaiveCarryForward,
                    "as_factory": lambda: NaiveCarryForward(),
                    "as_instance": NaiveCarryForward(),
                },
                data=data,
                start="2026Q1",
                end="2027Q4",
                as_of_offsets_days=(0,),
                n_draws=50,
                benchmark=None,
            )
        counts = result.diagnostics["n_rows_by_model"]
        assert set(counts) == {"as_class", "as_factory", "as_instance"}
        assert min(counts.values()) > 0
        # All three are the same model, so the nowcasts must coincide.
        points = result.rows.pivot_table(
            index=["quarter", "horizon"], columns="model", values="point"
        )
        np.testing.assert_allclose(
            points["as_class"].to_numpy(), points["as_instance"].to_numpy()
        )

    def test_a_model_that_never_fits_is_not_silently_dropped(self, linear_data):
        _, data = linear_data

        class _Broken(NaiveCarryForward):
            def fit(self, info):
                raise RuntimeError("deliberately broken")

        with pytest.warns(EvaluationWarning, match="produced no evaluation row"):
            result = rolling_origin_evaluation(
                models={"broken": _Broken, "naive": NaiveCarryForward},
                data=data,
                start="2026Q1",
                end="2027Q4",
                as_of_offsets_days=(0,),
                n_draws=50,
                benchmark=None,
            )
        assert result.diagnostics["models_with_no_rows"] == ["broken"]
        assert result.diagnostics["n_rows_by_model"]["naive"] > 0

    def test_strict_policy_propagates_a_model_failure(self, linear_data):
        _, data = linear_data

        class _Broken(NaiveCarryForward):
            def fit(self, info):
                raise RuntimeError("deliberately broken")

        with pytest.raises(RuntimeError, match="deliberately broken"):
            rolling_origin_evaluation(
                models={"broken": _Broken},
                data=data,
                start="2026Q1",
                end="2027Q4",
                as_of_offsets_days=(0,),
                n_draws=50,
                fallback_policy=FallbackPolicy.STRICT,
            )

    def test_multi_date_grid_does_not_inflate_the_metrics(self, linear_data):
        """Several nowcasts of one quarter are one observation, not several."""
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={"naive": NaiveCarryForward},
                data=data,
                start="2026Q1",
                end="2028Q4",
                as_of_offsets_days=(0, 30, 60),
                n_draws=50,
                benchmark=None,
            )
        assert sorted(result.rows["offset_days"].unique()) == [0, 30, 60]
        primary = result.rows.loc[result.rows["offset_days"] == 0]
        assert len(result.rows) > len(primary)
        metrics_n = int(
            result.metrics.loc[
                (result.metrics["subperiod"] == "full")
                & (result.metrics["horizon"] == 1),
                "n",
            ].iloc[0]
        )
        assert metrics_n == int((primary["horizon"] == 1).sum())
        assert result.diagnostics["primary_offset_days"] == 0

    def test_horizon_two_is_only_reachable_at_offset_zero(self, linear_data):
        """A ~100-day publication lag leaves a ~9-day window with two unpublished."""
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={"naive": NaiveCarryForward},
                data=data,
                start="2026Q1",
                end="2028Q4",
                as_of_offsets_days=(0, 30),
                n_draws=50,
                benchmark=None,
            )
        by_offset = result.rows.groupby("offset_days")["horizon"].agg(set).to_dict()
        assert by_offset[0] == {1, 2}
        assert by_offset[30] == {1}

    def test_evolution_tracks_a_quarter_across_horizons(self, linear_data):
        """Evolution is event-driven: it moves when the previous quarter publishes."""
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={
                    "smoothing_regression": lambda: SmoothingRegressionNowcaster(
                        n_lags=1
                    )
                },
                data=data,
                start="2026Q1",
                end="2028Q4",
                as_of_offsets_days=(0, 30),
                n_draws=50,
                benchmark=None,
            )
        counts = result.rows.groupby("quarter")["as_of"].nunique()
        multi = counts.loc[counts > 1]
        assert not multi.empty
        block = result.evolution(multi.index[0])
        assert set(block["horizon"]) == {1, 2}
        assert block["as_of"].is_monotonic_increasing
        # The h=1 nowcast conditions on a published value, the h=2 one on a
        # nowcast of it, so they must differ.
        assert block["point"].nunique() > 1

    def test_min_train_quarters_skips_early_dates(self, linear_data):
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={"naive": NaiveCarryForward},
                data=data,
                start="2002Q1",
                end="2012Q4",
                as_of_offsets_days=(0,),
                n_draws=50,
                min_train_quarters=40,
                benchmark=None,
            )
        assert result.rows["n_train"].min() >= 39
        reasons = [s["reason"] for s in result.diagnostics["skipped"]]
        assert any("published quarters" in r for r in reasons)

    def test_subperiods_present_and_empty_ones_omitted(self, report):
        subperiods = set(report.metrics["subperiod"])
        assert "full" in subperiods
        assert {"covid", "2022"} <= subperiods
        # The evaluation window starts in 2016, so the GFC window has no scored
        # rows and is omitted rather than reported with n=0.
        assert "gfc" not in subperiods
        assert set(report.diagnostics["subperiods"]) == set(DEFAULT_SUBPERIODS)

    def test_custom_subperiods(self, linear_data):
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={"naive": NaiveCarryForward},
                data=data,
                start="2026Q1",
                end="2028Q4",
                as_of_offsets_days=(0,),
                n_draws=50,
                subperiods={"window": ("2027Q1", "2027Q4")},
                benchmark=None,
            )
        assert set(result.metrics["subperiod"]) == {"full", "window"}
        window = result.metrics.loc[result.metrics["subperiod"] == "window"]
        assert int(window["n"].max()) <= 4

    def test_reproducible_given_the_same_seed(self, linear_data):
        _, data = linear_data
        kwargs = dict(
            models={"naive": NaiveCarryForward},
            data=data,
            start="2026Q1",
            end="2027Q4",
            as_of_offsets_days=(0,),
            n_draws=200,
            benchmark=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = rolling_origin_evaluation(rng=7, **kwargs)
            second = rolling_origin_evaluation(rng=7, **kwargs)
            third = rolling_origin_evaluation(rng=8, **kwargs)
        pd.testing.assert_frame_equal(first.rows, second.rows)
        assert not np.allclose(
            first.rows["lower_80"].to_numpy(), third.rows["lower_80"].to_numpy()
        )

    def test_naive_abstains_from_a_directional_call(self, report):
        """A carry-forward predicts no change, so it makes no call at all."""
        naive = report.full_sample().loc[report.full_sample()["model"] == "naive"]
        assert (naive["directional_coverage"] == 0.0).all()
        assert naive["directional_hit_rate"].isna().all()
        benchmark = report.full_sample().loc[
            report.full_sample()["model"] == "smoothing_regression"
        ]
        assert (benchmark["directional_coverage"] > 0.9).all()

    def test_coverage_is_near_nominal_for_the_benchmark(self, report):
        """Cross-checks the harness against spec test 7's independent measurement."""
        block = report.full_sample()
        benchmark = block.loc[block["model"] == "smoothing_regression"]
        for _, row in benchmark.iterrows():
            assert abs(row["coverage_80"] - 0.80) < 0.12, row.to_dict()
            assert abs(row["coverage_95"] - 0.95) < 0.10, row.to_dict()

    def test_input_validation(self, linear_data):
        _, data = linear_data
        base = dict(data=data, start="2026Q1", end="2027Q4", n_draws=20)
        with pytest.raises(ValueError, match="contiguous from 1"):
            rolling_origin_evaluation(
                models={"n": NaiveCarryForward}, horizons=(2,), **base
            )
        with pytest.raises(ValueError, match="`models` is empty"):
            rolling_origin_evaluation(models={}, **base)
        with pytest.raises(ValueError, match="as_of_offsets_days must be >= 0"):
            rolling_origin_evaluation(
                models={"n": NaiveCarryForward}, as_of_offsets_days=(-1,), **base
            )
        with pytest.raises(ValueError, match="primary_offset_days=99"):
            rolling_origin_evaluation(
                models={"n": NaiveCarryForward},
                as_of_offsets_days=(0,),
                primary_offset_days=99,
                **base,
            )
        with pytest.raises(ValueError, match="must not exceed"):
            rolling_origin_evaluation(
                models={"n": NaiveCarryForward},
                data=data,
                start="2027Q1",
                end="2026Q1",
                n_draws=20,
            )

    def test_no_usable_date_raises_with_the_skip_reasons(self, linear_data):
        _, data = linear_data
        with pytest.raises(ValueError, match="no evaluation row was produced"):
            rolling_origin_evaluation(
                models={"naive": NaiveCarryForward},
                data=data,
                start="1990Q1",
                end="1991Q4",
                as_of_offsets_days=(0,),
                n_draws=20,
            )


class TestDieboldMarianoReporting:
    def test_naive_is_reported_as_worse_than_the_benchmark(self, report):
        dm = report.dm_tests.loc[
            (report.dm_tests["subperiod"] == "full")
            & (report.dm_tests["model"] == "naive")
        ]
        assert not dm.empty
        assert (dm["rmse_ratio"] > 1.0).all()
        assert (dm["verdict"] == "worse than smoothing_regression").all()
        assert (dm["p_value"] < 0.05).all()

    def test_benchmark_is_not_tested_against_itself(self, report):
        assert "smoothing_regression" not in set(report.dm_tests["model"])

    def test_tests_are_paired_on_the_same_quarters(self, report):
        """An unpaired comparison would be meaningless; ``n`` proves the pairing."""
        full = report.dm_tests.loc[report.dm_tests["subperiod"] == "full"]
        assert not full.empty
        for _, row in full.iterrows():
            block = report.rows.loc[
                (report.rows["horizon"] == row["horizon"])
                & (report.rows["offset_days"] == 0)
            ]
            shared = set(
                block.loc[block["model"] == row["model"], "quarter"]
            ) & set(block.loc[block["model"] == row["benchmark"], "quarter"])
            assert row["n"] == len(shared)

    def test_benchmark_none_skips_the_tests(self, linear_data):
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={"naive": NaiveCarryForward},
                data=data,
                start="2026Q1",
                end="2027Q4",
                as_of_offsets_days=(0,),
                n_draws=50,
                benchmark=None,
            )
        assert result.dm_tests.empty

    def test_absent_benchmark_name_skips_the_tests(self, linear_data):
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={"naive": NaiveCarryForward},
                data=data,
                start="2026Q1",
                end="2027Q4",
                as_of_offsets_days=(0,),
                n_draws=50,
            )
        assert result.dm_tests.empty


class TestSummary:
    def test_reports_no_winner_plainly(self, report):
        text = report.summary()
        assert "No model beats smoothing_regression" in text
        assert "substantive result rather than a failed run" in text
        assert "reduced form of the library's own AR(1)/Rudin models" in text

    def test_includes_provenance(self, report):
        text = report.summary()
        assert "data mode            : vintage" in text
        assert "evaluation_target='first_print'" in text
        assert "primary offset" in text

    def test_missing_subperiod_is_stated_not_crashed(self, report):
        assert "No evaluation rows in subperiod 'gfc'" in report.summary(
            subperiod="gfc"
        )

    @pytest.mark.slow
    def test_reports_a_winner_when_there_is_one(self, path_data):
        """The mirror of ``test_reports_no_winner_plainly``, on the path DGP."""
        _, data = path_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={
                    "smoothing_regression": lambda: SmoothingRegressionNowcaster(
                        n_lags=1
                    ),
                    "signature": lambda: SignatureNowcaster(level=2, alpha=1e-3),
                },
                data=data,
                start="2020Q1",
                end="2032Q4",
                as_of_offsets_days=(0,),
                n_draws=400,
            )
        text = result.summary()
        assert "Beats smoothing_regression at the 5 % level" in text
        assert "not corrected for testing" in text, (
            "a multiple-comparison caveat must travel with a positive claim"
        )
        dm = result.dm_tests.loc[result.dm_tests["subperiod"] == "full"]
        assert (dm["rmse_ratio"] < 0.80).all()


# ──────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────


class TestPlots:
    def test_nowcast_vs_realised(self, report):
        figure = plot_nowcast_vs_realised(report, horizon=1)
        assert figure.axes
        axis = figure.axes[0]
        assert "smoothing_regression" in axis.get_title()
        labels = [line.get_label() for line in axis.get_lines()]
        assert any("nowcast" in str(label) for label in labels)
        assert any("realised" in str(label) for label in labels)
        figure.clf()

    def test_nowcast_vs_realised_rejects_an_absent_series(self, report):
        with pytest.raises(ValueError, match="no rows for model"):
            plot_nowcast_vs_realised(report, model="nope", horizon=1)

    def test_error_by_horizon(self, report):
        figure = plot_error_by_horizon(report)
        axis = figure.axes[0]
        assert axis.get_ylabel() == "RMSE"
        assert [t.get_text() for t in axis.get_xticklabels()] == ["h=1", "h=2"]
        figure.clf()

    def test_error_by_horizon_rejects_an_empty_subperiod(self, report):
        with pytest.raises(ValueError, match="no metrics for subperiod"):
            plot_error_by_horizon(report, subperiod="gfc")

    def test_evolution_needs_more_than_one_offset(self, report):
        """``report`` uses the default two-offset grid, so this one works ..."""
        figure = plot_nowcast_evolution(report, max_panels=2)
        assert len(figure.axes) >= 2
        figure.clf()

    def test_evolution_refuses_a_single_offset_run(self, linear_data):
        """... and a single-offset run has nothing to show, so it says so."""
        _, data = linear_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_origin_evaluation(
                models={"naive": NaiveCarryForward},
                data=data,
                start="2026Q1",
                end="2027Q4",
                as_of_offsets_days=(0,),
                n_draws=50,
                benchmark=None,
            )
        with pytest.raises(ValueError, match="needs more than one as_of offset"):
            plot_nowcast_evolution(result)
