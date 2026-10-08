"""
Tests for ``nowcast/uncertainty.py`` — residual pools and recursive simulation.

Covers module-spec tests 7 (interval coverage within ±5pp of nominal on the
linear DGP, for h=1 and h=2) and 8 (h=2 intervals wider than h=1), plus the
mechanics everything else rests on: that the rolling-origin backtest is really
out of sample, that the <20-error fallback engages and is flagged, and that the
recursion reproduces the analytic variance of the AR(1) reduced form.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.nowcast import build_information_set
from private_assets_frequency.nowcast.models import (
    NaiveCarryForward,
    SmoothingRegressionNowcaster,
)
from private_assets_frequency.nowcast.uncertainty import (
    DEFAULT_MIN_OOS,
    ResidualPool,
    ResidualPoolWarning,
    build_residual_pool,
    empirical_interval,
    point_chain,
    recursive_predictive_draws,
    rolling_origin_backtest,
)

from .conftest import make_linear_smoothing_dataset

# ──────────────────────────────────────────────────────────────────
# ResidualPool
# ──────────────────────────────────────────────────────────────────


def _pool(errors: dict[int, np.ndarray]) -> ResidualPool:
    return ResidualPool(
        errors=errors,
        source={h: "oos" for h in errors},
        n_oos={h: int(v.size) for h, v in errors.items()},
        inflation=1.0,
        min_oos=DEFAULT_MIN_OOS,
    )


class TestResidualPool:
    def test_sample_draws_only_from_the_pool(self):
        pool = _pool({1: np.array([-1.0, 0.0, 2.0])})
        draws = pool.sample(1, 500, np.random.default_rng(0))
        assert draws.shape == (500,)
        assert set(np.unique(draws)).issubset({-1.0, 0.0, 2.0})

    def test_sample_is_reproducible(self):
        pool = _pool({1: np.arange(30.0)})
        a = pool.sample(1, 100, np.random.default_rng(42))
        b = pool.sample(1, 100, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_missing_horizon_raises_rather_than_substituting(self):
        """No silent substitution of one horizon's errors for another's."""
        pool = _pool({1: np.arange(30.0)})
        assert pool.has(1) and not pool.has(2)
        with pytest.raises(KeyError, match="no residual pool at horizon 2"):
            pool.sample(2, 10, np.random.default_rng(0))

    def test_sampler_ignores_the_step_horizon(self):
        pool = _pool({1: np.arange(30.0), 2: np.arange(100.0, 130.0)})
        sampler = pool.sampler(1)
        draws = sampler(2, 200, np.random.default_rng(0))
        assert draws.max() < 100.0, "sampler(1) must keep drawing from pool 1"

    def test_scale_and_fallback_flags(self):
        pool = _pool({1: np.array([-1.0, 1.0])})
        assert pool.scale(1) == pytest.approx(np.sqrt(2.0))
        assert not pool.is_fallback(1)
        assert pool.method_suffix([1]) == ""


class TestBuildResidualPool:
    def test_uses_oos_when_there_are_enough(self):
        oos = {1: np.full(30, 0.1), 2: np.full(25, 0.2)}
        pool = build_residual_pool(
            oos,
            np.full(50, 0.01),
            n_params=3,
            horizons=(1, 2),
            min_oos=20,
        )
        assert pool.source == {1: "oos", 2: "oos"}
        assert pool.n_oos == {1: 30, 2: 25}
        np.testing.assert_array_equal(pool.errors[1], oos[1])

    def test_falls_back_below_min_oos_and_warns(self):
        """Spec: fewer than 20 OOS errors → inflated in-sample residuals, flagged."""
        oos = {1: np.full(30, 0.1), 2: np.full(19, 0.2)}
        in_sample = np.full(40, 0.01)
        with pytest.warns(ResidualPoolWarning, match="fewer than 20"):
            pool = build_residual_pool(
                oos, in_sample, n_params=4, horizons=(1, 2), min_oos=20
            )
        assert pool.source == {1: "oos", 2: "in_sample_inflated"}
        assert pool.n_oos[2] == 19, "the honest count survives the fallback"
        expected_inflation = np.sqrt(1.0 + 4 / 40)
        assert pool.inflation == pytest.approx(expected_inflation)
        np.testing.assert_allclose(pool.errors[2], in_sample * expected_inflation)
        assert pool.is_fallback(2) and not pool.is_fallback(1)
        assert pool.method_suffix([1, 2]) == "+in_sample_inflated"
        assert pool.method_suffix([1]) == ""

    def test_fallback_without_in_sample_residuals_raises(self):
        with pytest.raises(ValueError, match="no basis for a prediction interval"):
            build_residual_pool(
                {1: np.empty(0)},
                np.empty(0),
                n_params=3,
                horizons=(1,),
                min_oos=20,
            )

    def test_non_finite_errors_are_dropped(self):
        oos = {1: np.concatenate([np.full(25, 0.1), [np.nan, np.inf]])}
        pool = build_residual_pool(
            oos, np.full(30, 0.01), n_params=2, horizons=(1,), min_oos=20
        )
        assert pool.n_oos[1] == 25
        assert np.all(np.isfinite(pool.errors[1]))

    def test_warn_false_suppresses_the_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_residual_pool(
                {1: np.full(5, 0.1)},
                np.full(30, 0.01),
                n_params=2,
                horizons=(1,),
                min_oos=20,
                warn=False,
            )


# ──────────────────────────────────────────────────────────────────
# Rolling-origin backtest
# ──────────────────────────────────────────────────────────────────


class TestRollingOriginBacktest:
    def test_errors_are_actual_minus_prediction_at_the_right_rows(self):
        y = np.arange(10.0)

        def predict_fn(n_train, hs):
            # Deliberately predict 0 so the error equals the actual.
            return {h: 0.0 for h in hs}

        errors, diag = rolling_origin_backtest(
            y, min_train=5, horizons=(1, 2), predict_fn=predict_fn
        )
        np.testing.assert_array_equal(errors[1], np.arange(5.0, 10.0))
        np.testing.assert_array_equal(errors[2], np.arange(6.0, 10.0))
        assert diag["n_errors_by_horizon"] == {1: 5, 2: 4}

    def test_predict_fn_only_ever_sees_training_rows(self):
        """The backtest is the guard against the model peeking forward."""
        y = np.arange(20.0)
        seen: list[int] = []

        def predict_fn(n_train, hs):
            seen.append(n_train)
            return {h: float(y[n_train - 1]) for h in hs}

        errors, _ = rolling_origin_backtest(
            y, min_train=10, horizons=(1,), predict_fn=predict_fn
        )
        assert seen == list(range(10, 20))
        # Carry-forward on a unit ramp has error exactly 1 at every origin.
        np.testing.assert_allclose(errors[1], 1.0)

    def test_max_origins_keeps_the_most_recent(self):
        y = np.arange(30.0)

        def predict_fn(n_train, hs):
            return {h: 0.0 for h in hs}

        errors, diag = rolling_origin_backtest(
            y, min_train=10, horizons=(1,), predict_fn=predict_fn, max_origins=5
        )
        assert errors[1].size == 5
        np.testing.assert_array_equal(errors[1], np.arange(25.0, 30.0))
        assert diag["first_origin"] == 25

    def test_unfittable_origin_is_skipped_not_fatal(self):
        y = np.arange(15.0)

        def predict_fn(n_train, hs):
            if n_train == 12:
                raise np.linalg.LinAlgError("singular")
            return {h: 0.0 for h in hs}

        errors, diag = rolling_origin_backtest(
            y, min_train=10, horizons=(1,), predict_fn=predict_fn
        )
        assert diag["n_origins_skipped"] == 1
        assert errors[1].size == 4

    def test_nan_prediction_is_declined(self):
        y = np.arange(15.0)

        def predict_fn(n_train, hs):
            return {h: (np.nan if n_train == 11 else 0.0) for h in hs}

        errors, diag = rolling_origin_backtest(
            y, min_train=10, horizons=(1,), predict_fn=predict_fn
        )
        assert diag["n_declined_by_horizon"][1] == 1
        assert errors[1].size == 4

    def test_rejects_bad_arguments(self):
        y = np.arange(10.0)
        with pytest.raises(ValueError, match="min_train must be >= 1"):
            rolling_origin_backtest(
                y, min_train=0, horizons=(1,), predict_fn=lambda n, h: {}
            )
        with pytest.raises(ValueError, match="horizons must all be >= 1"):
            rolling_origin_backtest(
                y, min_train=5, horizons=(0,), predict_fn=lambda n, h: {}
            )


# ──────────────────────────────────────────────────────────────────
# Recursion
# ──────────────────────────────────────────────────────────────────


class TestRecursion:
    @staticmethod
    def _ar1_mean(c: float):
        def mean_fn(horizon: int, history: np.ndarray) -> np.ndarray:
            return c * history[:, 0]

        return mean_fn

    def test_point_chain_matches_the_closed_form(self):
        c = 0.6
        points = point_chain(
            horizons=(1, 2, 3),
            conditional_mean=self._ar1_mean(c),
            history=np.array([1.0]),
        )
        assert points[1] == pytest.approx(c)
        assert points[2] == pytest.approx(c**2)
        assert points[3] == pytest.approx(c**3)

    def test_recursion_reproduces_the_analytic_variance(self):
        r"""For :math:`s_Q = c s_{Q-1} + e_Q`, Var(h=2) = σ²(1 + c²)."""
        c, sigma = 0.6, 0.02
        pool = _pool({1: np.random.default_rng(1).normal(0, sigma, 20_000)})
        draws = recursive_predictive_draws(
            horizons=(1, 2, 3),
            conditional_mean=self._ar1_mean(c),
            history=np.array([0.05]),
            innovation_sampler=pool.sampler(1),
            n_draws=200_000,
            rng=np.random.default_rng(2),
        )
        var1 = draws[1].var(ddof=1)
        var2 = draws[2].var(ddof=1)
        var3 = draws[3].var(ddof=1)
        assert var1 == pytest.approx(sigma**2, rel=0.03)
        assert var2 == pytest.approx(sigma**2 * (1 + c**2), rel=0.03)
        assert var3 == pytest.approx(
            sigma**2 * (1 + c**2 + c**4), rel=0.03
        )

    def test_later_horizons_are_wider(self):
        c, sigma = 0.6, 0.02
        pool = _pool({1: np.random.default_rng(1).normal(0, sigma, 5_000)})
        draws = recursive_predictive_draws(
            horizons=(1, 2, 3),
            conditional_mean=self._ar1_mean(c),
            history=np.array([0.05]),
            innovation_sampler=pool.sampler(1),
            n_draws=50_000,
            rng=np.random.default_rng(3),
        )
        widths = [
            empirical_interval(draws[h], 0.80)[1]
            - empirical_interval(draws[h], 0.80)[0]
            for h in (1, 2, 3)
        ]
        assert widths[0] < widths[1] < widths[2]

    def test_conditioning_on_draws_not_points(self):
        """A nonlinear mean exposes plug-in intervals; the recursion must not use them."""

        def mean_fn(horizon: int, history: np.ndarray) -> np.ndarray:
            return history[:, 0] ** 2

        pool = _pool({1: np.random.default_rng(4).normal(0, 1.0, 5_000)})
        draws = recursive_predictive_draws(
            horizons=(1, 2),
            conditional_mean=mean_fn,
            history=np.array([0.0]),
            innovation_sampler=pool.sampler(1),
            n_draws=50_000,
            rng=np.random.default_rng(5),
        )
        # Plugging in the point (0² = 0) would give Var(h=2) = 1; conditioning on
        # the draws gives Var(e₁²) + 1 ≈ 3.
        assert draws[2].var(ddof=1) > 2.0

    def test_history_depth_two_shifts_correctly(self):
        def mean_fn(horizon: int, history: np.ndarray) -> np.ndarray:
            # Reads only the second lag, so the answer pins down the shifting.
            return history[:, 1]

        points = point_chain(
            horizons=(1, 2, 3),
            conditional_mean=mean_fn,
            history=np.array([10.0, 20.0]),
        )
        assert points[1] == pytest.approx(20.0)  # s_{Q-2}
        assert points[2] == pytest.approx(10.0)  # shifted: old s_{Q-1}
        assert points[3] == pytest.approx(20.0)  # shifted: the h=1 point

    def test_non_contiguous_horizons_rejected(self):
        pool = _pool({1: np.arange(30.0)})
        with pytest.raises(ValueError, match="contiguous horizons"):
            recursive_predictive_draws(
                horizons=(1, 3),
                conditional_mean=self._ar1_mean(0.5),
                history=np.array([1.0]),
                innovation_sampler=pool.sampler(1),
                n_draws=10,
                rng=np.random.default_rng(0),
            )
        with pytest.raises(ValueError, match="contiguous horizons"):
            point_chain(
                horizons=(2,),
                conditional_mean=self._ar1_mean(0.5),
                history=np.array([1.0]),
            )

    def test_bad_shapes_rejected(self):
        pool = _pool({1: np.arange(30.0)})
        with pytest.raises(ValueError, match="n_draws must be >= 1"):
            recursive_predictive_draws(
                horizons=(1,),
                conditional_mean=self._ar1_mean(0.5),
                history=np.array([1.0]),
                innovation_sampler=pool.sampler(1),
                n_draws=0,
                rng=np.random.default_rng(0),
            )
        with pytest.raises(ValueError, match="returned 1 values, expected 10"):
            recursive_predictive_draws(
                horizons=(1,),
                conditional_mean=lambda h, hist: np.array([0.0]),
                history=np.array([1.0]),
                innovation_sampler=pool.sampler(1),
                n_draws=10,
                rng=np.random.default_rng(0),
            )


class TestEmpiricalInterval:
    def test_quantiles_are_equal_tailed(self):
        draws = np.arange(1001.0)
        lo, hi = empirical_interval(draws, 0.80)
        assert lo == pytest.approx(100.0)
        assert hi == pytest.approx(900.0)

    def test_bad_level_rejected(self):
        with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
            empirical_interval(np.arange(10.0), 1.0)


# ──────────────────────────────────────────────────────────────────
# Spec tests 7 and 8 — coverage and horizon widening, end to end
# ──────────────────────────────────────────────────────────────────


def _rolling_coverage(
    dataset,
    model_factory,
    *,
    first_origin: int = 65,
    n_draws: int = 1500,
    seed: int = 11,
) -> dict[str, dict[int, float]]:
    r"""Replay the nowcast quarter by quarter and tally realised coverage.

    At each origin ``o`` the information set is built at
    ``as_of = quarters[o+2].end_time`` — the instant that quarter closes, not
    midnight on its last day, which would leave it still open. Under the 100-day
    publication lag that
    makes quarter ``o`` the last published one, so ``o+1`` sits at horizon 1 and
    ``o+2`` at horizon 2 — the module's motivating configuration, replayed
    across the sample.

    ``first_origin`` is 65 rather than the model's own ``min_train`` of 40 so
    that the internal rolling-origin pool has comfortably more than the 20
    errors it needs; below that the pool falls back to inflated in-sample
    residuals and the run would be measuring the fallback instead.

    Everything is re-fitted at every origin: coefficients, hyperparameters and
    the residual pool. That is the expensive-by-design behaviour described in
    ``DESIGN_NOTES.md``; reusing one global fit here would make the coverage
    numbers meaningless.

    Returns
    -------
    dict
        ``{'coverage_80': {h: frac}, 'coverage_95': {h: frac}, 'n': {h: count},
        'width_80': {h: mean width}, 'rmse': {h: rmse}}``.
    """
    quarters = pd.PeriodIndex(dataset.reported.index, freq="Q")
    realised = dataset.first_print
    rng = np.random.default_rng(seed)

    hits80: dict[int, list[bool]] = {1: [], 2: []}
    hits95: dict[int, list[bool]] = {1: [], 2: []}
    widths: dict[int, list[float]] = {1: [], 2: []}
    errors: dict[int, list[float]] = {1: [], 2: []}

    for o in range(first_origin, len(quarters) - 2):
        as_of = quarters[o + 2].end_time
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = build_information_set(
                dataset.reported,
                dataset.daily_factors,
                as_of,
                vintages=dataset.vintages,
            )
            if info.last_published_quarter != quarters[o]:
                continue
            targets = info.unpublished_quarters()
            if len(targets) < 2:
                continue
            model = model_factory().fit(info)
            results = model.predict(info, targets, n_draws=n_draws, rng=rng)
        for res in results:
            if res.horizon not in hits80 or res.quarter not in realised.index:
                continue
            actual = float(realised[res.quarter])
            hits80[res.horizon].append(res.covers(actual, 0.80))
            hits95[res.horizon].append(res.covers(actual, 0.95))
            widths[res.horizon].append(res.interval_80[1] - res.interval_80[0])
            errors[res.horizon].append(actual - res.point)

    return {
        "coverage_80": {h: float(np.mean(v)) for h, v in hits80.items() if v},
        "coverage_95": {h: float(np.mean(v)) for h, v in hits95.items() if v},
        "n": {h: len(v) for h, v in hits80.items()},
        "width_80": {h: float(np.mean(v)) for h, v in widths.items() if v},
        "rmse": {
            h: float(np.sqrt(np.mean(np.square(v)))) for h, v in errors.items() if v
        },
    }


@pytest.fixture(scope="module")
def coverage_dataset():
    """Linear DGP, no revisions, long enough for a meaningful rolling run.

    Two deliberate choices:

    ``revision_sd=0`` isolates what spec test 7 is about — whether the interval
    machinery is calibrated — from the separate question of how much a revision
    process widens realised errors. The revision case is tested on its own in
    :meth:`TestCoverage.test_coverage_degrades_with_revisions`.

    ``n_quarters=200`` rather than ~100, because the spec's ±5pp tolerance is
    about *one* standard error of a coverage estimate at 73 origins
    (``sqrt(.8*.2/73) = 4.7pp``). At 133 origins the standard error falls to
    3.5pp, so the test measures calibration rather than the luck of a seed — it
    passes with the same margin on seeds 777 and 999 as on the pinned one.
    """
    return make_linear_smoothing_dataset(
        n_quarters=200, revision_sd=0.0, seed=4242
    )


@pytest.mark.slow
class TestCoverage:
    """Spec test 7: realised coverage within ±5pp of nominal, for h=1 and h=2."""

    def test_smoothing_regression_is_calibrated(self, coverage_dataset):
        stats = _rolling_coverage(
            coverage_dataset,
            lambda: SmoothingRegressionNowcaster(n_lags=1),
        )
        assert stats["n"][1] >= 100 and stats["n"][2] >= 100, stats["n"]
        for horizon in (1, 2):
            assert abs(stats["coverage_80"][horizon] - 0.80) <= 0.05, (
                f"h={horizon} 80% coverage {stats['coverage_80'][horizon]:.3f} "
                f"(n={stats['n'][horizon]})"
            )
            assert abs(stats["coverage_95"][horizon] - 0.95) <= 0.05, (
                f"h={horizon} 95% coverage {stats['coverage_95'][horizon]:.3f} "
                f"(n={stats['n'][horizon]})"
            )

    def test_horizon_two_is_wider_and_less_accurate(self, coverage_dataset):
        """Spec test 8, over a whole rolling run rather than one date."""
        stats = _rolling_coverage(
            coverage_dataset,
            lambda: SmoothingRegressionNowcaster(n_lags=1),
        )
        assert stats["width_80"][2] > stats["width_80"][1]
        assert stats["rmse"][2] > stats["rmse"][1]

    def test_naive_intervals_widen_but_undercover(self, coverage_dataset):
        r"""The baseline's intervals widen with horizon but are not calibrated.

        This is a property of the model, not a defect in the interval
        machinery, and it is worth pinning down rather than papering over. The
        ``empirical_direct`` pool is the *unconditional* distribution of
        :math:`s_{Q+h} - s_Q` over history, so the intervals have roughly the
        right unconditional width. But a carry-forward's error is conditionally
        biased on a stationary series —
        :math:`\mathbb{E}[s_{Q+1} - s_Q \mid s_Q] = -(1-c)(s_Q - \mu)` — so
        when the last print sits far from the mean the error is systematically
        large and one-sided. Pooled intervals cannot absorb that, and realised
        coverage lands below nominal, more so at h=2.

        The assertion is therefore a wide documented band, not ±5pp. Spec test 7
        is scoped to the correctly specified model.
        """
        stats = _rolling_coverage(coverage_dataset, NaiveCarryForward)
        for horizon in (1, 2):
            assert 0.60 <= stats["coverage_80"][horizon] <= 0.92, stats
        assert stats["coverage_80"][2] <= stats["coverage_80"][1], (
            "conditional bias should bite harder at the longer horizon: "
            f"{stats['coverage_80']}"
        )
        assert stats["width_80"][2] > stats["width_80"][1]

    def test_smoothing_regression_beats_naive(self, coverage_dataset):
        """Sanity: on its own reduced form, the benchmark should dominate."""
        sr = _rolling_coverage(
            coverage_dataset, lambda: SmoothingRegressionNowcaster(n_lags=1)
        )
        naive = _rolling_coverage(coverage_dataset, NaiveCarryForward)
        assert sr["rmse"][1] < naive["rmse"][1]
        assert sr["rmse"][2] < naive["rmse"][2]

    def test_coverage_degrades_with_revisions(self):
        """Scoring against the first print when training on revisions costs coverage.

        This is the Risk 2 mechanism from ``DESIGN_NOTES.md``, measured: the
        model's residual pool reflects errors against the vintage in force,
        while scoring is against the first print, which carries extra revision
        noise the model never sees. Coverage should fall *below* nominal — never
        above, which would mean the intervals were loose for an unrelated
        reason.
        """
        noisy = make_linear_smoothing_dataset(
            n_quarters=140, revision_sd=0.010, seed=4242
        )
        stats = _rolling_coverage(
            noisy, lambda: SmoothingRegressionNowcaster(n_lags=1)
        )
        assert stats["coverage_80"][1] < 0.80
        assert stats["coverage_80"][1] > 0.45, (
            "a 1pp revision sd should dent coverage, not destroy it: "
            f"{stats['coverage_80']}"
        )
