"""
Tests for ``nowcast/models/`` — Models 0, 1, 2 and 3.

Covers module-spec test 1 at the level that matters most (perturbing
post-``as_of`` data leaves predictions **bit-for-bit** unchanged) and both halves
of spec test 5: the reduced form recovers ``b ≈ (1−λ)β`` and ``c ≈ λ`` on the
linear DGP, and ``StructuralSmoothingNowcaster`` matches it when given the true
parameters. Also covers spec test 4 (the signature model at ``level=1`` nests
the smoothing regression — exactly, not to a tolerance), test 6 (level 2 beats the
benchmark on the path-dependent DGP and does not on the linear one) and test 9
(the feature budget is refused under ``strict`` and skipped under ``warn``), and
pins the λ-convention arithmetic of CHECKPOINT 2 against simulation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import (
    BetaDist,
    FallbackPolicy,
    InverseGammaPrior,
    NormalPrior,
)
from private_assets_frequency.core.protocols import DesmoothedResult
from private_assets_frequency.desmoothing.ar1_bayesian import AR1BayesianSmoother
from private_assets_frequency.desmoothing.rudin_reparam import RudinReparamSmoothing
from private_assets_frequency.nowcast import build_information_set
from private_assets_frequency.nowcast.base import (
    Nowcaster,
    NowcastResult,
    make_nowcast_result,
)
from private_assets_frequency.nowcast.information_set import InformationSet
from private_assets_frequency.nowcast.models import (
    NaiveCarryForward,
    SignatureNowcaster,
    SmoothingRegressionNowcaster,
    StructuralSmoothingNowcaster,
)
from private_assets_frequency.nowcast.models._base import UnsupportedTargetError
from private_assets_frequency.nowcast.models.signature import (
    FeatureBudgetError,
    SignatureNowcasterWarning,
    _fit_selective_ridge,
    feature_budget,
)
from private_assets_frequency.nowcast.models.smoothing_regression import (
    SmoothingRegressionWarning,
    unwind_to_structural,
)
from private_assets_frequency.nowcast.models.structural import (
    StructuralSmoothingWarning,
    UnsupportedSmoothingModelError,
    convert_lambda_to_quarterly,
    infer_fitted_convention,
    rolling_annual_ar1_autocorrelation,
)
from private_assets_frequency.nowcast.signatures import MAX_SIGNATURE_LEVEL

from .conftest import (
    make_linear_smoothing_dataset,
    make_path_dependent_dataset,
)

AS_OF = pd.Timestamp("2026-10-07")


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _info(ds, **kwargs) -> InformationSet:
    params = dict(
        reported=ds.reported,
        public_factors=ds.daily_factors,
        as_of=AS_OF,
        vintages=ds.vintages,
    )
    params.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_information_set(**params)


def _fit_predict(model, info, quarters=None, *, seed: int = 7, n_draws: int = 1000):
    """Fit and predict with warnings silenced and a pinned generator."""
    quarters = quarters if quarters is not None else info.unpublished_quarters()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(info)
        return fitted, fitted.predict(
            info, quarters, n_draws=n_draws, rng=np.random.default_rng(seed)
        )


def _paired(a, b):
    """Pair up two result lists, asserting they are the same length.

    Stands in for ``zip(..., strict=True)``, which the available interpreter
    (3.9) does not accept — and is the stronger check anyway, since a silently
    truncated comparison would make a leakage test pass vacuously.
    """
    assert len(a) == len(b), f"result lists differ in length: {len(a)} vs {len(b)}"
    return [(a[i], b[i]) for i in range(len(a))]


def _models():
    """Every model under test, as fresh factories."""
    return [
        ("naive", NaiveCarryForward),
        ("smoothing_regression_q1", lambda: SmoothingRegressionNowcaster(n_lags=1)),
        ("smoothing_regression_sel", SmoothingRegressionNowcaster),
    ]


# ──────────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────────


class TestProtocol:
    @pytest.mark.parametrize("label,factory", _models(), ids=lambda v: getattr(v, "__name__", v))
    def test_satisfies_the_nowcaster_protocol(self, label, factory):
        assert isinstance(factory(), Nowcaster)

    def test_predict_before_fit_raises(self, motivating_dataset):
        info = _info(motivating_dataset)
        with pytest.raises(RuntimeError, match="call fit\\(\\) before predict"):
            NaiveCarryForward().predict(info, info.unpublished_quarters())

    def test_fit_rejects_raw_frames(self, motivating_dataset):
        with pytest.raises(TypeError, match="expects an InformationSet"):
            NaiveCarryForward().fit(motivating_dataset.reported)

    def test_predict_refuses_a_different_as_of(self, motivating_dataset):
        """Later data with earlier coefficients is a leak; it must not be allowed."""
        early = _info(motivating_dataset, as_of=pd.Timestamp("2025-10-07"))
        late = _info(motivating_dataset)
        model, _ = _fit_predict(NaiveCarryForward(), early)
        with pytest.raises(ValueError, match="fitted at as_of=.*but asked to predict"):
            model.predict(late, late.unpublished_quarters())

    def test_refit_does_not_inherit_previous_state(self, motivating_dataset):
        early = _info(motivating_dataset, as_of=pd.Timestamp("2024-10-07"))
        late = _info(motivating_dataset)
        model = SmoothingRegressionNowcaster(n_lags=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(early)
            early_coef = model.coefficients.copy()
            early_n = model.fit_diagnostics["n_train"]
            model.fit(late)
        assert model.fit_diagnostics["n_train"] > early_n
        assert not np.allclose(model.coefficients.to_numpy(), early_coef.to_numpy())
        # And the refitted model now predicts happily from the later info set.
        model.predict(late, late.unpublished_quarters(), rng=np.random.default_rng(0))

    @pytest.mark.parametrize("label,factory", _models(), ids=lambda v: getattr(v, "__name__", v))
    def test_results_are_complete_and_ordered_as_requested(
        self, label, factory, motivating_dataset
    ):
        info = _info(motivating_dataset)
        q2 = pd.Period("2026Q2", freq="Q")
        q3 = pd.Period("2026Q3", freq="Q")
        _, results = _fit_predict(factory(), info, [q3, q2])
        assert [r.quarter for r in results] == [q3, q2], "order must follow the request"
        assert [r.horizon for r in results] == [2, 1]
        for r in results:
            assert isinstance(r, NowcastResult)
            assert r.as_of == AS_OF
            assert r.target == "reported"
            assert r.n_draws == 1000
            assert r.interval_80[0] < r.point < r.interval_80[1]
            assert r.interval_95[0] <= r.interval_80[0]
            assert r.interval_95[1] >= r.interval_80[1]
            assert r.diagnostics["data_mode"] == "vintage"
            assert r.diagnostics["n_train"] > 0

    def test_single_horizon_request_does_not_need_the_other(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, results = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1),
            info,
            [pd.Period("2026Q3", freq="Q")],
        )
        assert len(results) == 1 and results[0].horizon == 2

    def test_empty_quarters_rejected(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(NaiveCarryForward(), info)
        with pytest.raises(ValueError, match="`quarters` is empty"):
            model.predict(info, [])

    def test_published_quarter_is_not_a_target(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(NaiveCarryForward(), info)
        with pytest.raises(ValueError, match="already published"):
            model.predict(info, [pd.Period("2026Q1", freq="Q")])

    @pytest.mark.parametrize("label,factory", _models(), ids=lambda v: getattr(v, "__name__", v))
    def test_reproducible_given_the_same_seed(
        self, label, factory, motivating_dataset
    ):
        info = _info(motivating_dataset)
        _, first = _fit_predict(factory(), info, seed=123)
        _, second = _fit_predict(factory(), info, seed=123)
        for a, b in _paired(first, second):
            np.testing.assert_array_equal(a.draws, b.draws)
            assert a.point == b.point
        _, third = _fit_predict(factory(), info, seed=124)
        assert not np.array_equal(first[0].draws, third[0].draws)

    def test_draws_are_read_only(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, results = _fit_predict(NaiveCarryForward(), info)
        with pytest.raises(ValueError):
            results[0].draws[0] = 999.0


# ──────────────────────────────────────────────────────────────────
# Spec test 1 — leakage, at the level of predictions
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("label,factory", _models(), ids=lambda v: getattr(v, "__name__", v))
class TestNoLeakage:
    """Perturbing post-``as_of`` data must leave predictions bit-for-bit equal.

    ``np.testing.assert_array_equal`` throughout, never ``allclose``: a single
    differing float means a future observation reached the model.
    """

    def test_future_public_factors(self, label, factory, motivating_dataset):
        base_info = _info(motivating_dataset)
        _, base = _fit_predict(factory(), base_info)

        perturbed = motivating_dataset.daily_factors.copy()
        future = perturbed.index > AS_OF
        assert future.sum() > 100
        perturbed.loc[future] += 0.25
        other_info = _info(motivating_dataset, public_factors=perturbed)
        _, other = _fit_predict(factory(), other_info)

        for a, b in _paired(base, other):
            assert a.point == b.point
            np.testing.assert_array_equal(a.draws, b.draws)
            assert a.interval_80 == b.interval_80
            assert a.interval_95 == b.interval_95

    def test_future_vintage_releases(self, label, factory, motivating_dataset):
        base_info = _info(motivating_dataset)
        _, base = _fit_predict(factory(), base_info)

        perturbed = motivating_dataset.vintages.copy()
        future = perturbed["release_date"] > AS_OF
        assert future.sum() > 5
        perturbed.loc[future, "value"] += 0.5
        other_info = _info(motivating_dataset, vintages=perturbed)
        _, other = _fit_predict(factory(), other_info)

        for a, b in _paired(base, other):
            assert a.point == b.point
            np.testing.assert_array_equal(a.draws, b.draws)

    def test_truncating_the_future_entirely(self, label, factory, motivating_dataset):
        base_info = _info(motivating_dataset)
        _, base = _fit_predict(factory(), base_info)

        truncated_factors = motivating_dataset.daily_factors.loc[
            motivating_dataset.daily_factors.index <= AS_OF
        ]
        visible_vintages = motivating_dataset.vintages.loc[
            motivating_dataset.vintages["release_date"] <= AS_OF
        ].reset_index(drop=True)
        other_info = _info(
            motivating_dataset,
            public_factors=truncated_factors,
            vintages=visible_vintages,
        )
        _, other = _fit_predict(factory(), other_info)

        for a, b in _paired(base, other):
            assert a.point == b.point
            np.testing.assert_array_equal(a.draws, b.draws)

    def test_unpublished_reported_values_in_lag_rule_mode(
        self, label, factory, motivating_dataset
    ):
        perturbed = motivating_dataset.reported.copy()
        for q in ("2026Q2", "2026Q3"):
            perturbed.loc[pd.Period(q, freq="Q")] += 1.0

        def build(reported):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return build_information_set(
                    reported, motivating_dataset.daily_factors, AS_OF
                )

        _, base = _fit_predict(factory(), build(motivating_dataset.reported))
        _, other = _fit_predict(factory(), build(perturbed))
        for a, b in _paired(base, other):
            assert a.point == b.point
            np.testing.assert_array_equal(a.draws, b.draws)


# ──────────────────────────────────────────────────────────────────
# Model 0 — NaiveCarryForward
# ──────────────────────────────────────────────────────────────────


class TestNaiveCarryForward:
    def test_point_is_the_last_published_value_at_every_horizon(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        last = float(info.reported.iloc[-1])
        _, results = _fit_predict(NaiveCarryForward(), info)
        for r in results:
            assert r.point == pytest.approx(last)

    def test_it_carries_the_vintage_in_force_not_the_revised_value(
        self, motivating_dataset
    ):
        """The baseline is real-time too, or it would be an unfair benchmark."""
        info = _info(motivating_dataset)
        q1 = pd.Period("2026Q1", freq="Q")
        _, results = _fit_predict(NaiveCarryForward(), info)
        assert results[0].point == pytest.approx(float(info.reported[q1]))
        assert results[0].point != pytest.approx(
            float(motivating_dataset.reported[q1])
        )

    def test_direct_pool_is_used_by_default(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, results = _fit_predict(NaiveCarryForward(), info)
        assert results[0].interval_method == "empirical_direct"
        assert model.residual_pool.source == {1: "oos", 2: "oos"}
        assert results[0].diagnostics["innovation_horizon"] == 1
        assert results[1].diagnostics["innovation_horizon"] == 2

    def test_recursive_method_also_available(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, results = _fit_predict(
            NaiveCarryForward(interval_method="empirical_recursive"), info
        )
        assert results[0].interval_method == "empirical_recursive"
        assert results[1].draw_std > results[0].draw_std

    def test_direct_method_needs_a_pool_at_every_horizon(self, motivating_dataset):
        info = _info(motivating_dataset)
        model = NaiveCarryForward(horizons=(1,))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(info)
            with pytest.raises(ValueError, match="needs a residual pool at horizon 2"):
                model.predict(info, info.unpublished_quarters())

    def test_zero_features_and_params(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(NaiveCarryForward(), info)
        diag = model.fit_diagnostics
        assert diag["n_params"] == 0
        assert diag["features"] == ["s_lag1"]
        assert diag["feature_budget"]["within_budget"]

    def test_does_not_support_target_true(self, motivating_dataset):
        info = _info(motivating_dataset)
        model = NaiveCarryForward()
        model.target = "true"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(info)
            with pytest.raises(UnsupportedTargetError, match="no factor decomposition"):
                model.predict(info, info.unpublished_quarters())

    def test_needs_at_least_two_published_quarters(self, motivating_dataset):
        short = motivating_dataset.vintages.loc[
            motivating_dataset.vintages["quarter"] <= pd.Period("2001Q1", freq="Q")
        ]
        info = build_information_set(
            reported=None,
            public_factors=motivating_dataset.daily_factors,
            as_of=AS_OF,
            vintages=short,
            vintage_backfill=False,
        )
        with pytest.raises(ValueError, match="at least 2 published quarters"):
            NaiveCarryForward().fit(info)


# ──────────────────────────────────────────────────────────────────
# Model 1 — Spec test 5 (reduced form)
# ──────────────────────────────────────────────────────────────────


class TestSmoothingRegressionRecovery:
    r"""Spec test 5: recover :math:`b \approx (1-\lambda)\beta` and :math:`c \approx \lambda`."""

    @pytest.mark.parametrize("lam", [0.3, 0.6, 0.85])
    def test_reduced_form_coefficients(self, lam, make_linear):
        ds = make_linear(n_quarters=160, lam=lam, seed=5150)
        info = _info(ds, as_of=ds.reported.index[-1].end_time + pd.Timedelta(days=200))
        model, _ = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        coef = model.coefficients
        target_b = ds.params["reduced_form_b"]["equity_market"]
        assert coef["equity_market"] == pytest.approx(target_b, abs=0.04), (
            f"b={coef['equity_market']:.4f} vs (1-λ)β={target_b:.4f}"
        )
        assert coef["s_lag1"] == pytest.approx(lam, abs=0.05), (
            f"c₁={coef['s_lag1']:.4f} vs λ={lam:.4f}"
        )

    @pytest.mark.parametrize("lam", [0.3, 0.6, 0.85])
    def test_unwound_structural_parameters(self, lam, make_linear):
        r"""Unwinding :math:`\beta = b/\theta_0` recovers the true loading."""
        ds = make_linear(n_quarters=160, lam=lam, seed=5150)
        info = _info(ds, as_of=ds.reported.index[-1].end_time + pd.Timedelta(days=200))
        model, _ = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        s = model.structural_params
        assert s["stable"]
        assert s["lambda_implied"] == pytest.approx(lam, abs=0.05)
        assert s["theta_0"] == pytest.approx(1.0 - lam, abs=0.05)
        assert s["beta"]["equity_market"] == pytest.approx(
            ds.params["beta"]["equity_market"], rel=0.15
        )
        assert s["alpha"] == pytest.approx(ds.params["alpha"], abs=0.01)

    def test_two_factor_recovery(self, linear_dataset_two_factor):
        ds = linear_dataset_two_factor
        info = _info(ds, as_of=ds.reported.index[-1].end_time + pd.Timedelta(days=200))
        model, _ = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        for col, beta in ds.params["beta"].items():
            assert model.structural_params["beta"][col] == pytest.approx(
                beta, abs=0.25
            ), f"{col}: {model.structural_params['beta'][col]:.3f} vs {beta}"

    def test_r_squared_is_high_on_its_own_reduced_form(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        assert model.fit_diagnostics["r_squared"] > 0.85

    def test_residual_sd_matches_the_dgp(self, motivating_dataset):
        """σ_e should recover (1−λ)σ_ε."""
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        expected = (1.0 - motivating_dataset.params["lam"]) * motivating_dataset.params[
            "sigma_eps"
        ]
        assert model.sigma == pytest.approx(expected, rel=0.25)


class TestSmoothingRegressionSpecification:
    def test_explicit_n_lags_skips_selection(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SmoothingRegressionNowcaster(n_lags=2), info)
        assert model.fit_diagnostics["n_lags"] == 2
        assert model.fit_diagnostics["selection"]["selected_by"] == "user"
        assert "s_lag2" in model.fit_diagnostics["features"]

    def test_selection_runs_when_n_lags_is_none(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SmoothingRegressionNowcaster(), info)
        selection = model.fit_diagnostics["selection"]
        assert selection["selected_by"] == "inner_rolling_origin"
        assert set(selection["scores"]) == {"q=1,alpha=0", "q=2,alpha=0"}
        assert model.fit_diagnostics["n_lags"] in (1, 2)

    def test_selection_picks_q1_on_an_ar1_dgp(self, motivating_dataset):
        """The DGP is AR(1); q=2 adds a parameter for nothing and should lose."""
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SmoothingRegressionNowcaster(), info)
        assert model.fit_diagnostics["n_lags"] == 1

    def test_inner_selection_cannot_see_the_outer_quarter(self, motivating_dataset):
        """Spec critical note 1 / DESIGN_NOTES Risk 3, asserted not assumed."""
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SmoothingRegressionNowcaster(), info)
        unpublished = {str(q) for q in info.unpublished_quarters()}
        assert unpublished == {"2026Q2", "2026Q3"}
        for detail in model.fit_diagnostics["selection"]["detail"].values():
            assert not unpublished.intersection(set(detail["selection_index"]))
        first, last = model.fit_diagnostics["training_quarters"]
        assert last == "2026Q1", "training must stop at the last published quarter"

    def test_ridge_selection_grid(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, ridge_alpha=None), info
        )
        selection = model.fit_diagnostics["selection"]
        assert len(selection["scores"]) == 4
        assert model.fit_diagnostics["ridge_alpha"] in (0.0, 1e-4, 1e-3, 1e-2)

    def test_ridge_shrinks_factor_coefficients_only(self, motivating_dataset):
        """The lag coefficient is the smoothing parameter; it must not be shrunk."""
        info = _info(motivating_dataset)
        plain, _ = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, ridge_alpha=0.0), info
        )
        ridged, _ = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, ridge_alpha=10.0), info
        )
        assert abs(ridged.coefficients["equity_market"]) < abs(
            plain.coefficients["equity_market"]
        )
        # With the factor term shrunk the lag coefficient absorbs signal rather
        # than being shrunk itself — it moves away from zero, not toward it.
        assert abs(ridged.coefficients["s_lag1"]) >= abs(
            plain.coefficients["s_lag1"]
        ) - 1e-9

    def test_lagged_factor_terms_are_off_by_default(self, motivating_dataset):
        info = _info(motivating_dataset)
        off, _ = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        assert not any("_lag1" in f for f in off.fit_diagnostics["features"] if f != "s_lag1")
        on, _ = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, include_lagged_factors=True), info
        )
        assert "equity_market_lag1" in on.fit_diagnostics["features"]
        assert on.fit_diagnostics["n_params"] == off.fit_diagnostics["n_params"] + 1

    def test_gaps_in_the_reported_series_drop_adjacent_rows(self, motivating_dataset):
        """Lags are formed on a contiguous grid, never by positional shift."""
        gapped = motivating_dataset.vintages.loc[
            ~motivating_dataset.vintages["quarter"].isin(
                [pd.Period("2010Q2", freq="Q")]
            )
        ].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = build_information_set(
                reported=None,
                public_factors=motivating_dataset.daily_factors,
                as_of=AS_OF,
                vintages=gapped,
            )
        model, _ = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        # 2010Q2 is absent as a row and 2010Q3 is dropped for want of its lag.
        assert info.n_published == 100
        assert model.fit_diagnostics["n_train"] == 98

    def test_design_too_small_raises(self, motivating_dataset):
        short = motivating_dataset.vintages.loc[
            motivating_dataset.vintages["quarter"] <= pd.Period("2001Q3", freq="Q")
        ]
        info = build_information_set(
            reported=None,
            public_factors=motivating_dataset.daily_factors,
            as_of=AS_OF,
            vintages=short,
            vintage_backfill=False,
        )
        with pytest.raises(ValueError, match="usable rows for .* parameters"):
            SmoothingRegressionNowcaster(n_lags=1).fit(info)

    def test_bad_interval_method_raises(self, motivating_dataset):
        info = _info(motivating_dataset)
        with pytest.raises(ValueError, match="unknown interval_method"):
            SmoothingRegressionNowcaster(interval_method="magic").fit(info)

    def test_negative_ridge_rejected(self, motivating_dataset):
        info = _info(motivating_dataset)
        with pytest.raises(ValueError, match="ridge_alpha must be >= 0"):
            SmoothingRegressionNowcaster(n_lags=1, ridge_alpha=-1.0).fit(info)


class TestSmoothingRegressionIntervals:
    def test_recursive_is_the_default_and_widens(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, results = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1), info, n_draws=20_000
        )
        assert results[0].interval_method == "empirical_recursive"
        assert results[1].draw_std > results[0].draw_std

    def test_recursive_widening_matches_the_analytic_factor(self, motivating_dataset):
        r"""Var(h=2)/Var(h=1) should be :math:`1 + c_1^2`."""
        info = _info(motivating_dataset)
        model, results = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1), info, n_draws=200_000
        )
        c1 = float(model.coefficients["s_lag1"])
        ratio = (results[1].draw_std / results[0].draw_std) ** 2
        assert ratio == pytest.approx(1.0 + c1**2, rel=0.05)

    def test_direct_cross_check_is_recorded(self, motivating_dataset):
        """A materially wider direct interval is a misspecification signal."""
        info = _info(motivating_dataset)
        _, results = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        for r in results:
            lo, hi = r.diagnostics["interval_80_direct"]
            assert lo < r.point < hi
        # On a correctly specified DGP the two should be comparable at h=1.
        direct = results[0].diagnostics["interval_80_direct"]
        recursive = results[0].interval_80
        direct_width = direct[1] - direct[0]
        recursive_width = recursive[1] - recursive[0]
        assert 0.7 < direct_width / recursive_width < 1.4

    def test_draw_mean_is_the_point_shifted_by_the_pool_mean(
        self, motivating_dataset
    ):
        """The innovation pool is not centred, and that is documented behaviour.

        Out-of-sample errors in a small sample have a non-zero mean, which is
        kept rather than subtracted (see
        :mod:`private_assets_frequency.nowcast.uncertainty`). So the predictive
        draws sit ``mean(pool)`` away from the raw point forecast. Pinning the
        identity here is what stops that gap from being mistaken for a bug — or
        from silently changing.
        """
        info = _info(motivating_dataset)
        model, results = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1), info, n_draws=200_000
        )
        pool_mean = float(np.mean(model.residual_pool.errors[1]))
        assert pool_mean != 0.0
        for r in results:
            assert r.diagnostics["pool_mean"] == pytest.approx(pool_mean)
            mc_tol = 4.0 * r.draw_std / np.sqrt(r.n_draws) + 1e-9
            # h=1 adds one innovation; h=2 adds a second, shifted by c₁.
            n_shifts = 1.0 if r.horizon == 1 else 1.0 + float(
                model.coefficients["s_lag1"]
            )
            assert r.draw_mean == pytest.approx(
                r.point + n_shifts * pool_mean, abs=mc_tol
            )

    def test_parametric_mode(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, results = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, interval_method="parametric"),
            info,
            n_draws=50_000,
        )
        assert results[0].interval_method == "parametric"
        # Gaussian with the regression's prediction variance: slightly wider
        # than sigma because of the leverage term, and close to it.
        assert model.sigma < results[0].draw_std < model.sigma * 1.15

    def test_thin_pool_falls_back_and_labels_itself(self, motivating_dataset):
        """Spec: <20 OOS errors → inflated in-sample residuals, flagged."""
        info = _info(motivating_dataset, as_of=pd.Timestamp("2014-01-15"))
        model = SmoothingRegressionNowcaster(n_lags=1, min_train=40)
        with pytest.warns(Warning):
            model.fit(info)
            results = model.predict(
                info, info.unpublished_quarters(), rng=np.random.default_rng(0)
            )
        assert model.residual_pool.n_oos[1] < 20
        assert model.residual_pool.is_fallback(1)
        assert results[0].interval_method.endswith("+in_sample_inflated")
        assert results[0].diagnostics["residual_source"] == "in_sample_inflated"


class TestSmoothingRegressionTargetTrue:
    r"""``target='true'`` is a wrapper over the factor decomposition, not a nowcast."""

    def test_point_is_alpha_plus_beta_f(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, results = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, target="true"), info
        )
        s = model.structural_params
        for r in results:
            expected = s["alpha"] + s["beta"]["equity_market"] * float(
                info.factor_row(r.quarter)["equity_market"]
            )
            assert r.point == pytest.approx(expected)
            assert r.target == "true"

    def test_interval_width_does_not_grow_with_horizon(self, motivating_dataset):
        """The idiosyncratic part is irreducible and horizon-independent."""
        info = _info(motivating_dataset)
        _, results = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, target="true"),
            info,
            n_draws=100_000,
        )
        assert results[1].draw_std == pytest.approx(results[0].draw_std, rel=0.03)

    def test_interval_is_much_wider_than_for_the_reported_target(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        _, reported = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1), info, n_draws=50_000
        )
        _, true = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, target="true"), info, n_draws=50_000
        )
        # σ_ε = σ_e/θ₀, so with θ₀ ≈ 0.4 the true-return interval is ~2.5× wider.
        assert true[0].draw_std > 1.8 * reported[0].draw_std

    def test_recovers_the_latent_true_return_on_average(self, motivating_dataset):
        """It should be unbiased for r_Q, just not precise."""
        info = _info(motivating_dataset)
        model, _ = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, target="true"), info
        )
        s = model.structural_params
        f = motivating_dataset.quarterly_factors["equity_market"]
        published = pd.PeriodIndex(info.reported.index, freq="Q")
        fitted = s["alpha"] + s["beta"]["equity_market"] * f.loc[published]
        actual = motivating_dataset.true_returns.loc[published]
        assert float((actual - fitted).mean()) == pytest.approx(0.0, abs=0.01)

    def test_diagnostics_carry_the_caveat(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, results = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1, target="true"), info
        )
        note = results[0].diagnostics["interpretation"]
        assert "not a nowcast in the same sense" in note
        assert "idiosyncratic" in note


# ──────────────────────────────────────────────────────────────────
# Coefficient unwinding
# ──────────────────────────────────────────────────────────────────


class TestUnwindToStructural:
    def test_ar1_case(self):
        out = unwind_to_structural(
            np.array([0.002, 0.44, 0.60]), ["const", "F", "s_lag1"], ["F"]
        )
        assert out["theta_0"] == pytest.approx(0.40)
        assert out["lambda_implied"] == pytest.approx(0.60)
        assert out["beta"]["F"] == pytest.approx(1.10)
        assert out["alpha"] == pytest.approx(0.005)
        assert out["stable"]

    def test_two_lag_case_uses_the_rudin_sum(self):
        out = unwind_to_structural(
            np.array([0.001, 0.30, 0.50, 0.20]),
            ["const", "F", "s_lag1", "s_lag2"],
            ["F"],
        )
        assert out["theta_0"] == pytest.approx(0.30)
        np.testing.assert_allclose(out["theta"], [0.30, 0.50, 0.20])
        assert out["beta"]["F"] == pytest.approx(1.0)

    def test_unstable_theta0_is_flagged_not_silently_divided(self):
        out = unwind_to_structural(
            np.array([0.001, 0.05, 0.95]), ["const", "F", "s_lag1"], ["F"]
        )
        assert out["theta_0"] == pytest.approx(0.05)
        assert not out["stable"]
        assert np.isnan(out["beta"]["F"])
        assert np.isnan(out["alpha"])

    def test_model_warns_and_refuses_target_true_when_unstable(self, make_linear):
        """A λ near 1 makes β = b/θ₀ meaningless; say so rather than returning it."""
        ds = make_linear(n_quarters=160, lam=0.95, sigma_eps=0.02, seed=31337)
        info = _info(ds, as_of=ds.reported.index[-1].end_time + pd.Timedelta(days=200))
        model = SmoothingRegressionNowcaster(n_lags=1, target="true")
        with pytest.warns(SmoothingRegressionWarning, match="stability floor"):
            model.fit(info)
        if not model.structural_params["stable"]:
            with pytest.raises(ValueError, match="stability floor"):
                model.predict(info, info.unpublished_quarters())

    def test_strict_policy_escalates_the_warning(self, make_linear):
        ds = make_linear(n_quarters=160, lam=0.95, sigma_eps=0.02, seed=31337)
        info = _info(ds, as_of=ds.reported.index[-1].end_time + pd.Timedelta(days=200))
        model = SmoothingRegressionNowcaster(
            n_lags=1, fallback_policy=FallbackPolicy.STRICT
        )
        with pytest.raises(SmoothingRegressionWarning):
            model.fit(info)


# ──────────────────────────────────────────────────────────────────
# Result construction
# ──────────────────────────────────────────────────────────────────


class TestMakeNowcastResult:
    def _kwargs(self, **over):
        params = dict(
            quarter=pd.Period("2026Q2", freq="Q"),
            horizon=1,
            as_of=AS_OF,
            model="test",
            target="reported",
            point=0.01,
            draws=np.linspace(-0.05, 0.07, 1001),
            interval_method="empirical_recursive",
        )
        params.update(over)
        return params

    def test_intervals_come_from_the_draws(self):
        res = make_nowcast_result(**self._kwargs())
        assert res.interval_80 == pytest.approx(np.quantile(res.draws, [0.1, 0.9]))
        assert res.interval_95 == pytest.approx(
            np.quantile(res.draws, [0.025, 0.975])
        )
        assert res.interval(0.5)[0] > res.interval_80[0]

    def test_non_finite_draws_raise_rather_than_propagate(self):
        draws = np.linspace(-0.05, 0.07, 100)
        draws[3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            make_nowcast_result(**self._kwargs(draws=draws))

    def test_non_finite_point_raises(self):
        with pytest.raises(ValueError, match="point nowcast .* is not finite"):
            make_nowcast_result(**self._kwargs(point=np.nan))

    def test_empty_draws_raise(self):
        with pytest.raises(ValueError, match="are empty"):
            make_nowcast_result(**self._kwargs(draws=np.empty(0)))

    def test_covers_and_to_row(self):
        res = make_nowcast_result(**self._kwargs())
        assert res.covers(0.01, 0.80)
        assert not res.covers(0.50, 0.95)
        row = res.to_row()
        assert row["quarter"] == pd.Period("2026Q2", freq="Q")
        assert row["lower_80"] == res.interval_80[0]
        assert "draws" not in row

    def test_bad_level_rejected(self):
        res = make_nowcast_result(**self._kwargs())
        with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
            res.interval(0.0)


# ──────────────────────────────────────────────────────────────────
# Model 2 — the λ convention (CHECKPOINT 2)
# ──────────────────────────────────────────────────────────────────


class TestRollingAnnualAutocorrelation:
    r"""The :math:`\rho_1` of an overlapping sum, against simulation."""

    @pytest.mark.parametrize(
        "phi,expected",
        [(0.0, 0.7501), (0.3, 0.8405), (0.6, 0.9082), (0.8, 0.9524), (0.9, 0.9756)],
    )
    def test_matches_simulation(self, phi, expected):
        """Simulated values from a 4M-draw AR(1), 4-quarter overlapping sum."""
        assert rolling_annual_ar1_autocorrelation(phi) == pytest.approx(
            expected, abs=2e-4
        )

    def test_overlap_floor_at_zero_persistence(self):
        """Working (1960): a rolling 4-period sum is 75 % autocorrelated at φ=0."""
        assert rolling_annual_ar1_autocorrelation(0.0) == pytest.approx(0.75)
        assert rolling_annual_ar1_autocorrelation(0.0, window=2) == pytest.approx(0.5)

    def test_monotone_and_bounded(self):
        phis = np.linspace(0.0, 0.99, 50)
        rho = rolling_annual_ar1_autocorrelation(phis)
        assert rho.shape == phis.shape
        assert np.all(np.diff(rho) > 0)
        assert np.all((rho >= 0.75) & (rho < 1.0))

    def test_scalar_and_vector_agree(self):
        vec = rolling_annual_ar1_autocorrelation(np.array([0.3, 0.6]))
        assert vec[0] == pytest.approx(rolling_annual_ar1_autocorrelation(0.3))
        assert vec[1] == pytest.approx(rolling_annual_ar1_autocorrelation(0.6))


class TestLambdaConversion:
    """The three conventions, and why the fourth (a power law on rolling) is absent."""

    def test_quarterly_is_the_identity(self):
        lam_q, details = convert_lambda_to_quarterly(0.6, "quarterly")
        assert lam_q == 0.6
        assert "identity" in details["formula"]

    @pytest.mark.parametrize("lam_q", [0.1, 0.3, 0.6, 0.85, 0.95])
    def test_non_overlapping_annual_round_trips(self, lam_q):
        r"""Sampling every 4th quarter gives :math:`\lambda_a = \lambda_q^4`."""
        lam_a = lam_q**4
        recovered, details = convert_lambda_to_quarterly(
            lam_a, "annual_non_overlapping"
        )
        assert recovered == pytest.approx(lam_q, abs=1e-12)
        assert details["formula"] == "lambda_q = lambda_a ** (1/4)"

    @pytest.mark.parametrize("lam_q", [0.05, 0.3, 0.6, 0.85, 0.95])
    def test_rolling_annual_round_trips_via_inversion(self, lam_q):
        lam_a = rolling_annual_ar1_autocorrelation(lam_q)
        recovered, details = convert_lambda_to_quarterly(lam_a, "annual_rolling")
        assert recovered == pytest.approx(lam_q, abs=1e-8)
        assert "NOT a power law" in details["formula"]

    def test_rolling_annual_power_law_would_be_catastrophic(self):
        r"""The number this module refuses to return, and the damage it would do.

        A true quarterly λ of 0.60 shows up as λ_a ≈ 0.908 when an AR(1) is
        fitted to rolling four-quarter returns. Desmoothing divides by (1−λ), so
        mistaking 0.908^(1/4) = 0.976 for the quarterly λ would inflate the
        recovered volatility by a factor of ~17.
        """
        lam_a = rolling_annual_ar1_autocorrelation(0.60)
        assert lam_a == pytest.approx(0.9081, abs=1e-4)
        lam_q, details = convert_lambda_to_quarterly(lam_a, "annual_rolling")
        assert lam_q == pytest.approx(0.60, abs=1e-8)
        assert details["power_law_would_give"] == pytest.approx(0.9762, abs=1e-4)
        vol_error = (1.0 - lam_q) / (1.0 - details["power_law_would_give"])
        assert vol_error > 15.0

    def test_rolling_annual_below_the_overlap_floor_raises(self):
        """No quarterly λ can produce ρ₁ < 0.75 for a 4-quarter overlap."""
        with pytest.raises(ValueError, match="overlap floor"):
            convert_lambda_to_quarterly(0.5, "annual_rolling")
        with pytest.raises(ValueError, match="overlap floor"):
            convert_lambda_to_quarterly(0.75, "annual_rolling")

    def test_conversions_are_ordered_as_expected(self):
        """For the same fitted λ, rolling < non-overlapping-power-law."""
        lam_a = 0.90
        rolling, _ = convert_lambda_to_quarterly(lam_a, "annual_rolling")
        power, _ = convert_lambda_to_quarterly(lam_a, "annual_non_overlapping")
        assert rolling < power
        assert power == pytest.approx(lam_a**0.25)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_lambda_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError, match=r"lambda must be in \[0, 1\)"):
            convert_lambda_to_quarterly(bad, "quarterly")

    def test_unknown_convention_rejected(self):
        with pytest.raises(ValueError, match="unknown lambda convention"):
            convert_lambda_to_quarterly(0.6, "annual")


class TestInferFittedConvention:
    @staticmethod
    def _result(index) -> DesmoothedResult:
        return DesmoothedResult(
            true_returns=pd.Series(np.zeros(len(index)), index=index),
            smoothing_params={},
            posterior_summary={},
            diagnostics={"method": "ar1_bayesian"},
        )

    def test_quarterly_datetime_index(self):
        index = pd.period_range("2001Q1", periods=40, freq="Q").to_timestamp(how="end")
        convention, details = infer_fitted_convention(self._result(index))
        assert convention == "quarterly"
        assert 88 <= details["median_spacing_days"] <= 93

    def test_quarterly_period_index(self):
        index = pd.period_range("2001Q1", periods=40, freq="Q")
        convention, _ = infer_fitted_convention(self._result(index))
        assert convention == "quarterly"

    def test_annual_datetime_index(self):
        index = pd.date_range("2001-12-31", periods=25, freq="365D")
        convention, _ = infer_fitted_convention(self._result(index))
        assert convention == "annual_non_overlapping"

    def test_annual_period_index(self):
        index = pd.period_range("2001", periods=25, freq="Y")
        convention, _ = infer_fitted_convention(self._result(index))
        assert convention == "annual_non_overlapping"

    def test_monthly_index_refuses_to_guess(self):
        # "M" not "ME": the installed pandas is 2.1.x, where "ME" is invalid.
        index = pd.date_range("2001-01-31", periods=120, freq="M")
        with pytest.raises(ValueError, match="neither quarterly .* nor annual"):
            infer_fitted_convention(self._result(index))

    def test_too_short_to_infer(self):
        index = pd.period_range("2001Q1", periods=1, freq="Q")
        with pytest.raises(ValueError, match="cannot infer"):
            infer_fitted_convention(self._result(index))

    def test_rolling_annual_is_indistinguishable_by_index(self):
        """A documented limitation, pinned so it cannot be forgotten.

        A rolling four-quarter return series is indexed *quarterly* — only its
        values are annual — so inference cannot tell it from a quarterly fit and
        reports ``'quarterly'``, leaving λ unconverted and far too high. The
        caller must pass ``fitted_convention='annual_rolling'`` explicitly.
        """
        index = pd.period_range("2001Q1", periods=40, freq="Q")
        convention, _ = infer_fitted_convention(self._result(index))
        assert convention == "quarterly", (
            "if this ever returns 'annual_rolling', the docstring warnings in "
            "structural.py and nowcast_notes.md can be relaxed"
        )


# ──────────────────────────────────────────────────────────────────
# Model 2 — parameter extraction and nowcasting
# ──────────────────────────────────────────────────────────────────


def _ar1_priors(beta_mean: float = 1.15) -> dict:
    return {
        "lambda": BetaDist(2, 2),
        "beta": {"equity_market": NormalPrior(beta_mean, 0.5)},
        "alpha": NormalPrior(0.0, 0.05),
        "sigma_eps": InverseGammaPrior(3, 0.02),
    }


def _published_frames(info):
    """Published reported returns and aligned quarterly factors, timestamp-indexed.

    The desmoothing models take a ``DatetimeIndex``, and fitting them *inside* the
    information set is what keeps the borrowed parameters non-leaky.
    """
    published = pd.PeriodIndex(info.reported.index, freq="Q")
    obs = info.reported.copy()
    obs.index = published.to_timestamp(how="end")
    factors = info.factors_quarterly.loc[published].copy()
    factors.index = obs.index
    return obs, factors


def _fit_ar1(info, *, beta_mean: float = 1.15):
    """Fit ``ar1_bayesian`` on the information set's published rows only."""
    smoother = AR1BayesianSmoother()
    obs, factors = _published_frames(info)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smoother, smoother.fit(obs, factors, _ar1_priors(beta_mean))


def _fit_rudin(info, *, n_lags: int = 1):
    """Fit ``rudin_reparam`` on the information set's published rows only."""
    smoother = RudinReparamSmoothing(n_lags=n_lags)
    obs, factors = _published_frames(info)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smoother, smoother.fit(obs, factors, {})


def _true_param_result(ds, info) -> DesmoothedResult:
    """A ``DesmoothedResult`` carrying the DGP's *true* parameters.

    Lets the structural nowcaster be checked against the closed form with no
    estimation error in the way — the second half of spec test 5.
    """
    published = pd.PeriodIndex(info.reported.index, freq="Q")
    return DesmoothedResult(
        true_returns=pd.Series(
            ds.true_returns.loc[published].to_numpy(),
            index=published.to_timestamp(how="end"),
        ),
        smoothing_params={
            "lambda": ds.params["lam"],
            "beta": dict(ds.params["beta"]),
            "alpha": ds.params["alpha"],
            "sigma_eps": ds.params["sigma_eps"],
        },
        posterior_summary={},
        diagnostics={"method": "ar1_bayesian"},
    )


class TestStructuralParameterExtraction:
    def test_ar1_alpha_is_already_unsmoothed_scale(self, motivating_dataset):
        r"""``ar1_bayesian`` regresses :math:`(s_t-\lambda s_{t-1})/(1-\lambda)`.

        So its reported ``alpha`` and ``beta`` are true-return quantities, and
        the *reported*-equation intercept is the derived :math:`\theta_0\alpha`.
        """
        info = _info(motivating_dataset)
        _, result = _fit_ar1(info)
        model = StructuralSmoothingNowcaster(desmoothed=result)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(info)
        params = model.structural_params
        assert params.alpha_true == pytest.approx(result.smoothing_params["alpha"])
        assert params.intercept_reported == pytest.approx(
            params.theta_0 * params.alpha_true
        )
        assert params.theta_0 == pytest.approx(
            1.0 - result.smoothing_params["lambda"]
        )
        assert params.beta == result.smoothing_params["beta"]

    def test_rudin_alpha_is_rescaled_to_the_unsmoothed_scale(
        self, motivating_dataset
    ):
        r"""``rudin_reparam`` reports the raw Eq. 4 intercept but an unwound β.

        The two are on different scales in that result object. This module takes
        the intercept as the reported-equation one and derives
        :math:`\alpha = a/\theta_0`, consistent with its own
        :math:`\beta_i = c_i/\theta_0`.
        """
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        model = StructuralSmoothingNowcaster(desmoothed=result)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(info)
        params = model.structural_params
        raw_intercept = result.smoothing_params["alpha"]
        assert params.intercept_reported == pytest.approx(raw_intercept)
        assert params.alpha_true == pytest.approx(raw_intercept / params.theta_0)
        assert params.theta_0 == pytest.approx(
            result.smoothing_params["theta"][0]
        )

    def test_okunev_white_is_refused_with_an_explanation(self, motivating_dataset):
        info = _info(motivating_dataset)
        factor_free = DesmoothedResult(
            true_returns=info.reported.copy(),
            smoothing_params={"theta": np.array([0.5, 0.5])},
            posterior_summary={},
            diagnostics={"method": "okunev_white"},
        )
        model = StructuralSmoothingNowcaster(
            desmoothed=factor_free, fitted_convention="quarterly"
        )
        with pytest.raises(UnsupportedSmoothingModelError, match="factor-free"):
            model.fit(info)

    def test_unknown_method_is_refused(self, motivating_dataset):
        info = _info(motivating_dataset)
        unknown = DesmoothedResult(
            true_returns=info.reported.copy(),
            smoothing_params={"lambda": 0.6, "beta": {"equity_market": 1.0}},
            posterior_summary={},
            diagnostics={"method": "ma_glm"},
        )
        model = StructuralSmoothingNowcaster(
            desmoothed=unknown, fitted_convention="quarterly"
        )
        with pytest.raises(UnsupportedSmoothingModelError, match="does not support"):
            model.fit(info)

    def test_missing_betas_refused(self, motivating_dataset):
        info = _info(motivating_dataset)
        no_beta = DesmoothedResult(
            true_returns=info.reported.copy(),
            smoothing_params={"lambda": 0.6, "beta": {}},
            posterior_summary={},
            diagnostics={"method": "ar1_bayesian"},
        )
        model = StructuralSmoothingNowcaster(
            desmoothed=no_beta, fitted_convention="quarterly"
        )
        with pytest.raises(UnsupportedSmoothingModelError, match="no factor loadings"):
            model.fit(info)

    def test_rudin_two_lags_on_an_annual_convention_refuses_to_invent_a_mapping(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info, n_lags=2)
        model = StructuralSmoothingNowcaster(
            desmoothed=result, fitted_convention="annual_rolling"
        )
        with pytest.raises(ValueError, match="will not invent one"):
            model.fit(info)

    def test_rudin_two_lags_on_quarterly_is_fine(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info, n_lags=2)
        model = StructuralSmoothingNowcaster(desmoothed=result)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(info)
        assert model.structural_params.n_lags == 2
        assert model._lag_depth() == 2
        assert model.lambda_quarterly is None

    def test_convention_is_inferred_as_quarterly_for_the_shipped_fit(
        self, motivating_dataset
    ):
        """The headline finding: as built, no λ conversion is needed."""
        info = _info(motivating_dataset)
        _, result = _fit_ar1(info)
        model = StructuralSmoothingNowcaster(desmoothed=result)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(info)
        diag = model.fit_diagnostics
        assert diag["fitted_convention"] == "quarterly"
        assert "identity" in diag["lambda_conversion"]["formula"]
        assert diag["lambda_quarterly"] == pytest.approx(
            result.smoothing_params["lambda"]
        )

    def test_explicit_rolling_convention_changes_lambda(self, motivating_dataset):
        """And if you declare a rolling fit, λ is converted downward, hard."""
        info = _info(motivating_dataset)
        _, result = _fit_ar1(info)
        lam_fitted = result.smoothing_params["lambda"]
        model = StructuralSmoothingNowcaster(
            desmoothed=result, fitted_convention="annual_rolling"
        )
        if lam_fitted > 0.75:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(info)
            assert model.lambda_quarterly < lam_fitted
        else:
            with pytest.raises(ValueError, match="overlap floor"):
                model.fit(info)


class TestStructuralNowcasting:
    def test_rudin_reproduces_the_smoothing_regression_exactly(
        self, motivating_dataset
    ):
        r"""Both are the same OLS reduced form, so the points must coincide.

        ``rudin_reparam`` fits ``s_t = a + c·F_t + θ₁·s_{t-1}`` and reports
        ``β = c/θ₀``; multiplying back by ``θ₀`` recovers ``c`` exactly, so the
        structural nowcast is the regression's own prediction to floating-point
        precision. This is the tightest available check that the Eq. 4
        reconstruction is right.
        """
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        structural, s_res = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result), info
        )
        _, r_res = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        for a, b in _paired(s_res, r_res):
            assert a.point == pytest.approx(b.point, abs=1e-12)

    def test_true_parameters_match_the_closed_form(self, motivating_dataset):
        r"""Spec test 5, second half: with the true (λ, α, β) the formula is exact."""
        info = _info(motivating_dataset)
        result = _true_param_result(motivating_dataset, info)
        model, results = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result), info
        )
        lam = motivating_dataset.params["lam"]
        alpha = motivating_dataset.params["alpha"]
        beta = motivating_dataset.params["beta"]["equity_market"]
        s_prev = float(info.reported.iloc[-1])

        q2, q3 = info.unpublished_quarters()
        f2 = float(info.factor_row(q2)["equity_market"])
        f3 = float(info.factor_row(q3)["equity_market"])
        expected_h1 = (1 - lam) * (alpha + beta * f2) + lam * s_prev
        expected_h2 = (1 - lam) * (alpha + beta * f3) + lam * expected_h1
        assert results[0].point == pytest.approx(expected_h1, abs=1e-12)
        assert results[1].point == pytest.approx(expected_h2, abs=1e-12)

    def test_true_parameters_agree_with_the_estimated_benchmark(
        self, motivating_dataset
    ):
        """Spec test 5: the two routes should land within estimation error."""
        info = _info(motivating_dataset)
        result = _true_param_result(motivating_dataset, info)
        structural, s_res = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result), info
        )
        regression, r_res = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        sigma = regression.sigma
        for a, b in _paired(s_res, r_res):
            assert abs(a.point - b.point) < 1.0 * sigma, (
                f"h={a.horizon}: structural {a.point:.5f} vs regression "
                f"{b.point:.5f}, sigma={sigma:.5f}"
            )

    def test_ar1_bayesian_route_is_close_to_the_benchmark(self, motivating_dataset):
        """Prior-driven differences are expected and bounded, not arbitrary."""
        info = _info(motivating_dataset)
        _, result = _fit_ar1(info)
        _, s_res = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result), info
        )
        regression, r_res = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        for a, b in _paired(s_res, r_res):
            assert abs(a.point - b.point) < 1.5 * regression.sigma

    def test_beta_prior_moves_lambda_the_confound_in_action(self, motivating_dataset):
        r"""The λ-β confound the parent library warns about, demonstrated.

        Raising the β prior pulls β up, and because λ and β are only jointly
        identified, λ follows. This is why λ-only posterior draws understate
        parameter uncertainty.
        """
        info = _info(motivating_dataset)
        _, low = _fit_ar1(info, beta_mean=0.7)
        _, high = _fit_ar1(info, beta_mean=1.7)
        assert high.smoothing_params["beta"]["equity_market"] > (
            low.smoothing_params["beta"]["equity_market"]
        )
        assert high.smoothing_params["lambda"] > low.smoothing_params["lambda"]

    def test_horizon_two_is_wider(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        _, results = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result), info, n_draws=20_000
        )
        assert results[1].draw_std > results[0].draw_std

    def test_direct_interval_method(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        _, results = _fit_predict(
            StructuralSmoothingNowcaster(
                desmoothed=result, interval_method="empirical_direct"
            ),
            info,
        )
        assert results[0].interval_method == "empirical_direct"

    def test_parametric_interval_method_is_refused_with_a_pointer(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        model = StructuralSmoothingNowcaster(
            desmoothed=result, interval_method="parametric"
        )
        with pytest.raises(ValueError, match="integrate_lambda_posterior"):
            model.fit(info)

    def test_missing_desmoothed_result_is_refused(self, motivating_dataset):
        info = _info(motivating_dataset)
        with pytest.raises(ValueError, match="requires `desmoothed`"):
            StructuralSmoothingNowcaster().fit(info)

    def test_wrong_type_for_desmoothed(self, motivating_dataset):
        info = _info(motivating_dataset)
        model = StructuralSmoothingNowcaster(desmoothed={"lambda": 0.6})
        with pytest.raises(TypeError, match="must be a DesmoothedResult"):
            model.fit(info)

    def test_satisfies_the_protocol(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        assert isinstance(
            StructuralSmoothingNowcaster(desmoothed=result), Nowcaster
        )

    def test_zero_estimated_parameters_reported(self, motivating_dataset):
        """Nothing is estimated here, and the diagnostics say so."""
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        model, _ = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result), info
        )
        assert model.fit_diagnostics["n_params"] == 0
        assert model.residual_pool.inflation == pytest.approx(1.0)

    def test_target_true_uses_the_borrowed_alpha_and_beta(self, motivating_dataset):
        info = _info(motivating_dataset)
        result = _true_param_result(motivating_dataset, info)
        model, results = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result, target="true"), info
        )
        alpha = motivating_dataset.params["alpha"]
        beta = motivating_dataset.params["beta"]["equity_market"]
        for r in results:
            expected = alpha + beta * float(
                info.factor_row(r.quarter)["equity_market"]
            )
            assert r.point == pytest.approx(expected, abs=1e-12)
            assert r.target == "true"

    def test_target_true_width_is_horizon_invariant(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        _, results = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result, target="true"),
            info,
            n_draws=100_000,
        )
        assert results[1].draw_std == pytest.approx(results[0].draw_std, rel=0.03)


class TestStructuralLeakGuards:
    """The borrowed-parameter leak that ``DESIGN_NOTES.md`` Risk 1 promises to flag."""

    def test_full_sample_fit_warns_about_a_leaky_backtest(self, motivating_dataset):
        info = _info(motivating_dataset)
        full_obs = motivating_dataset.reported.copy()
        full_obs.index = pd.PeriodIndex(full_obs.index, freq="Q").to_timestamp(
            how="end"
        )
        full_factors = motivating_dataset.quarterly_factors.loc[
            pd.PeriodIndex(motivating_dataset.reported.index, freq="Q")
        ].copy()
        full_factors.index = full_obs.index
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leaky = AR1BayesianSmoother().fit(
                full_obs, full_factors, _ar1_priors()
            )
        model = StructuralSmoothingNowcaster(desmoothed=leaky)
        with pytest.warns(StructuralSmoothingWarning, match="leaky"):
            model.fit(info)
        first, last = model.fit_diagnostics["structural_fit_span"]
        assert last > str(info.last_published_quarter)

    def test_strict_policy_escalates_the_leak_warning(self, motivating_dataset):
        info = _info(motivating_dataset)
        full_obs = motivating_dataset.reported.copy()
        full_obs.index = pd.PeriodIndex(full_obs.index, freq="Q").to_timestamp(
            how="end"
        )
        full_factors = motivating_dataset.quarterly_factors.loc[
            pd.PeriodIndex(motivating_dataset.reported.index, freq="Q")
        ].copy()
        full_factors.index = full_obs.index
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leaky = AR1BayesianSmoother().fit(full_obs, full_factors, _ar1_priors())
        model = StructuralSmoothingNowcaster(
            desmoothed=leaky, fallback_policy=FallbackPolicy.STRICT
        )
        with pytest.raises(StructuralSmoothingWarning, match="leaky"):
            model.fit(info)

    def test_in_sample_fit_does_not_warn(self, motivating_dataset):
        """Fitted inside the information set — the non-leaky path."""
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        model = StructuralSmoothingNowcaster(desmoothed=result)
        with warnings.catch_warnings():
            warnings.simplefilter("error", StructuralSmoothingWarning)
            model.fit(info)
        _, last = model.fit_diagnostics["structural_fit_span"]
        assert last == str(info.last_published_quarter)

    def test_predictions_are_bit_for_bit_stable_against_future_data(
        self, motivating_dataset
    ):
        """Spec test 1 for Model 2, with the desmoothing refitted inside each set."""

        def run(public_factors, vintages):
            info = _info(
                motivating_dataset,
                public_factors=public_factors,
                vintages=vintages,
            )
            _, result = _fit_rudin(info)
            return _fit_predict(
                StructuralSmoothingNowcaster(desmoothed=result), info
            )[1]

        base = run(motivating_dataset.daily_factors, motivating_dataset.vintages)

        perturbed_factors = motivating_dataset.daily_factors.copy()
        perturbed_factors.loc[perturbed_factors.index > AS_OF] += 0.25
        perturbed_vintages = motivating_dataset.vintages.copy()
        perturbed_vintages.loc[
            perturbed_vintages["release_date"] > AS_OF, "value"
        ] += 0.5
        other = run(perturbed_factors, perturbed_vintages)

        for a, b in _paired(base, other):
            assert a.point == b.point
            np.testing.assert_array_equal(a.draws, b.draws)

    def test_missing_factor_loading_warns(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        widened = DesmoothedResult(
            true_returns=result.true_returns,
            smoothing_params={
                **result.smoothing_params,
                "beta": {**result.smoothing_params["beta"], "ghost_factor": 0.3},
            },
            posterior_summary=result.posterior_summary,
            diagnostics=result.diagnostics,
        )
        model = StructuralSmoothingNowcaster(desmoothed=widened)
        with pytest.warns(StructuralSmoothingWarning, match="ghost_factor"):
            model.fit(info)

    def test_no_overlapping_factors_is_fatal(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        disjoint = DesmoothedResult(
            true_returns=result.true_returns,
            smoothing_params={**result.smoothing_params, "beta": {"unrelated": 1.0}},
            posterior_summary=result.posterior_summary,
            diagnostics=result.diagnostics,
        )
        model = StructuralSmoothingNowcaster(desmoothed=disjoint)
        with pytest.raises(ValueError, match="identically zero"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(info)


class TestStructuralPosteriorIntegration:
    def test_lambda_grid_draws_widen_the_interval_and_warn(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, result = _fit_ar1(info)
        _, point_based = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result), info, n_draws=50_000
        )
        model = StructuralSmoothingNowcaster(
            desmoothed=result, integrate_lambda_posterior=True
        )
        with pytest.warns(StructuralSmoothingWarning, match="confound"):
            model.fit(info)
        integrated = model.predict(
            info,
            info.unpublished_quarters(),
            n_draws=50_000,
            rng=np.random.default_rng(7),
        )
        assert model.fit_diagnostics["posterior_draw_mode"] == "lambda_marginal"
        assert integrated[0].draw_std > point_based[0].draw_std

    def test_joint_draws_via_the_smoother_instance(self, motivating_dataset):
        info = _info(motivating_dataset)
        smoother, result = _fit_ar1(info)
        model = StructuralSmoothingNowcaster(
            desmoothed=result, integrate_lambda_posterior=True, smoother=smoother
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(info)
            results = model.predict(
                info,
                info.unpublished_quarters(),
                n_draws=50_000,
                rng=np.random.default_rng(7),
            )
        assert model.fit_diagnostics["posterior_draw_mode"] == "joint"
        _, point_based = _fit_predict(
            StructuralSmoothingNowcaster(desmoothed=result), info, n_draws=50_000
        )
        assert results[0].draw_std > point_based[0].draw_std

    def test_non_ar1_result_warns_and_falls_back_to_point_parameters(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        _, result = _fit_rudin(info)
        model = StructuralSmoothingNowcaster(
            desmoothed=result, integrate_lambda_posterior=True
        )
        with pytest.warns(StructuralSmoothingWarning, match="only meaningful"):
            model.fit(info)
        assert model.fit_diagnostics["posterior_draw_mode"] == "none"

    def test_missing_grid_warns_and_falls_back(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, result = _fit_ar1(info)
        stripped = DesmoothedResult(
            true_returns=result.true_returns,
            smoothing_params=result.smoothing_params,
            posterior_summary={"lambda": {"mean": 0.6}},
            diagnostics=result.diagnostics,
        )
        model = StructuralSmoothingNowcaster(
            desmoothed=stripped, integrate_lambda_posterior=True
        )
        with pytest.warns(StructuralSmoothingWarning, match="no 'grid'/'density'"):
            model.fit(info)
        assert model.fit_diagnostics["posterior_draw_mode"] == "none"


# ──────────────────────────────────────────────────────────────────
# Model 3 — the feature budget (spec test 9)
# ──────────────────────────────────────────────────────────────────


class TestFeatureBudgetCounting:
    """Counted from word counts alone, before any path or signature is built."""

    def test_counts_exclude_time_only_terms(self):
        budget = feature_budget(100, n_channels=2, level=2, keep_sigs="linear")
        # linear words at d=2, L=2: (0), (1), (0,0), (0,1), (1,0).
        # Time-only — (0) and (0,0) — are structurally constant and dropped.
        assert budget["n_signature_terms"] == 3
        assert budget["n_time_only_dropped"] == 2
        assert budget["n_features"] == 5  # 3 signature + s_lag1 + intercept
        assert budget["budget"] == 20.0
        assert budget["within_budget"]

    @pytest.mark.parametrize(
        "n_channels,level,keep_sigs,n_features,within",
        [
            (2, 1, "linear", 3, True),
            (2, 2, "linear", 5, True),
            (2, 2, "all", 6, True),
            (3, 2, "linear", 10, True),
            (3, 3, "linear", 22, False),
            (2, 4, "all", 28, False),
        ],
    )
    def test_budget_boundary_at_one_hundred_rows(
        self, n_channels, level, keep_sigs, n_features, within
    ):
        budget = feature_budget(
            100, n_channels=n_channels, level=level, keep_sigs=keep_sigs
        )
        assert budget["n_features"] == n_features
        assert budget["within_budget"] is within

    def test_multiplier_terms_count_toward_the_budget(self):
        none = feature_budget(100, n_channels=2, level=2, keep_sigs="linear")
        time_mult = feature_budget(
            100, n_channels=2, level=2, keep_sigs="linear", multiplier_terms="time"
        )
        all_mult = feature_budget(
            100, n_channels=2, level=2, keep_sigs="linear", multiplier_terms="all"
        )
        assert time_mult["n_features"] == none["n_features"] + 2
        assert all_mult["n_features"] == none["n_features"] + 3

    def test_bad_arguments(self):
        with pytest.raises(ValueError, match="keep_sigs must be"):
            feature_budget(100, n_channels=2, level=2, keep_sigs="some")
        with pytest.raises(ValueError, match="multiplier_terms must be"):
            feature_budget(
                100,
                n_channels=2,
                level=2,
                keep_sigs="linear",
                multiplier_terms="other",
            )


class TestFeatureBudgetGuard:
    """Spec test 9: refused under ``strict``, skipped with a warning under ``warn``."""

    def test_over_budget_configuration_refused_under_strict(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        model = SignatureNowcaster(
            level=4,
            keep_sigs="all",
            alpha=1e-3,
            fallback_policy=FallbackPolicy.STRICT,
        )
        with pytest.raises(FeatureBudgetError, match="exceed the feature budget"):
            model.fit(info)

    def test_over_budget_configuration_skipped_with_a_warning_under_warn(
        self, motivating_dataset
    ):
        """The whole grid is over budget, so there is nothing to fall back to."""
        info = _info(motivating_dataset)
        model = SignatureNowcaster(level=4, keep_sigs="all", alpha=1e-3)
        with pytest.warns(SignatureNowcasterWarning, match="exceed the feature budget"):
            with pytest.raises(FeatureBudgetError, match="No configuration is admissible|every configuration"):
                model.fit(info)

    def test_over_budget_members_of_a_grid_are_skipped_not_fatal(
        self, motivating_dataset
    ):
        """A mixed grid proceeds on its admissible members."""
        info = _info(motivating_dataset)
        model = SignatureNowcaster(
            level=None, level_candidates=(2, 4), keep_sigs="all", alpha=1e-3
        )
        with pytest.warns(SignatureNowcasterWarning, match="1 of 2 configuration"):
            model.fit(info)
        assert model.fit_diagnostics["level"] == 2
        assert model.fit_diagnostics["selection"]["n_skipped_over_budget"] == 1

    def test_budget_is_reported_in_the_diagnostics(self, motivating_dataset):
        """Spec: 'state the computed budget in the fitted model's diagnostics'."""
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SignatureNowcaster(), info)
        budget = model.fit_diagnostics["feature_budget"]
        assert budget["n_features"] == 5
        assert budget["n_train"] == 100
        assert budget["budget"] == 20.0
        assert budget["budget_divisor"] == 5
        assert budget["within_budget"]
        assert budget["n_time_only_dropped"] == 2

    def test_a_short_history_shrinks_the_budget(self, motivating_dataset):
        """The guard binds on sample size, not only on configuration."""
        short = _info(motivating_dataset, as_of=pd.Timestamp("2008-01-15"))
        assert short.n_published < 30
        model = SignatureNowcaster(
            level=3, keep_sigs="all", alpha=1e-3, fallback_policy=FallbackPolicy.STRICT
        )
        with pytest.raises(FeatureBudgetError):
            model.fit(short)

    def test_every_configuration_over_budget_names_the_remedies(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        model = SignatureNowcaster(level=5, keep_sigs="all", alpha=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(FeatureBudgetError) as excinfo:
                model.fit(info)
        message = str(excinfo.value)
        assert "keep_sigs='linear'" in message
        assert "n_components" in message


# ──────────────────────────────────────────────────────────────────
# Model 3 — spec test 4, nesting
# ──────────────────────────────────────────────────────────────────


class TestSignatureNesting:
    r"""Spec test 4: at ``level=1`` the signature model *is* the benchmark."""

    def test_predictions_match_the_smoothing_regression_exactly(
        self, motivating_dataset
    ):
        r"""Exact, not approximate — two deliberate choices make it so.

        ``level_channel='simple'`` makes the level-1 signature term equal
        :math:`F_Q` to the last bit rather than :math:`\log(1+F_Q)`, and dropping
        the structurally constant time-only terms removes what would otherwise be
        exact collinearity with the intercept. What remains is
        ``[intercept, F_Q, s_lag1]`` — the smoothing regression's own design.
        """
        info = _info(motivating_dataset)
        signature, sig_res = _fit_predict(
            SignatureNowcaster(
                level=1,
                lookback=1,
                keep_sigs="linear",
                alpha=0.0,
                l1_ratio=0.0,
                level_channel="simple",
            ),
            info,
        )
        regression, reg_res = _fit_predict(
            SmoothingRegressionNowcaster(n_lags=1), info
        )
        assert signature.fit_diagnostics["features"] == [
            "intercept",
            "sig[equity_market]",
            "s_lag1",
        ]
        assert signature.fit_diagnostics["n_train"] == (
            regression.fit_diagnostics["n_train"]
        )
        for a, b in _paired(sig_res, reg_res):
            assert a.point == pytest.approx(b.point, abs=1e-12), (
                f"h={a.horizon}: {a.point!r} vs {b.point!r}"
            )

    def test_the_lag_coefficient_is_identical(self, motivating_dataset):
        """Standardisation rescales the factor coefficient but not this one."""
        info = _info(motivating_dataset)
        signature, _ = _fit_predict(
            SignatureNowcaster(
                level=1, alpha=0.0, l1_ratio=0.0, level_channel="simple"
            ),
            info,
        )
        regression, _ = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        assert signature.coefficients["s_lag1"] == pytest.approx(
            regression.coefficients["s_lag1"], abs=1e-12
        )

    def test_standardisation_does_not_change_predictions(self, motivating_dataset):
        """It is a linear reparameterisation, so OLS fitted values are invariant."""
        info = _info(motivating_dataset)
        _, scaled = _fit_predict(
            SignatureNowcaster(level=2, alpha=0.0, standardize=True), info
        )
        _, unscaled = _fit_predict(
            SignatureNowcaster(level=2, alpha=0.0, standardize=False), info
        )
        for a, b in _paired(scaled, unscaled):
            assert a.point == pytest.approx(b.point, abs=1e-9)

    def test_log_channel_nests_only_approximately(self, motivating_dataset):
        r"""The default ``'log'`` channel gives :math:`\log(1+F_Q)`, not :math:`F_Q`.

        Documents why ``'simple'`` exists: the difference is second order but it
        is not zero, so the nesting would be a tolerance argument rather than an
        identity.
        """
        info = _info(motivating_dataset)
        _, log_res = _fit_predict(
            SignatureNowcaster(level=1, alpha=0.0, level_channel="log"), info
        )
        _, reg_res = _fit_predict(SmoothingRegressionNowcaster(n_lags=1), info)
        gaps = [abs(a.point - b.point) for a, b in _paired(log_res, reg_res)]
        assert max(gaps) > 1e-10, "if this is exact, 'simple' is redundant"
        assert max(gaps) < 0.01, "but it should still be a second-order difference"

    def test_time_only_terms_are_dropped_and_recorded(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SignatureNowcaster(level=2), info)
        features = model.fit_diagnostics["features"]
        assert "sig[time]" not in features
        assert "sig[time,time]" not in features
        assert model.fit_diagnostics["path"]["dropped_time_only_terms"] == (
            "sig[time]",
            "sig[time,time]",
        )
        # The informative mixed terms survive.
        assert "sig[time,equity_market]" in features
        assert "sig[equity_market,time]" in features


# ──────────────────────────────────────────────────────────────────
# Model 3 — spec test 6, path dependence
# ──────────────────────────────────────────────────────────────────


def _oos_rmse(
    dataset,
    factories: dict,
    *,
    first_origin: int = 75,
    horizons: tuple[int, ...] = (1, 2),
) -> dict[str, dict[int, float]]:
    r"""Rolling-origin out-of-sample RMSE for several models on one dataset.

    At each origin ``o`` the information set is built at
    ``as_of = quarters[o+2].end_time``, which under the 100-day publication lag
    leaves ``o+1`` at horizon 1 and ``o+2`` at horizon 2. Every model is re-fitted
    from scratch at every origin — coefficients, standardisation, hyperparameters
    and residual pool — so the comparison is between forecasts a user could
    actually have made.
    """
    quarters = pd.PeriodIndex(dataset.reported.index, freq="Q")
    errors = {name: {h: [] for h in horizons} for name in factories}
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
            for name, factory in factories.items():
                model = factory().fit(info)
                for result in model.predict(
                    info, targets, n_draws=20, rng=np.random.default_rng(0)
                ):
                    if result.horizon in horizons:
                        errors[name][result.horizon].append(
                            float(dataset.first_print[result.quarter]) - result.point
                        )
    return {
        name: {
            h: float(np.sqrt(np.mean(np.square(v)))) if v else float("nan")
            for h, v in per_h.items()
        }
        for name, per_h in errors.items()
    }


@pytest.mark.slow
class TestPathDependence:
    """Spec test 6, both directions."""

    def test_level_two_beats_the_benchmark_when_the_path_matters(self):
        r"""On the path-dependent DGP the appraiser marks against the average level.

        That statistic is the level-2 term :math:`S^{(x,t)} = \int X\,\mathrm{d}s`,
        which ``level=1`` cannot see. The margin should be large, not marginal —
        this DGP was built so the signal is unambiguous.
        """
        dataset = make_path_dependent_dataset(n_quarters=140, seed=20261008)
        rmse = _oos_rmse(
            dataset,
            {
                "benchmark": lambda: SmoothingRegressionNowcaster(n_lags=1),
                "signature_l2": lambda: SignatureNowcaster(level=2),
            },
        )
        for horizon in (1, 2):
            ratio = rmse["signature_l2"][horizon] / rmse["benchmark"][horizon]
            assert ratio < 0.80, (
                f"h={horizon}: signature RMSE ratio {ratio:.3f} vs benchmark "
                f"{rmse}"
            )

    def test_level_one_does_not_beat_the_benchmark_on_the_same_data(self):
        """Isolates the source of the gain: it is the level-2 terms, not signatures."""
        dataset = make_path_dependent_dataset(n_quarters=140, seed=20261008)
        rmse = _oos_rmse(
            dataset,
            {
                "benchmark": lambda: SmoothingRegressionNowcaster(n_lags=1),
                "signature_l1": lambda: SignatureNowcaster(
                    level=1, alpha=0.0, level_channel="simple"
                ),
            },
            horizons=(1,),
        )
        ratio = rmse["signature_l1"][1] / rmse["benchmark"][1]
        assert 0.95 < ratio < 1.05, f"level 1 should be at parity, got {ratio:.3f}"

    def test_no_material_edge_on_the_linear_dgp(self):
        """The overfitting check: no path signal to find, so no edge to claim.

        A small degradation is the expected and correct outcome — the extra
        features are noise, and regularisation limits but does not eliminate the
        cost of carrying them.
        """
        dataset = make_linear_smoothing_dataset(
            n_quarters=140, revision_sd=0.0, seed=4242
        )
        rmse = _oos_rmse(
            dataset,
            {
                "benchmark": lambda: SmoothingRegressionNowcaster(n_lags=1),
                "signature_l2": lambda: SignatureNowcaster(level=2),
            },
        )
        for horizon in (1, 2):
            ratio = rmse["signature_l2"][horizon] / rmse["benchmark"][horizon]
            assert ratio > 0.95, (
                f"h={horizon}: signature claims a {1 - ratio:.1%} edge on a DGP "
                f"with no path signal, which would mean the test is unsound: {rmse}"
            )
            assert ratio < 1.25, (
                f"h={horizon}: ratio {ratio:.3f} — the cost of the extra features "
                "should be modest, not ruinous"
            )


# ──────────────────────────────────────────────────────────────────
# Model 3 — everything else
# ──────────────────────────────────────────────────────────────────


class TestSignatureNowcasterBehaviour:
    def test_satisfies_the_protocol(self):
        assert isinstance(SignatureNowcaster(), Nowcaster)

    def test_defaults_match_the_spec(self):
        model = SignatureNowcaster()
        assert model.level == 2
        assert model.lookback == 1
        assert model.keep_sigs == "linear"
        assert model.basepoint is True
        assert model.multiplier_terms == "none"
        assert model.l1_ratio == 0.0
        assert model.alpha is None, "alpha is the one hyperparameter selected by default"

    def test_only_alpha_is_selected_by_default(self, motivating_dataset):
        """A four-point grid, per DESIGN_NOTES Risk 3 on selection bias."""
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SignatureNowcaster(), info)
        selection = model.fit_diagnostics["selection"]
        assert selection["selected_by"] == "inner_rolling_origin"
        assert selection["n_configurations"] == 4
        assert all("level=2" in key for key in selection["scores"])

    def test_pinned_configuration_skips_selection(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SignatureNowcaster(alpha=1e-3), info)
        assert model.fit_diagnostics["selection"]["selected_by"] == "user"
        assert model.fit_diagnostics["alpha"] == 1e-3

    def test_level_and_lookback_selection(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(
            SignatureNowcaster(
                level=None,
                level_candidates=(1, 2),
                lookback=None,
                lookback_candidates=(1, 2),
                alpha=1e-3,
            ),
            info,
        )
        assert model.fit_diagnostics["selection"]["n_configurations"] == 4
        assert model.fit_diagnostics["level"] in (1, 2)
        assert model.fit_diagnostics["lookback"] in (1, 2)

    def test_inner_selection_cannot_see_the_outer_quarter(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SignatureNowcaster(), info)
        unpublished = {str(q) for q in info.unpublished_quarters()}
        for detail in model.fit_diagnostics["selection"]["detail"].values():
            if "selection_index" in detail:
                assert not unpublished.intersection(set(detail["selection_index"]))
        assert model.fit_diagnostics["training_quarters"][1] == "2026Q1"

    def test_reproducible_given_the_same_seed(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, first = _fit_predict(SignatureNowcaster(), info, seed=5)
        _, second = _fit_predict(SignatureNowcaster(), info, seed=5)
        for a, b in _paired(first, second):
            assert a.point == b.point
            np.testing.assert_array_equal(a.draws, b.draws)

    def test_horizon_two_is_wider(self, motivating_dataset):
        info = _info(motivating_dataset)
        _, results = _fit_predict(SignatureNowcaster(), info, n_draws=20_000)
        assert results[1].draw_std > results[0].draw_std

    def test_predictions_are_bit_for_bit_stable_against_future_data(
        self, motivating_dataset
    ):
        """Spec test 1 for Model 3 — including the path, the scaler and selection."""
        base_info = _info(motivating_dataset)
        _, base = _fit_predict(SignatureNowcaster(), base_info)

        perturbed_factors = motivating_dataset.daily_factors.copy()
        perturbed_factors.loc[perturbed_factors.index > AS_OF] += 0.25
        perturbed_vintages = motivating_dataset.vintages.copy()
        perturbed_vintages.loc[
            perturbed_vintages["release_date"] > AS_OF, "value"
        ] += 0.5
        other_info = _info(
            motivating_dataset,
            public_factors=perturbed_factors,
            vintages=perturbed_vintages,
        )
        _, other = _fit_predict(SignatureNowcaster(), other_info)
        for a, b in _paired(base, other):
            assert a.point == b.point
            np.testing.assert_array_equal(a.draws, b.draws)

    def test_target_true_is_refused_with_a_pointer(self, motivating_dataset):
        info = _info(motivating_dataset)
        model = SignatureNowcaster(target="true")
        with pytest.raises(ValueError, match="supports target='reported' only"):
            model.fit(info)

    def test_bad_interval_method(self, motivating_dataset):
        info = _info(motivating_dataset)
        with pytest.raises(ValueError, match="unknown interval_method"):
            SignatureNowcaster(interval_method="parametric").fit(info)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"level": 0}, "candidates must be >= 1"),
            ({"level": MAX_SIGNATURE_LEVEL + 1}, "level must be <="),
            ({"lookback": 0}, "candidates must be >= 1"),
            ({"keep_sigs": "some"}, "keep_sigs must be"),
            ({"alpha": -1.0}, "alpha must be >= 0"),
            ({"l1_ratio": 1.5}, "l1_ratio must be in"),
        ],
    )
    def test_grid_validation(self, motivating_dataset, kwargs, match):
        info = _info(motivating_dataset)
        with pytest.raises(ValueError, match=match):
            SignatureNowcaster(**kwargs).fit(info)

    def test_lookback_four_uses_a_longer_window(self, motivating_dataset):
        info = _info(motivating_dataset)
        short, _ = _fit_predict(SignatureNowcaster(lookback=1, alpha=1e-3), info)
        long, _ = _fit_predict(SignatureNowcaster(lookback=4, alpha=1e-3), info)
        assert long.fit_diagnostics["lookback"] == 4
        # A 4-quarter window needs three more quarters of factor history, so the
        # first few rows drop out.
        assert long.fit_diagnostics["n_train"] < short.fit_diagnostics["n_train"]

    def test_monthly_path_frequency(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, results = _fit_predict(
            SignatureNowcaster(path_frequency="monthly", alpha=1e-3), info
        )
        assert model.fit_diagnostics["path_frequency"] == "monthly"
        assert np.isfinite(results[0].point)

    def test_keep_sigs_all_adds_the_squared_term(self, motivating_dataset):
        info = _info(motivating_dataset)
        linear, _ = _fit_predict(
            SignatureNowcaster(keep_sigs="linear", alpha=1e-3), info
        )
        every, _ = _fit_predict(SignatureNowcaster(keep_sigs="all", alpha=1e-3), info)
        assert every.fit_diagnostics["n_params"] == (
            linear.fit_diagnostics["n_params"] + 1
        )
        assert "sig[equity_market,equity_market]" in every.fit_diagnostics["features"]

    def test_two_factor_panel(self, linear_dataset_two_factor):
        as_of = linear_dataset_two_factor.reported.index[-1].end_time + pd.Timedelta(
            days=200
        )
        info = _info(linear_dataset_two_factor, as_of=as_of)
        model, results = _fit_predict(SignatureNowcaster(alpha=1e-3), info)
        assert model.fit_diagnostics["channel_names"] == (
            "time",
            "equity_market",
            "credit_proxy",
        )
        assert model.fit_diagnostics["feature_budget"]["n_features"] == 10
        assert np.isfinite(results[0].point)

    def test_pca_channels_and_the_recorded_caveat(self, linear_dataset_two_factor):
        as_of = linear_dataset_two_factor.reported.index[-1].end_time + pd.Timedelta(
            days=200
        )
        info = _info(linear_dataset_two_factor, as_of=as_of)
        model, results = _fit_predict(
            SignatureNowcaster(n_components=1, alpha=1e-3), info
        )
        assert model.fit_diagnostics["channel_names"] == ("time", "pc1")
        assert model.fit_diagnostics["pca_fit_scope"] == "information_set"
        assert np.isfinite(results[0].point)

    def test_early_signal_channel_is_used(self, early_signal_dataset):
        as_of = early_signal_dataset.reported.index[-1].end_time + pd.Timedelta(
            days=200
        )
        info = _info(
            early_signal_dataset,
            as_of=as_of,
            early_signals=early_signal_dataset.early_signals,
        )
        with_signal, _ = _fit_predict(SignatureNowcaster(alpha=1e-3), info)
        without, _ = _fit_predict(
            SignatureNowcaster(alpha=1e-3, include_early_signals=False), info
        )
        assert with_signal.fit_diagnostics["channel_names"] == (
            "time",
            "equity_market",
            "listed_pe_nav",
        )
        assert without.fit_diagnostics["channel_names"] == ("time", "equity_market")
        assert (
            with_signal.fit_diagnostics["n_params"]
            > without.fit_diagnostics["n_params"]
        )

    def test_rectilinear_paths_fit(self, motivating_dataset):
        info = _info(motivating_dataset)
        model, results = _fit_predict(
            SignatureNowcaster(missing="rectilinear", alpha=1e-3), info
        )
        assert model.fit_diagnostics["missing"] == "rectilinear"
        assert np.isfinite(results[0].point)


class TestSignatureMultiplierTerms:
    r"""The reference paper's Eq. 11, and why its literal form is degenerate here."""

    def test_time_multipliers_warn_that_they_are_collinear(self, motivating_dataset):
        """Time-only terms are constants, so interacting them with s_lag1 adds nothing.

        Time is rescaled to ``[0, 1]`` over every window, so ``sig[time] = 1`` and
        ``sig[time,time] = 1/2`` for every row. ``s_lag1 * constant`` is a rescaled
        copy of ``s_lag1``: exactly collinear. Implemented for fidelity to the
        spec, with a warning rather than a silent rank deficiency.
        """
        info = _info(motivating_dataset)
        model = SignatureNowcaster(multiplier_terms="time", alpha=1e-3)
        with pytest.warns(SignatureNowcasterWarning, match="exactly collinear"):
            model.fit(info)
        results = model.predict(
            info, info.unpublished_quarters(), rng=np.random.default_rng(0)
        )
        assert np.isfinite(results[0].point)

    def test_time_multipliers_change_nothing_material(self, motivating_dataset):
        """The degeneracy, measured: predictions barely move."""
        info = _info(motivating_dataset)
        _, plain = _fit_predict(SignatureNowcaster(alpha=1e-3), info)
        model = SignatureNowcaster(multiplier_terms="time", alpha=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(info)
            degenerate = model.predict(
                info, info.unpublished_quarters(), rng=np.random.default_rng(7)
            )
        for a, b in _paired(plain, degenerate):
            assert a.point == pytest.approx(b.point, abs=5e-3)

    def test_all_multipliers_give_a_path_dependent_ar_coefficient(
        self, motivating_dataset
    ):
        """The non-degenerate version: s_lag1 interacted with the data terms."""
        info = _info(motivating_dataset)
        model, results = _fit_predict(
            SignatureNowcaster(multiplier_terms="all", alpha=1e-3), info
        )
        names = model.fit_diagnostics["features"]
        assert "s_lag1*sig[equity_market]" in names
        assert "s_lag1*sig[equity_market,time]" in names
        assert model.fit_diagnostics["n_params"] == 8  # 3 sig + lag + intercept + 3
        assert np.isfinite(results[0].point)


class TestSignatureEstimators:
    def test_default_path_is_pure_numpy(self, motivating_dataset):
        """So Model 3 fits without the ``nowcast`` extra installed."""
        info = _info(motivating_dataset)
        model, _ = _fit_predict(SignatureNowcaster(alpha=1e-3), info)
        assert model.fit_diagnostics["estimator"] == "numpy_ridge"

    def test_elasticnet_path_warns_about_penalising_the_lag(self, motivating_dataset):
        info = _info(motivating_dataset)
        model = SignatureNowcaster(alpha=1e-4, l1_ratio=0.5)
        with pytest.warns(SignatureNowcasterWarning, match="penalises every coefficient"):
            model.fit(info)
        assert model.fit_diagnostics["estimator"] == "sklearn_elasticnet"
        results = model.predict(
            info, info.unpublished_quarters(), rng=np.random.default_rng(0)
        )
        assert np.isfinite(results[0].point)

    def test_ridge_shrinks_signature_coefficients_toward_zero(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        weak, _ = _fit_predict(SignatureNowcaster(alpha=1e-6), info)
        strong, _ = _fit_predict(SignatureNowcaster(alpha=1.0), info)
        weak_norm = np.linalg.norm(
            [v for k, v in weak.coefficients.items() if k.startswith("sig[")]
        )
        strong_norm = np.linalg.norm(
            [v for k, v in strong.coefficients.items() if k.startswith("sig[")]
        )
        assert strong_norm < weak_norm

    def test_ridge_does_not_shrink_the_lag_coefficient_to_zero(
        self, motivating_dataset
    ):
        """The penalty exemption: ``c`` is the smoothing parameter, not a nuisance."""
        info = _info(motivating_dataset)
        strong, _ = _fit_predict(SignatureNowcaster(alpha=100.0), info)
        # With the signature block crushed, the lag coefficient must survive.
        assert abs(strong.coefficients["s_lag1"]) > 0.5
        for name, value in strong.coefficients.items():
            if name.startswith("sig["):
                assert abs(value) < 1e-3

    def test_numpy_ridge_matches_sklearn_when_nothing_is_exempt(self):
        r"""Pins the :math:`n\alpha` scaling claim against scikit-learn's objective.

        ``_fit_selective_ridge`` adds ``n * alpha`` to the penalised diagonal.
        That scaling is what makes its ``alpha`` mean the same thing as
        scikit-learn's ``ElasticNet(l1_ratio=0)`` objective
        :math:`\tfrac{1}{2n}\lVert y-Xw\rVert^2 + \tfrac{\alpha}{2}\lVert
        w\rVert^2`, whose stationary condition is
        :math:`(X'X + n\alpha I)w = X'y`.

        The comparison is against ``Ridge(alpha=n*alpha)``, which solves exactly
        that system, rather than against ``ElasticNet`` itself: with
        ``l1_ratio=0`` sklearn's coordinate descent does not converge to a tight
        tolerance and says so. Ridge is the same objective by a direct solve.
        """
        sklearn_linear = pytest.importorskip("sklearn.linear_model")
        rng = np.random.default_rng(0)
        n, p = 60, 4
        features = rng.standard_normal((n, p))
        y = features @ np.array([1.0, -0.5, 0.25, 0.0]) + 0.1 * rng.standard_normal(n)
        alpha = 0.05

        design = np.hstack([np.ones((n, 1)), features])
        mine = _fit_selective_ridge(
            design, y, penalised=np.arange(1, p + 1), alpha=alpha
        )
        theirs = sklearn_linear.Ridge(
            alpha=alpha * n, fit_intercept=True, solver="cholesky"
        ).fit(features, y)
        np.testing.assert_allclose(mine[1:], theirs.coef_, atol=1e-10)
        assert mine[0] == pytest.approx(theirs.intercept_, abs=1e-10)

    def test_alpha_zero_is_exact_ols(self):
        rng = np.random.default_rng(1)
        n, p = 40, 3
        features = rng.standard_normal((n, p))
        y = features @ np.array([2.0, -1.0, 0.5]) + 0.01 * rng.standard_normal(n)
        design = np.hstack([np.ones((n, 1)), features])
        mine = _fit_selective_ridge(
            design, y, penalised=np.arange(1, p + 1), alpha=0.0
        )
        ols, *_ = np.linalg.lstsq(design, y, rcond=None)
        np.testing.assert_allclose(mine, ols, atol=1e-10)
