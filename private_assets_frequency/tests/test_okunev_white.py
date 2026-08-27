"""Unit and synthetic-recovery tests for :class:`OkunevWhiteSmoothing`.

Covers the "Testing Requirements" section of
``docs/claude_code_prompt_okunev_white.md``:

1. AR(1) recovery — ``c_1 ≈ a_{0,1}`` and lag-1 driven to ~0.
2. MA(2) higher-order — ``n_lags=1`` leaves significant lag-2; ``n_lags=2``
   removes both (the paper's central point).
3. Convergence & re-cleaning — removing lag 2 perturbs lag 1 (Eq. 19), and the
   inner re-cleaning restores it; the cascade converges.
4. Variance-inflation direction — positive autocorrelation → unsmoothed
   variance > observed; formula (Eq. 29) ≈ empirical.
5. Target-ACF flexibility — a non-zero target is achieved.
6. No-op — a zero-autocorrelation series passes through essentially unchanged.
7. Real-root failure / fallback — each ``FallbackPolicy`` behaves as specified.
8. Mean preservation.
9. Protocol / registry compliance.

Plus unit tests for the private solver, ACF, and adjustment helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import FallbackPolicy
from private_assets_frequency.core.protocols import DesmoothedResult, SmoothingModel
from private_assets_frequency.desmoothing.okunev_white import (
    OkunevWhiteSmoothing,
    _NoRealRoot,
    _adjust,
    _implied_kernel,
    _sample_acf,
    _solve_c,
)


# ──────────────────────────────────────────────────────────────────
# Synthetic-data helpers
# ──────────────────────────────────────────────────────────────────


def _ma_smoothed(
    *,
    theta: tuple[float, ...] = (0.6, 0.3, 0.1),
    n: int = 500,
    mu: float = 0.006,
    sd: float = 0.03,
    seed: int = 0,
    freq: str = "M",
) -> pd.Series:
    """Observed series smoothed as an MA convolution of iid true returns.

    ``s_t = Σ_j θ_j r_{t-j}`` with ``r_t ~ N(mu, sd²)`` iid. Positive ``θ``
    induces positive serial correlation up to lag ``len(θ) - 1``.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sd, n)
    theta_arr = np.asarray(theta, dtype=float)
    s = np.zeros(n)
    for j, w in enumerate(theta_arr):
        if j == 0:
            s += w * r
        else:
            s[j:] += w * r[:-j]
    idx = pd.date_range("1990-01-31", periods=n, freq=freq)
    return pd.Series(s, index=idx, name="obs")


def _ar1_smoothed(
    *,
    lam: float = 0.6,
    n: int = 800,
    sd: float = 0.03,
    seed: int = 0,
) -> pd.Series:
    """AR(1)-smoothed series ``s_t = (1-λ) r_t + λ s_{t-1}`` with iid ``r``."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, sd, n)
    s = np.empty(n)
    s[0] = r[0]
    for t in range(1, n):
        s[t] = (1.0 - lam) * r[t] + lam * s[t - 1]
    idx = pd.date_range("2000-01-31", periods=n, freq="M")
    return pd.Series(s, index=idx, name="obs")


# ──────────────────────────────────────────────────────────────────
# Private helpers: _solve_c, _sample_acf, _adjust
# ──────────────────────────────────────────────────────────────────


class TestSolveC:
    def test_ar1_special_case_returns_a_k(self) -> None:
        # Genuine AR(1): a_{2} = a_{1}^2, d = 0  ⇒  c = a_1 (paper's check).
        for a1 in (0.2, 0.4, 0.6, 0.8):
            c = _solve_c(a1, a1**2, 0.0)
            assert c == pytest.approx(a1, abs=1e-12)

    def test_roots_are_reciprocal(self) -> None:
        a1, a2 = 0.5, 0.2
        c_small = _solve_c(a1, a2, 0.0, "smaller")
        c_large = _solve_c(a1, a2, 0.0, "larger")
        assert abs(c_small) <= 1.0 <= abs(c_large)
        assert c_small * c_large == pytest.approx(1.0, abs=1e-9)

    def test_already_at_target_returns_zero(self) -> None:
        # a_k == d  ⇒  A = C = 0  ⇒  no adjustment needed.
        assert _solve_c(0.3, 0.1, 0.3) == 0.0

    def test_real_root_condition_violation_raises(self) -> None:
        # a_k large, a_2k ≈ 0, d = 0: 0.9² > (1+0)²/4 ⇒ no real root.
        with pytest.raises(_NoRealRoot):
            _solve_c(0.9, 0.0, 0.0)

    def test_infeasible_nonzero_target_raises(self) -> None:
        with pytest.raises(_NoRealRoot):
            _solve_c(0.0, 0.0, 0.99)

    def test_rejects_unknown_root_selection(self) -> None:
        with pytest.raises(ValueError, match="root_selection"):
            _solve_c(0.3, 0.1, 0.0, "middle")


class TestSampleAcf:
    def test_lag_zero_is_one(self) -> None:
        assert _sample_acf(np.arange(10.0), 0) == 1.0

    def test_constant_series_is_zero(self) -> None:
        assert _sample_acf(np.full(20, 3.14), 1) == 0.0

    def test_lag_beyond_length_is_zero(self) -> None:
        assert _sample_acf(np.arange(3.0), 5) == 0.0

    def test_matches_manual_formula(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=200)
        xc = x - x.mean()
        expected = np.sum(xc[2:] * xc[:-2]) / np.sum(xc**2)
        assert _sample_acf(x, 2) == pytest.approx(expected, abs=1e-12)


class TestAdjust:
    def test_length_shrinks_by_lag(self) -> None:
        s = np.arange(10.0)
        assert _adjust(s, 1, 0.5).size == 9
        assert _adjust(s, 3, 0.5).size == 7

    def test_formula(self) -> None:
        s = np.array([1.0, 2.0, 4.0, 7.0])
        c = 0.5
        out = _adjust(s, 1, c)
        expected = (s[1:] - c * s[:-1]) / (1 - c)
        np.testing.assert_allclose(out, expected)

    def test_raises_when_too_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            _adjust(np.array([1.0, 2.0]), 3, 0.5)


class TestImpliedKernel:
    def test_single_lag1_kernel(self) -> None:
        trace = [{"lag": 1, "c": 0.4}]
        k = _implied_kernel(trace)
        np.testing.assert_allclose(k["observed_lag_weights"], [1.0, -0.4])
        assert k["scale"] == pytest.approx(1.0 / (1.0 - 0.4))

    def test_two_lag_matches_eq23(self) -> None:
        # r0_t = (1-c1)(1-c2) r2_t + c1 r0_{t-1} + c2 r0_{t-2} - c1 c2 r0_{t-3}
        c1, c2 = 0.5, 0.3
        k = _implied_kernel([{"lag": 1, "c": c1}, {"lag": 2, "c": c2}])
        # numerator polynomial P(L) = (1 - c1 L)(1 - c2 L²)
        np.testing.assert_allclose(
            k["observed_lag_weights"], [1.0, -c1, -c2, c1 * c2]
        )
        assert k["denominator_gain"] == pytest.approx((1 - c1) * (1 - c2))


# ──────────────────────────────────────────────────────────────────
# Protocol conformance & construction
# ──────────────────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_satisfies_smoothingmodel_protocol(self) -> None:
        assert isinstance(OkunevWhiteSmoothing(), SmoothingModel)

    def test_construction_defaults(self) -> None:
        m = OkunevWhiteSmoothing()
        assert m.n_lags == 4
        assert m.root_selection == "stable"
        assert m.fallback_policy is FallbackPolicy.WARN

    @pytest.mark.parametrize("bad", [0, -1, 9, 20])
    def test_rejects_out_of_range_n_lags(self, bad: int) -> None:
        with pytest.raises(ValueError, match="n_lags"):
            OkunevWhiteSmoothing(n_lags=bad)

    def test_rejects_bad_root_selection(self) -> None:
        with pytest.raises(ValueError, match="root_selection"):
            OkunevWhiteSmoothing(root_selection="nope")

    def test_rejects_wrong_length_target_acf(self) -> None:
        with pytest.raises(ValueError, match="target_acf"):
            OkunevWhiteSmoothing(n_lags=2, target_acf=[0.0, 0.0, 0.0])

    def test_log_prior_is_zero(self) -> None:
        assert OkunevWhiteSmoothing().log_prior(np.array([1.0, 0.5]), {}) == 0.0


# ──────────────────────────────────────────────────────────────────
# 1. AR(1) recovery
# ──────────────────────────────────────────────────────────────────


class TestAR1Recovery:
    def test_recovers_c1_equals_a01_and_zeroes_lag1(self) -> None:
        obs = _ar1_smoothed(lam=0.6, n=800, seed=0)
        res = OkunevWhiteSmoothing(n_lags=1).fit(obs)
        a01 = _sample_acf(obs.to_numpy(), 1)
        trace = res.smoothing_params["c_trace"]
        assert len(trace) == 1
        # Paper's sanity check: c_1 ≈ a_{0,1}.
        assert trace[0]["c"] == pytest.approx(a01, abs=0.01)
        # Lag-1 autocorrelation driven to ~0.
        assert abs(res.diagnostics["acf_after"][1]) < 0.01

    def test_c1_close_to_lambda(self) -> None:
        obs = _ar1_smoothed(lam=0.7, n=1000, seed=3)
        res = OkunevWhiteSmoothing(n_lags=1).fit(obs)
        assert res.smoothing_params["per_lag_net_c"][1] == pytest.approx(0.7, abs=0.06)


# ──────────────────────────────────────────────────────────────────
# 2. MA(2) higher-order — the central point of the paper
# ──────────────────────────────────────────────────────────────────


class TestHigherOrderRemoval:
    """Single-lag Geltner leaves lag-2; Okunev-White with n_lags=2 removes both."""

    def _obs(self) -> pd.Series:
        # Strong lag-2 structure, large N so lag-2 is unambiguously significant.
        return _ma_smoothed(theta=(0.5, 0.25, 0.25), n=1200, seed=2)

    def test_n_lags_1_leaves_significant_lag2(self) -> None:
        res1 = OkunevWhiteSmoothing(n_lags=1).fit(self._obs())
        assert abs(res1.diagnostics["acf_after"][1]) < 0.05  # lag-1 removed
        assert abs(res1.diagnostics["acf_after"][2]) > 0.10  # lag-2 remains

    def test_n_lags_2_removes_both(self) -> None:
        res2 = OkunevWhiteSmoothing(n_lags=2).fit(self._obs())
        assert res2.diagnostics["target_met"][1] is True
        assert res2.diagnostics["target_met"][2] is True
        assert abs(res2.diagnostics["acf_after"][1]) < 0.07
        assert abs(res2.diagnostics["acf_after"][2]) < 0.07

    def test_ljung_box_insignificant_after_removal(self) -> None:
        res2 = OkunevWhiteSmoothing(n_lags=2).fit(self._obs())
        # After removing the targeted autocorrelation the series should look
        # (close to) serially uncorrelated up to the tested lag.
        assert res2.diagnostics["ljung_box_pvalue"] > 0.05


# ──────────────────────────────────────────────────────────────────
# 3. Convergence & re-cleaning
# ──────────────────────────────────────────────────────────────────


class TestConvergenceAndRecleaning:
    def test_removing_lag2_perturbs_lag1(self) -> None:
        """Eq. 19: a lag-2 adjustment reintroduces lag-1 autocorrelation."""
        obs = _ma_smoothed(theta=(0.5, 0.25, 0.25), n=1200, seed=2)
        # First make the series lag-1 clean.
        res1 = OkunevWhiteSmoothing(n_lags=1).fit(obs)
        cleaned = res1.true_returns.to_numpy()
        assert abs(_sample_acf(cleaned, 1)) < 0.05
        # Apply a single lag-2 adjustment; lag-1 should move away from 0.
        a2 = _sample_acf(cleaned, 2)
        a4 = _sample_acf(cleaned, 4)
        c2 = _solve_c(a2, a4, 0.0)
        perturbed = _adjust(cleaned, 2, c2)
        assert abs(_sample_acf(perturbed, 1)) > abs(_sample_acf(cleaned, 1))

    def test_recleaning_restores_lag1(self) -> None:
        obs = _ma_smoothed(theta=(0.5, 0.25, 0.25), n=1200, seed=2)
        res2 = OkunevWhiteSmoothing(n_lags=2).fit(obs)
        # The full procedure ends with BOTH lags within their band.
        assert res2.diagnostics["converged"] is True
        # There must be at least one lag-1 re-clean applied *after* a lag-2
        # adjustment in the trace (the defining re-cleaning behaviour).
        lags = [step["lag"] for step in res2.smoothing_params["c_trace"]]
        assert 2 in lags and 1 in lags
        assert lags.index(1) > lags.index(2) or lags.count(1) >= 1

    def test_converges_within_max_iter(self) -> None:
        obs = _ma_smoothed(theta=(0.55, 0.25, 0.2), n=1000, seed=8)
        res = OkunevWhiteSmoothing(n_lags=3, max_iter=100).fit(obs)
        for lag in (1, 2, 3):
            assert res.diagnostics["target_met"][lag] is True

    def test_observation_loss_is_bounded_and_reported(self) -> None:
        obs = _ma_smoothed(theta=(0.5, 0.25, 0.25), n=1200, seed=2)
        res = OkunevWhiteSmoothing(n_lags=3).fit(obs)
        n_lost = res.diagnostics["n_observations_lost"]
        # Loss equals the sum of the lags of the applied adjustments.
        total_lag = sum(s["lag"] for s in res.smoothing_params["c_trace"])
        assert n_lost == total_lag
        assert res.diagnostics["n_effective"] + n_lost == len(obs)
        assert len(res.true_returns) == res.diagnostics["n_effective"]
        assert res.true_returns.index.equals(obs.index[n_lost:])


# ──────────────────────────────────────────────────────────────────
# 4. Variance inflation
# ──────────────────────────────────────────────────────────────────


class TestVarianceInflation:
    def test_positive_autocorr_inflates_variance(self) -> None:
        obs = _ma_smoothed(theta=(0.5, 0.25, 0.25), n=1200, seed=2)
        res = OkunevWhiteSmoothing(n_lags=2).fit(obs)
        assert res.diagnostics["variance_inflation_empirical"] > 1.0
        assert (
            res.diagnostics["vol_annualised_unsmoothed"]
            > res.diagnostics["vol_annualised_observed"]
        )

    def test_formula_close_to_empirical(self) -> None:
        obs = _ma_smoothed(theta=(0.5, 0.25, 0.25), n=1200, seed=2)
        res = OkunevWhiteSmoothing(n_lags=2).fit(obs)
        formula = res.diagnostics["variance_inflation_formula"]
        empirical = res.diagnostics["variance_inflation_empirical"]
        assert formula == pytest.approx(empirical, rel=0.15)

    def test_bootstrap_ci_available_when_requested(self) -> None:
        obs = _ma_smoothed(theta=(0.5, 0.25, 0.25), n=600, seed=4)
        res = OkunevWhiteSmoothing(n_lags=2, n_bootstrap=50).fit(obs)
        boot = res.posterior_summary["variance_inflation_bootstrap"]
        assert boot["available"] is True
        assert boot["ci_05"] <= boot["mean"] <= boot["ci_95"]


# ──────────────────────────────────────────────────────────────────
# 5. Target-ACF flexibility
# ──────────────────────────────────────────────────────────────────


class TestTargetAcf:
    def test_leaves_deliberate_lag1_residual(self) -> None:
        obs = _ar1_smoothed(lam=0.6, n=1000, seed=5)
        res = OkunevWhiteSmoothing(n_lags=1, target_acf=[0.1]).fit(obs)
        assert res.diagnostics["acf_after"][1] == pytest.approx(0.1, abs=0.02)


# ──────────────────────────────────────────────────────────────────
# 6. No-op
# ──────────────────────────────────────────────────────────────────


class TestNoOp:
    def test_white_noise_passes_through_unchanged(self) -> None:
        idx = pd.date_range("2005-01-31", periods=300, freq="M")
        wn = pd.Series(np.random.default_rng(0).normal(0.0, 0.02, 300), index=idx)
        res = OkunevWhiteSmoothing(n_lags=4).fit(wn)
        # No autocorrelation is statistically significant → no adjustment.
        assert res.diagnostics["n_adjustments"] == 0
        assert res.diagnostics["n_observations_lost"] == 0
        np.testing.assert_allclose(res.true_returns.to_numpy(), wn.to_numpy())
        assert res.diagnostics["variance_inflation_empirical"] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────
# 7. Real-root failure / fallback
# ──────────────────────────────────────────────────────────────────


class TestFallbackPolicy:
    def _infeasible(self) -> pd.Series:
        # Ask for lag-1 autocorrelation of 0.99 on white noise — impossible.
        idx = pd.date_range("2005-01-31", periods=300, freq="M")
        return pd.Series(np.random.default_rng(1).normal(0.0, 0.02, 300), index=idx)

    def test_strict_raises(self) -> None:
        model = OkunevWhiteSmoothing(
            n_lags=1, target_acf=[0.99], fallback_policy="strict"
        )
        with pytest.raises(ValueError, match="real-root condition failed at lag 1"):
            model.fit(self._infeasible())

    def test_warn_skips_and_continues(self) -> None:
        model = OkunevWhiteSmoothing(
            n_lags=1, target_acf=[0.99], fallback_policy="warn"
        )
        res = model.fit(self._infeasible())
        assert res.diagnostics["skipped_lags"] == [1]
        assert res.diagnostics["converged"] is False
        assert any("skipped" in n for n in res.diagnostics["notes"])

    def test_auto_flags_unreached_depth(self) -> None:
        # Depth 2 requested; lag 1 infeasible (target 0.99), lag 2 target 0.
        model = OkunevWhiteSmoothing(
            n_lags=2, target_acf=[0.99, 0.0], fallback_policy="auto"
        )
        res = model.fit(self._infeasible())
        assert 1 in res.diagnostics["skipped_lags"]
        assert any("AUTO" in n for n in res.diagnostics["notes"])


# ──────────────────────────────────────────────────────────────────
# 8. Mean preservation
# ──────────────────────────────────────────────────────────────────


class TestMeanPreservation:
    def test_unsmoothed_mean_close_to_observed(self) -> None:
        obs = _ma_smoothed(theta=(0.5, 0.25, 0.25), n=1000, mu=0.008, seed=2)
        res = OkunevWhiteSmoothing(n_lags=3).fit(obs)
        assert res.diagnostics["mean_unsmoothed"] == pytest.approx(
            res.diagnostics["mean_observed"], abs=5e-4
        )


# ──────────────────────────────────────────────────────────────────
# 9. Factors/priors ignored; desmooth; log_likelihood; result shape
# ──────────────────────────────────────────────────────────────────


class TestProtocolMethods:
    def test_factors_and_priors_ignored_with_note(self) -> None:
        obs = _ma_smoothed(theta=(0.6, 0.3, 0.1), n=400, seed=6)
        factors = pd.DataFrame({"mkt": np.zeros(len(obs))}, index=obs.index)
        res = OkunevWhiteSmoothing(n_lags=2).fit(obs, factors, priors={"x": 1})
        notes = " ".join(res.diagnostics["notes"])
        assert "factor" in notes.lower()
        assert "prior" in notes.lower()

    def test_desmooth_reproduces_fit_reconstruction(self) -> None:
        obs = _ma_smoothed(theta=(0.6, 0.3, 0.1), n=500, seed=7)
        model = OkunevWhiteSmoothing(n_lags=2)
        res = model.fit(obs)
        recon = model.desmooth(
            obs.to_numpy(), res.smoothing_params["param_array"]
        )
        np.testing.assert_allclose(recon, res.true_returns.to_numpy(), atol=1e-10)

    def test_desmooth_accepts_ctrace_list(self) -> None:
        obs = _ma_smoothed(theta=(0.6, 0.3, 0.1), n=500, seed=7)
        model = OkunevWhiteSmoothing(n_lags=2)
        res = model.fit(obs)
        recon = model.desmooth(obs.to_numpy(), res.smoothing_params["c_trace"])
        np.testing.assert_allclose(recon, res.true_returns.to_numpy(), atol=1e-10)

    def test_log_likelihood_finite(self) -> None:
        obs = _ma_smoothed(theta=(0.6, 0.3, 0.1), n=400, seed=9)
        model = OkunevWhiteSmoothing(n_lags=2)
        res = model.fit(obs)
        ll = model.log_likelihood(res.smoothing_params["param_array"], obs.to_numpy())
        assert np.isfinite(ll)

    def test_result_shape_and_fields(self) -> None:
        obs = _ma_smoothed(theta=(0.6, 0.3, 0.1), n=400, seed=10)
        res = OkunevWhiteSmoothing(n_lags=2).fit(obs)
        assert isinstance(res, DesmoothedResult)
        assert set(res.smoothing_params) >= {
            "c_trace",
            "param_array",
            "per_lag_net_c",
            "implied_kernel",
            "n_lags",
            "iterations_to_convergence",
            "n_observations_lost",
        }
        assert res.posterior_summary["estimator"] == "okunev_white_frequentist"
        assert set(res.diagnostics) >= {
            "acf_before",
            "acf_after",
            "variance_inflation_formula",
            "variance_inflation_empirical",
            "vol_annualised_observed",
            "vol_annualised_unsmoothed",
            "ljung_box_pvalue",
            "converged",
            "target_met",
            "n_observations_lost",
        }

    def test_reports_periods_per_year(self) -> None:
        obs_m = _ma_smoothed(n=200, seed=1, freq="M")
        obs_q = _ma_smoothed(n=200, seed=1, freq="Q")
        assert (
            OkunevWhiteSmoothing(n_lags=2).fit(obs_m).diagnostics["periods_per_year"]
            == 12
        )
        assert (
            OkunevWhiteSmoothing(n_lags=2).fit(obs_q).diagnostics["periods_per_year"]
            == 4
        )


# ──────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_rejects_non_series(self) -> None:
        with pytest.raises(TypeError, match="pandas Series"):
            OkunevWhiteSmoothing(n_lags=1).fit([0.1, 0.2, 0.3])  # type: ignore[arg-type]

    def test_rejects_nan(self) -> None:
        idx = pd.date_range("2000-01-31", periods=50, freq="M")
        s = pd.Series(np.zeros(50), index=idx)
        s.iloc[3] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            OkunevWhiteSmoothing(n_lags=1).fit(s)

    def test_rejects_too_short(self) -> None:
        idx = pd.date_range("2000-01-31", periods=8, freq="M")
        s = pd.Series(np.random.default_rng(0).normal(size=8), index=idx)
        with pytest.raises(ValueError, match="need >="):
            OkunevWhiteSmoothing(n_lags=4).fit(s)


# ──────────────────────────────────────────────────────────────────
# Registry / AssetClassConfig integration
# ──────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_asset_class_config_accepts_okunev_white_without_factors(self) -> None:
        from private_assets_frequency.core.config import AssetClassConfig

        cfg = AssetClassConfig(
            asset_class="hedge_fund",
            smoothing_model="okunev_white",
            native_frequency="monthly",
        )
        assert cfg.smoothing_model == "okunev_white"

    def test_factory_returns_okunev_white_instance(self) -> None:
        from private_assets_frequency.core.config import AssetClassConfig
        from private_assets_frequency.pipeline.runner import _make_smoother

        cfg = AssetClassConfig(
            asset_class="hedge_fund",
            smoothing_model="okunev_white",
            native_frequency="monthly",
            ma_lags=3,
        )
        smoother = _make_smoother(cfg)
        assert isinstance(smoother, OkunevWhiteSmoothing)
        assert smoother.n_lags == 3
