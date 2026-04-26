"""Tests for ``desmoothing.ma_glm``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.config import (
    InverseGammaPrior,
    NormalPrior,
)
from private_assets_frequency.core.protocols import DesmoothedResult, SmoothingModel
from private_assets_frequency.desmoothing.ma_glm import (
    MAGLMSmoother,
    _build_ma_matrix,
    _theta_valid,
)
from private_assets_frequency.tests.conftest import (
    MAqSyntheticDataset,
    make_ma_hf_dataset,
)
from private_assets_frequency.utils.returns import realised_volatility


# ──────────────────────────────────────────────────────────────────
# Default priors
# ──────────────────────────────────────────────────────────────────


def _default_priors():
    return {
        "beta": {
            "equity_market": NormalPrior(0.4, 0.2),
            "smb": NormalPrior(0.1, 0.15),
        },
        "alpha": NormalPrior(0.0, 0.03),
        "sigma_eps": InverseGammaPrior(3.0, 0.0008),  # E[σ²] = 0.0004 → σ ≈ 2 % monthly
    }


# ──────────────────────────────────────────────────────────────────
# Construction & helpers
# ──────────────────────────────────────────────────────────────────


class TestMAGLMConstruction:
    def test_default(self):
        sm = MAGLMSmoother()
        assert sm.q == 2
        assert sm.theta_grid.ndim == 2
        assert sm.theta_grid.shape[1] == 3

    def test_q3_grid_nonempty(self):
        sm = MAGLMSmoother(q=3, n_grid_per_axis=8)
        assert sm.theta_grid.shape[0] > 0
        assert sm.theta_grid.shape[1] == 4

    def test_q4_unsupported(self):
        with pytest.raises(NotImplementedError):
            MAGLMSmoother(q=4)

    def test_invalid_q(self):
        with pytest.raises(ValueError):
            MAGLMSmoother(q=0)

    def test_invalid_n_grid(self):
        with pytest.raises(ValueError):
            MAGLMSmoother(n_grid_per_axis=2)

    def test_invalid_floor(self):
        with pytest.raises(ValueError):
            MAGLMSmoother(theta_0_floor=1.5)

    def test_dirichlet_concentrations_validation(self):
        with pytest.raises(ValueError):
            MAGLMSmoother(q=2, dirichlet_concentrations=(1.0, 1.0))  # wrong length
        with pytest.raises(ValueError):
            MAGLMSmoother(q=2, dirichlet_concentrations=(1.0, 0.0, 1.0))  # zero entry

    def test_implements_protocol(self):
        assert isinstance(MAGLMSmoother(), SmoothingModel)


class TestThetaValidator:
    def test_valid_ordered(self):
        assert _theta_valid(np.array([0.6, 0.3, 0.1]), 0.2)

    def test_invalid_unordered(self):
        assert not _theta_valid(np.array([0.3, 0.6, 0.1]), 0.2)

    def test_below_floor(self):
        assert not _theta_valid(np.array([0.15, 0.5, 0.35]), 0.2)

    def test_negative_entry(self):
        assert not _theta_valid(np.array([0.7, 0.4, -0.1]), 0.2)

    def test_does_not_sum_to_one(self):
        assert not _theta_valid(np.array([0.5, 0.3, 0.1]), 0.2)


class TestMaMatrixBuilder:
    def test_lower_triangular_toeplitz_structure(self):
        theta = np.array([0.6, 0.3, 0.1])
        n = 5
        T = _build_ma_matrix(theta, n)
        assert T.shape == (n, n)
        # Diagonal is θ_0
        np.testing.assert_array_equal(np.diag(T), [0.6] * n)
        # Sub-diagonal 1 is θ_1 (length n-1)
        np.testing.assert_array_equal(np.diag(T, -1), [0.3] * (n - 1))
        # Sub-diagonal 2 is θ_2 (length n-2)
        np.testing.assert_array_equal(np.diag(T, -2), [0.1] * (n - 2))
        # Upper triangular zeros
        assert np.all(np.triu(T, k=1) == 0.0)


# ──────────────────────────────────────────────────────────────────
# desmooth (deterministic round-trip)
# ──────────────────────────────────────────────────────────────────


class TestMADesmoothFunction:
    def test_round_trip_known_theta(self):
        rng = np.random.default_rng(0)
        n = 100
        r_true = rng.normal(0.005, 0.02, size=n)
        theta = np.array([0.6, 0.3, 0.1])
        # Forward MA(q): s_t = Σ θ_j r_{t-j} (truncated at boundaries)
        s = np.zeros(n)
        for j, w in enumerate(theta):
            if j == 0:
                s += w * r_true
            else:
                s[j:] += w * r_true[:-j]
        # Inverse via MA smoother's desmooth
        r_back = MAGLMSmoother(q=2).desmooth(s, theta)
        np.testing.assert_allclose(r_back, r_true, atol=1e-10)

    def test_invalid_theta_size(self):
        with pytest.raises(ValueError):
            MAGLMSmoother(q=2).desmooth(np.zeros(10), np.array([0.5, 0.5]))

    def test_zero_theta_0_rejected(self):
        with pytest.raises(ValueError):
            MAGLMSmoother(q=2).desmooth(np.zeros(10), np.array([0.0, 0.5, 0.5]))


# ──────────────────────────────────────────────────────────────────
# Fit on default synthetic dataset (q=2)
# ──────────────────────────────────────────────────────────────────


class TestMAFitOnDefaultSynthetic:
    @pytest.fixture
    def fit_result(self, ma_hf_dataset: MAqSyntheticDataset):
        sm = MAGLMSmoother(q=2)
        result = sm.fit(
            ma_hf_dataset.observed_monthly,
            ma_hf_dataset.factor_monthly,
            _default_priors(),
        )
        return sm, result, ma_hf_dataset

    def test_returns_desmoothed_result(self, fit_result):
        _, result, _ = fit_result
        assert isinstance(result, DesmoothedResult)
        assert "theta" in result.smoothing_params
        theta_post = result.smoothing_params["theta"]
        assert theta_post.shape == (3,)
        # Sums to one
        assert abs(theta_post.sum() - 1.0) < 1e-6
        # Ordered
        assert (np.diff(theta_post) <= 1e-9).all()

    def test_theta_close_to_truth(self, fit_result):
        _, result, ds = fit_result
        theta_post = result.smoothing_params["theta"]
        # True θ = (0.6, 0.3, 0.1); recovered should be within 0.10 elementwise
        np.testing.assert_allclose(theta_post, ds.theta, atol=0.10)

    def test_beta_close_to_truth(self, fit_result):
        _, result, ds = fit_result
        beta_post = result.smoothing_params["beta"]
        for k, v_true in ds.beta.items():
            assert abs(beta_post[k] - v_true) < 0.20, (
                f"β[{k}] = {beta_post[k]:.3f} vs true {v_true:.3f}"
            )

    def test_desmoothed_vol_higher_than_observed(self, fit_result):
        _, result, ds = fit_result
        sigma_obs = realised_volatility(ds.observed_monthly)
        sigma_des = realised_volatility(result.true_returns)
        assert sigma_des > sigma_obs

    def test_no_warnings_for_well_identified_data(self, fit_result):
        _, result, _ = fit_result
        # Default theta=(0.6,0.3,0.1) has θ_0=0.6 and good condition number
        flags = result.diagnostics["flags"]
        assert not flags["theta_0_low"]
        assert not flags["condition_number_high"]

    def test_density_normalises(self, fit_result):
        _, result, _ = fit_result
        density = result.posterior_summary["theta"]["density"]
        assert abs(density.sum() - 1.0) < 1e-9


# ──────────────────────────────────────────────────────────────────
# Condition-number / θ_0-low warning behaviour (the central test)
# ──────────────────────────────────────────────────────────────────


class TestMACondNumberWarnings:
    def test_low_theta_0_triggers_warning(self):
        """When posterior θ_0 < 0.3, the smoother must flag the cond / low-θ_0 warning."""
        # Generate q=3 data at the simplex floor θ_0 = 0.25 (equal weights). With low
        # noise + flat Dirichlet prior the posterior peaks at exactly that grid point.
        ds = make_ma_hf_dataset(
            n_years=20,
            theta=(0.25, 0.25, 0.25, 0.25),
            beta={"equity_market": 0.4, "smb": 0.1},
            sigma_eps_monthly=0.003,  # low noise so the data is informative
            rng=42,
        )
        sm = MAGLMSmoother(
            q=3,
            n_grid_per_axis=10,
            dirichlet_concentrations=(1.0, 1.0, 1.0, 1.0),
        )
        result = sm.fit(ds.observed_monthly, ds.factor_monthly, _default_priors())

        theta_post = result.smoothing_params["theta"]
        assert theta_post[0] < 0.3, (
            f"posterior θ_0 = {theta_post[0]:.3f} not below 0.3 (test design fault)"
        )

        flags = result.diagnostics["flags"]
        warns = result.diagnostics["warnings"]
        assert flags["theta_0_low"], "theta_0_low flag did not trip"
        # cond(Θ) explodes for small θ_0 — the matrix is lower-triangular Toeplitz with
        # tiny diagonal relative to off-diagonals, so its inverse amplifies noise.
        assert flags["condition_number_high"], (
            f"condition number {result.diagnostics['condition_number']:.1f} "
            f"did not exceed threshold; θ_0 = {theta_post[0]:.3f}"
        )
        joined = "\n".join(warns).lower()
        assert "θ_0" in joined or "theta_0" in joined or "condition number" in joined


# ──────────────────────────────────────────────────────────────────
# Posterior sampling
# ──────────────────────────────────────────────────────────────────


class TestMASamplePosterior:
    def test_sample_after_fit(self, ma_hf_dataset: MAqSyntheticDataset):
        sm = MAGLMSmoother(q=2)
        sm.fit(
            ma_hf_dataset.observed_monthly,
            ma_hf_dataset.factor_monthly,
            _default_priors(),
        )
        samples = sm.sample_posterior(n_samples=200, rng=42)
        assert samples["theta"].shape == (200, 3)
        # Each sampled θ is on the simplex, ordered, within floor
        assert np.allclose(samples["theta"].sum(axis=1), 1.0)
        assert (np.diff(samples["theta"], axis=1) <= 1e-9).all()
        assert (samples["theta"][:, 0] >= sm.theta_0_floor - 1e-9).all()
        assert (samples["sigma_eps"] > 0.0).all()

    def test_sample_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            MAGLMSmoother().sample_posterior(n_samples=10, rng=42)


# ──────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────


class TestMAValidation:
    def test_rejects_nan_returns(self, ma_hf_dataset):
        s = ma_hf_dataset.observed_monthly.copy()
        s.iloc[5] = np.nan
        with pytest.raises(ValueError):
            MAGLMSmoother(q=2).fit(s, ma_hf_dataset.factor_monthly, _default_priors())

    def test_rejects_short_series(self):
        idx = pd.date_range("2020-01-31", periods=8, freq="M")
        s = pd.Series(np.zeros(8), index=idx)
        f = pd.DataFrame({"f": np.zeros(8)}, index=idx)
        priors = {
            "beta": {"f": NormalPrior(0.4, 0.2)},
            "alpha": NormalPrior(0.0, 0.03),
            "sigma_eps": InverseGammaPrior(3.0, 0.0008),
        }
        with pytest.raises(ValueError, match="observations"):
            MAGLMSmoother(q=2).fit(s, f, priors)

    def test_missing_factor_prior(self, ma_hf_dataset):
        priors = _default_priors()
        priors["beta"] = {"equity_market": NormalPrior(0.4, 0.2)}  # missing smb
        with pytest.raises(ValueError, match="missing entries"):
            MAGLMSmoother(q=2).fit(
                ma_hf_dataset.observed_monthly,
                ma_hf_dataset.factor_monthly,
                priors,
            )

    def test_invalid_sigma_prior(self, ma_hf_dataset):
        priors = _default_priors()
        priors["sigma_eps"] = InverseGammaPrior(0.5, 0.001)
        with pytest.raises(ValueError, match="alpha"):
            MAGLMSmoother(q=2).fit(
                ma_hf_dataset.observed_monthly,
                ma_hf_dataset.factor_monthly,
                priors,
            )
