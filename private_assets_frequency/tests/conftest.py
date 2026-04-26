"""
Shared synthetic-data fixtures for the ``private_assets_frequency`` test suite.

Each generator produces a *known-truth* dataset so that round-trip tests can
verify that:

    1. desmoothing recovers the smoothing parameters within the posterior CI,
    2. factor decomposition recovers the betas, and
    3. temporal disaggregation reconstructs a high-frequency path consistent
       with the latent ground truth.

The three generators correspond to the three smoothing models specified in
``docs/claude_code_prompt_private_assets_frequency_v2.md``:

    * :func:`make_ar1_pe_dataset`        — AR(1) Bayesian (PE / infra / RE)
    * :func:`make_ma_hf_dataset`         — Getmansky-Lo-Makarov MA(q) (HF)
    * :func:`make_threshold_credit_dataset` — Threshold AR(1) (private credit)

All generators take a ``numpy.random.Generator`` (or seed) and return an
immutable dataclass bundling the *observed* (smoothed) series, the latent
*true* series, the factor returns, and the parameter values used to simulate.
Tests should pin a seed for determinism.

Notes
-----
The generators deliberately keep all return arithmetic in additive space.
For PE-magnitude returns the additive/multiplicative gap is 50+ bps per
quarter, so the AR(1) generator compounds monthly-to-quarterly via log-space
to match how ``utils.returns`` will aggregate inside the pipeline. The HF and
credit generators operate at native (monthly / quarterly) frequency where
additive smoothing is the model assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.utils.time_series import (
    MONTH_END_FREQ,
    QUARTER_END_FREQ,
)

# ──────────────────────────────────────────────────────────────────
# Index helpers
# ──────────────────────────────────────────────────────────────────


def _monthly_index(start: str, n_months: int) -> pd.DatetimeIndex:
    """Month-end DatetimeIndex of length ``n_months`` starting at ``start``."""
    return pd.date_range(start=start, periods=n_months, freq=MONTH_END_FREQ)


def _quarterly_index(start: str, n_quarters: int) -> pd.DatetimeIndex:
    """Quarter-end DatetimeIndex of length ``n_quarters`` starting at ``start``."""
    return pd.date_range(start=start, periods=n_quarters, freq=QUARTER_END_FREQ)


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


# ──────────────────────────────────────────────────────────────────
# AR(1) smoothed PE data (Geltner / MSCI methodology)
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AR1SyntheticDataset:
    """AR(1)-smoothed quarterly PE returns generated from a known monthly factor model.

    Attributes
    ----------
    observed_quarterly
        Smoothed quarterly returns. This is what the pipeline ingests.
    true_quarterly
        Unsmoothed quarterly returns (compounded from ``true_monthly``).
    true_monthly
        Latent monthly ground truth — the target for Stage 3 disaggregation.
    factor_monthly, factor_quarterly
        Factor return paths. Quarterly is compounded from monthly.
    lambda_
        True smoothing parameter applied at the quarterly level.
    beta
        Mapping ``{factor_name: true_beta}``.
    alpha
        True monthly alpha (per-period, additive).
    sigma_eps
        True monthly idiosyncratic volatility (additive).
    """

    observed_quarterly: pd.Series
    true_quarterly: pd.Series
    true_monthly: pd.Series
    factor_monthly: pd.DataFrame
    factor_quarterly: pd.DataFrame
    lambda_: float
    beta: dict[str, float]
    alpha: float
    sigma_eps: float


def make_ar1_pe_dataset(
    *,
    n_years: int = 30,
    lambda_: float = 0.6,
    beta: dict[str, float] | None = None,
    alpha_monthly: float = 0.001,
    factor_vol_annual: dict[str, float] | None = None,
    sigma_eps_monthly: float = 0.04,
    rng: np.random.Generator | int | None = None,
    start: str = "1995-01-31",
) -> AR1SyntheticDataset:
    """Generate an AR(1)-smoothed quarterly PE dataset with a known factor model.

    The data-generating process is::

        F_m  ~ N(0, σ²_f / 12)             monthly factor returns (independent)
        ε_m  ~ N(0, σ²_ε)                  idiosyncratic noise
        r_m  = β·F_m + α + ε_m             true monthly returns
        r_q  = compounded-aggregate(r_m)   true quarterly returns
        s_q  = (1−λ)·r_q + λ·s_{q−1}       smoothed (observed) quarterly returns

    Parameters
    ----------
    n_years
        Length of the simulated series in years.
    lambda_
        AR(1) smoothing parameter applied at the quarterly level. Must lie in
        ``[0, 1)``.
    beta
        Factor betas. Defaults to ``{'equity_market': 1.15}`` (US large-cap
        buyout-style).
    alpha_monthly
        Per-month additive alpha.
    factor_vol_annual
        Annualised factor volatilities. Defaults to 18 % per factor.
    sigma_eps_monthly
        Per-month additive idiosyncratic volatility.
    rng
        ``numpy.random.Generator`` instance, integer seed, or ``None``.
    start
        Calendar start of the monthly index (a month-end date).

    Returns
    -------
    AR1SyntheticDataset
    """
    if not (0.0 <= lambda_ < 1.0):
        raise ValueError(f"lambda_ must lie in [0, 1), got {lambda_}")
    rng = _as_rng(rng)

    if beta is None:
        beta = {"equity_market": 1.15}
    if factor_vol_annual is None:
        factor_vol_annual = {k: 0.18 for k in beta}
    if set(factor_vol_annual) != set(beta):
        raise ValueError("factor_vol_annual must have the same keys as beta")

    n_months = n_years * 12
    n_quarters = n_years * 4
    midx = _monthly_index(start, n_months)
    qidx = _quarterly_index(pd.Timestamp(start) + pd.offsets.QuarterEnd(0), n_quarters)

    factor_monthly = pd.DataFrame(
        {
            k: rng.normal(0.0, factor_vol_annual[k] / np.sqrt(12), size=n_months)
            for k in beta
        },
        index=midx,
    )

    systematic = np.zeros(n_months)
    for k, b in beta.items():
        systematic += b * factor_monthly[k].to_numpy()
    eps = rng.normal(0.0, sigma_eps_monthly, size=n_months)
    true_monthly_arr = systematic + alpha_monthly + eps
    true_monthly = pd.Series(true_monthly_arr, index=midx, name="r_true_monthly")

    # Compound monthly → quarterly via log-space (multiplicative compounding).
    log_m = np.log1p(true_monthly_arr)
    log_q = log_m.reshape(n_quarters, 3).sum(axis=1)
    true_quarterly_arr = np.expm1(log_q)
    true_quarterly = pd.Series(true_quarterly_arr, index=qidx, name="r_true_q")

    factor_quarterly = pd.DataFrame(
        {
            k: np.expm1(
                np.log1p(factor_monthly[k].to_numpy()).reshape(n_quarters, 3).sum(axis=1)
            )
            for k in factor_monthly
        },
        index=qidx,
    )

    # AR(1) smoothing on quarterly true returns.
    smoothed = np.empty(n_quarters)
    smoothed[0] = true_quarterly_arr[0]
    for t in range(1, n_quarters):
        smoothed[t] = (1.0 - lambda_) * true_quarterly_arr[t] + lambda_ * smoothed[t - 1]
    observed_quarterly = pd.Series(smoothed, index=qidx, name="r_obs_q")

    return AR1SyntheticDataset(
        observed_quarterly=observed_quarterly,
        true_quarterly=true_quarterly,
        true_monthly=true_monthly,
        factor_monthly=factor_monthly,
        factor_quarterly=factor_quarterly,
        lambda_=lambda_,
        beta=dict(beta),
        alpha=alpha_monthly,
        sigma_eps=sigma_eps_monthly,
    )


# ──────────────────────────────────────────────────────────────────
# MA(q) smoothed hedge-fund data (Getmansky-Lo-Makarov, 2004)
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MAqSyntheticDataset:
    """MA(q)-smoothed monthly hedge-fund returns from a known factor model.

    Attributes
    ----------
    observed_monthly
        Smoothed observed monthly returns (input to the pipeline).
    true_monthly
        Latent unsmoothed monthly returns.
    factor_monthly
        Monthly factor returns.
    theta
        True MA weights, ordered ``(θ_0, θ_1, …, θ_q)``. Sum to 1, non-negative.
    beta, alpha, sigma_eps
        Latent factor-model parameters.
    burn_in
        Number of leading observations whose smoothed value uses fewer than
        ``q+1`` lagged true returns. Tests that compare ``observed_monthly``
        moments to those implied by ``theta`` should drop these.
    """

    observed_monthly: pd.Series
    true_monthly: pd.Series
    factor_monthly: pd.DataFrame
    theta: np.ndarray
    beta: dict[str, float]
    alpha: float
    sigma_eps: float
    burn_in: int


def make_ma_hf_dataset(
    *,
    n_years: int = 20,
    theta: Sequence[float] = (0.6, 0.3, 0.1),
    beta: dict[str, float] | None = None,
    alpha_monthly: float = 0.002,
    factor_vol_annual: dict[str, float] | None = None,
    sigma_eps_monthly: float = 0.02,
    rng: np.random.Generator | int | None = None,
    start: str = "2005-01-31",
) -> MAqSyntheticDataset:
    r"""Generate an MA(q)-smoothed monthly hedge-fund dataset.

    The data-generating process is::

        F_m  ~ N(0, σ²_f / 12)
        ε_m  ~ N(0, σ²_ε)
        r_m  = β·F_m + α + ε_m              true monthly returns
        s_t  = Σ_{j=0}^{q} θ_j · r_{t−j}    smoothed (observed) monthly returns

    The MA weights satisfy ``Σ θ_j = 1`` and ``θ_j ≥ 0`` (Getmansky-Lo-Makarov
    constraints). For ``t < q`` the sum truncates — these leading observations
    are flagged via :attr:`MAqSyntheticDataset.burn_in`.

    Parameters
    ----------
    n_years
        Length of the simulated series in years.
    theta
        MA weights of length ``q+1``. Must be non-negative and sum to 1.
    beta, factor_vol_annual
        Factor model. Defaults give a hedged equity-L/S-style profile.
    alpha_monthly, sigma_eps_monthly
        Latent factor-model parameters.
    rng
        ``numpy.random.Generator``, integer seed, or ``None``.
    start
        Calendar start of the monthly index.

    Returns
    -------
    MAqSyntheticDataset
    """
    rng = _as_rng(rng)
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim != 1 or theta_arr.size < 1:
        raise ValueError("theta must be a 1-D array of length >= 1")
    if (theta_arr < 0).any():
        raise ValueError("theta entries must be non-negative")
    if not np.isclose(theta_arr.sum(), 1.0, atol=1e-10):
        raise ValueError(f"theta must sum to 1, got {theta_arr.sum():.6f}")

    if beta is None:
        beta = {"equity_market": 0.4, "smb": 0.1}
    if factor_vol_annual is None:
        factor_vol_annual = {"equity_market": 0.16, "smb": 0.10}
    if set(factor_vol_annual) != set(beta):
        raise ValueError("factor_vol_annual must have the same keys as beta")

    n_months = n_years * 12
    midx = _monthly_index(start, n_months)

    factor_monthly = pd.DataFrame(
        {
            k: rng.normal(0.0, factor_vol_annual[k] / np.sqrt(12), size=n_months)
            for k in beta
        },
        index=midx,
    )

    systematic = np.zeros(n_months)
    for k, b in beta.items():
        systematic += b * factor_monthly[k].to_numpy()
    eps = rng.normal(0.0, sigma_eps_monthly, size=n_months)
    true_monthly_arr = systematic + alpha_monthly + eps
    true_monthly = pd.Series(true_monthly_arr, index=midx, name="r_true_monthly")

    # MA(q) convolution; truncated for t < q.
    q = theta_arr.size - 1
    smoothed = np.zeros(n_months)
    for j, w in enumerate(theta_arr):
        if j == 0:
            smoothed += w * true_monthly_arr
        else:
            smoothed[j:] += w * true_monthly_arr[:-j]
    observed_monthly = pd.Series(smoothed, index=midx, name="r_obs_monthly")

    return MAqSyntheticDataset(
        observed_monthly=observed_monthly,
        true_monthly=true_monthly,
        factor_monthly=factor_monthly,
        theta=theta_arr,
        beta=dict(beta),
        alpha=alpha_monthly,
        sigma_eps=sigma_eps_monthly,
        burn_in=q,
    )


# ──────────────────────────────────────────────────────────────────
# Threshold AR(1) smoothed credit data (carry / MTM split)
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThresholdAR1SyntheticDataset:
    """Threshold-AR(1)-smoothed quarterly credit returns with a carry / MTM split.

    Attributes
    ----------
    observed_quarterly
        Total observed quarterly return, equal to ``carry_quarterly +
        smoothed_mtm_quarterly``. This is what the pipeline ingests as the
        return series.
    carry_quarterly
        Exact carry component, computed as ``yield_quarterly * 0.25``. Bypasses
        Stage 1 desmoothing per the spec.
    true_mtm_quarterly
        Latent unsmoothed mark-to-market component. Stage 1 should recover
        this (up to noise) from ``smoothed_mtm_quarterly``.
    smoothed_mtm_quarterly
        Smoothed MTM (i.e. ``observed_quarterly − carry_quarterly``).
    yield_quarterly
        Running yield used to derive the carry component.
    regime_indicator
        Binary regime per quarter (0 = normal, 1 = stress) derived from
        ``spread_indicator > regime_threshold_bps``.
    spread_indicator
        Simulated HY OAS series in basis points.
    factor_quarterly
        Quarterly factor returns.
    lambda_normal, lambda_stress
        True smoothing parameters in each regime.
    beta, alpha_mtm, sigma_eps_mtm
        Latent factor-model parameters for the MTM component.
    regime_threshold_bps
        Threshold on ``spread_indicator`` defining the stress regime.
    """

    observed_quarterly: pd.Series
    carry_quarterly: pd.Series
    true_mtm_quarterly: pd.Series
    smoothed_mtm_quarterly: pd.Series
    yield_quarterly: pd.Series
    regime_indicator: pd.Series
    spread_indicator: pd.Series
    factor_quarterly: pd.DataFrame
    lambda_normal: float
    lambda_stress: float
    beta: dict[str, float]
    alpha_mtm: float
    sigma_eps_mtm: float
    regime_threshold_bps: float


def make_threshold_credit_dataset(
    *,
    n_years: int = 25,
    lambda_normal: float = 0.75,
    lambda_stress: float = 0.15,
    beta: dict[str, float] | None = None,
    alpha_mtm_quarterly: float = 0.0,
    factor_vol_annual: dict[str, float] | None = None,
    sigma_eps_mtm_quarterly: float = 0.02,
    yield_annual_mean: float = 0.07,
    yield_annual_drift_vol: float = 0.005,
    spread_baseline_bps: float = 350.0,
    spread_uncond_vol_bps: float = 150.0,
    spread_persistence: float = 0.85,
    regime_threshold_bps: float = 500.0,
    rng: np.random.Generator | int | None = None,
    start: str = "2000-03-31",
) -> ThresholdAR1SyntheticDataset:
    r"""Generate a threshold-AR(1)-smoothed quarterly credit dataset.

    The data-generating process is::

        carry_t   = yield_t · 0.25                                 (smooth accrual)
        true_MTM  = β · F_q + α + ε                                (factor-driven)
        regime_t  = 1{ HY_OAS_t > threshold }                       (exogenous)
        smoothed_MTM_t = (1 − λ_{regime_t}) · true_MTM_t
                         + λ_{regime_t} · smoothed_MTM_{t−1}
        observed_t = carry_t + smoothed_MTM_t

    The HY OAS series is itself a mean-reverting AR(1) in basis points with
    persistence :math:`\phi` and unconditional volatility ``spread_uncond_vol_bps``.

    Parameters
    ----------
    n_years
        Length of the simulated series in years.
    lambda_normal, lambda_stress
        AR(1) smoothing parameters in the normal and stress regimes
        respectively. Must lie in ``[0, 1)``.
    beta, factor_vol_annual
        Factor model on the MTM component. Defaults give a leveraged-loan-like
        profile.
    alpha_mtm_quarterly, sigma_eps_mtm_quarterly
        Latent factor-model parameters for the MTM component.
    yield_annual_mean, yield_annual_drift_vol
        Running-yield process parameters. The yield path is a clipped random
        walk with small per-quarter drift around ``yield_annual_mean``.
    spread_baseline_bps, spread_uncond_vol_bps, spread_persistence
        AR(1) parameters for the HY OAS series in basis points.
    regime_threshold_bps
        Stress regime fires when OAS exceeds this threshold.
    rng
        ``numpy.random.Generator``, integer seed, or ``None``.
    start
        Calendar start of the quarterly index.

    Returns
    -------
    ThresholdAR1SyntheticDataset
    """
    for name, value in (("lambda_normal", lambda_normal), ("lambda_stress", lambda_stress)):
        if not (0.0 <= value < 1.0):
            raise ValueError(f"{name} must lie in [0, 1), got {value}")
    if not (0.0 <= spread_persistence < 1.0):
        raise ValueError(f"spread_persistence must lie in [0, 1), got {spread_persistence}")

    rng = _as_rng(rng)

    if beta is None:
        beta = {"credit_spread": 0.7, "rate_duration": -0.2}
    if factor_vol_annual is None:
        factor_vol_annual = {"credit_spread": 0.12, "rate_duration": 0.06}
    if set(factor_vol_annual) != set(beta):
        raise ValueError("factor_vol_annual must have the same keys as beta")

    n_quarters = n_years * 4
    qidx = _quarterly_index(start, n_quarters)

    # Quarterly factor returns: σ_q = σ_a / √4.
    factor_quarterly = pd.DataFrame(
        {
            k: rng.normal(0.0, factor_vol_annual[k] / np.sqrt(4), size=n_quarters)
            for k in beta
        },
        index=qidx,
    )

    # Running yield: clipped random walk with quarterly drift noise.
    yield_innov = rng.normal(0.0, yield_annual_drift_vol / np.sqrt(4), size=n_quarters)
    yield_q = yield_annual_mean + np.cumsum(yield_innov)
    yield_q = np.clip(yield_q, 0.02, 0.18)
    yield_quarterly = pd.Series(yield_q, index=qidx, name="running_yield")
    carry_arr = yield_q * 0.25
    carry_quarterly = pd.Series(carry_arr, index=qidx, name="carry")

    # HY OAS as AR(1) in bps; innovation scaled so unconditional vol matches.
    innov_sd = spread_uncond_vol_bps * np.sqrt(1.0 - spread_persistence**2)
    spread = np.empty(n_quarters)
    spread[0] = spread_baseline_bps
    innov = rng.normal(0.0, innov_sd, size=n_quarters)
    for t in range(1, n_quarters):
        spread[t] = (
            spread_baseline_bps
            + spread_persistence * (spread[t - 1] - spread_baseline_bps)
            + innov[t]
        )
    spread = np.clip(spread, 100.0, 2500.0)
    spread_indicator = pd.Series(spread, index=qidx, name="HY_OAS_bps")
    regime = (spread > regime_threshold_bps).astype(int)
    regime_indicator = pd.Series(regime, index=qidx, name="regime")

    # True (unsmoothed) MTM.
    systematic = np.zeros(n_quarters)
    for k, b in beta.items():
        systematic += b * factor_quarterly[k].to_numpy()
    eps = rng.normal(0.0, sigma_eps_mtm_quarterly, size=n_quarters)
    true_mtm_arr = systematic + alpha_mtm_quarterly + eps
    true_mtm_quarterly = pd.Series(true_mtm_arr, index=qidx, name="mtm_true")

    # Regime-dependent AR(1) smoothing of the MTM component.
    smoothed_mtm = np.empty(n_quarters)
    smoothed_mtm[0] = true_mtm_arr[0]
    for t in range(1, n_quarters):
        lam = lambda_stress if regime[t] == 1 else lambda_normal
        smoothed_mtm[t] = (1.0 - lam) * true_mtm_arr[t] + lam * smoothed_mtm[t - 1]
    smoothed_mtm_quarterly = pd.Series(smoothed_mtm, index=qidx, name="mtm_smoothed")

    observed_quarterly = pd.Series(
        carry_arr + smoothed_mtm, index=qidx, name="r_obs_q"
    )

    return ThresholdAR1SyntheticDataset(
        observed_quarterly=observed_quarterly,
        carry_quarterly=carry_quarterly,
        true_mtm_quarterly=true_mtm_quarterly,
        smoothed_mtm_quarterly=smoothed_mtm_quarterly,
        yield_quarterly=yield_quarterly,
        regime_indicator=regime_indicator,
        spread_indicator=spread_indicator,
        factor_quarterly=factor_quarterly,
        lambda_normal=lambda_normal,
        lambda_stress=lambda_stress,
        beta=dict(beta),
        alpha_mtm=alpha_mtm_quarterly,
        sigma_eps_mtm=sigma_eps_mtm_quarterly,
        regime_threshold_bps=regime_threshold_bps,
    )


# ──────────────────────────────────────────────────────────────────
# Pytest fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def ar1_pe_dataset() -> AR1SyntheticDataset:
    """Default 30-year AR(1)-smoothed PE dataset (λ=0.6, β=1.15, seed=42)."""
    return make_ar1_pe_dataset(rng=42)


@pytest.fixture
def ma_hf_dataset() -> MAqSyntheticDataset:
    """Default 20-year MA(2)-smoothed HF dataset (θ=(0.6,0.3,0.1), seed=7)."""
    return make_ma_hf_dataset(rng=7)


@pytest.fixture
def threshold_credit_dataset() -> ThresholdAR1SyntheticDataset:
    """Default 25-year threshold-AR(1) credit dataset (λ_n=0.75, λ_s=0.15, seed=13)."""
    return make_threshold_credit_dataset(rng=13)


@pytest.fixture
def make_ar1_pe():
    """Factory fixture: returns :func:`make_ar1_pe_dataset` for parameterised tests."""
    return make_ar1_pe_dataset


@pytest.fixture
def make_ma_hf():
    """Factory fixture: returns :func:`make_ma_hf_dataset` for parameterised tests."""
    return make_ma_hf_dataset


@pytest.fixture
def make_threshold_credit():
    """Factory fixture: returns :func:`make_threshold_credit_dataset` for parameterised tests."""
    return make_threshold_credit_dataset
