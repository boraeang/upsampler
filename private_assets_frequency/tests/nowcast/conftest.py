"""
Synthetic data generators for the nowcast test suite.

Three data-generating processes, each targeting a specific claim the module
makes:

1. :func:`make_linear_smoothing_dataset` — the reduced form the smoothing
   regression is *supposed* to recover exactly. A daily public factor drives a
   latent quarterly return ``r_Q = α + β·F_Q + ε_Q``, which is smoothed into the
   reported series ``s_Q = (1−λ)·r_Q + λ·s_{Q−1}``. Includes a publication lag
   and a revision process, so a vintage table is produced alongside the final
   revised series. On this DGP the signature model must *not* beat the
   smoothing regression by a significant margin — that is the overfitting
   check.

2. :func:`make_path_dependent_dataset` — identical, except the appraiser marks
   against the *path* of the public market within the quarter rather than its
   closing return (either the within-quarter average level, or a
   recency-weighted average of moves). Here within-quarter path information
   genuinely matters, and a level-2 signature model should win.

3. :func:`make_early_signal_dataset` — the linear DGP plus an irregularly
   observed "listed PE NAV" series correlated with the latent true return and
   released on random dates. This is where the irregular-data strength of
   signatures is most relevant.

All generators take an ``rng`` or ``seed`` and are fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

__all__ = [
    "NowcastDataset",
    "make_early_signal_dataset",
    "make_linear_smoothing_dataset",
    "make_path_dependent_dataset",
]


_BDAYS_PER_YEAR = 252
_DEFAULT_PUBLICATION_LAG_DAYS = 100


# ──────────────────────────────────────────────────────────────────
# Container
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NowcastDataset:
    """A synthetic nowcasting dataset with full ground truth.

    Attributes
    ----------
    reported
        Final revised reported quarterly returns, indexed by quarterly
        ``pd.PeriodIndex``. This is the series a user without a vintage table
        would supply.
    first_print
        The value first published for each quarter — what a real-time nowcast
        is scored against by default.
    vintages
        Long frame ``[quarter, release_date, value]`` with one row per release
        of each quarter including revisions. The last release of each quarter
        equals ``reported`` exactly.
    true_returns
        The latent unsmoothed economic return ``r_Q``. Ground truth for
        ``target='true'``.
    daily_factors
        Daily public factor simple returns on a business-day index.
    monthly_factors
        The same factors compounded to month end.
    quarterly_factors
        The same factors compounded to quarter end, indexed by quarterly
        ``pd.PeriodIndex``. What the smoothing regression's ``F_Q`` should be.
    early_signals
        Irregularly timestamped proxy series, or ``None``.
    params
        Ground-truth parameters: ``lam``, ``beta`` (dict), ``alpha``,
        ``sigma_eps``, ``publication_lag_days``, plus DGP-specific entries.
    """

    reported: pd.Series
    first_print: pd.Series
    vintages: pd.DataFrame
    true_returns: pd.Series
    daily_factors: pd.DataFrame
    monthly_factors: pd.DataFrame
    quarterly_factors: pd.DataFrame
    early_signals: pd.DataFrame | None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def factor_columns(self) -> list[str]:
        """Factor column names."""
        return list(self.daily_factors.columns)

    def as_of_for(
        self, quarter: str | pd.Period, *, extra_days: int = 0
    ) -> pd.Timestamp:
        """A nowcast date shortly after ``quarter`` closes.

        Parameters
        ----------
        quarter
            Reference quarter.
        extra_days
            Calendar days past quarter end. ``0`` returns quarter end itself.

        Returns
        -------
        pd.Timestamp
        """
        q = quarter if isinstance(quarter, pd.Period) else pd.Period(quarter, freq="Q")
        return q.end_time.normalize() + pd.Timedelta(days=extra_days)


# ──────────────────────────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────────────────────────


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _daily_factor_panel(
    start_quarter: pd.Period,
    n_quarters: int,
    factor_vols: dict[str, float],
    factor_corr: float,
    rng: np.random.Generator,
    *,
    extra_quarters: int = 2,
) -> pd.DataFrame:
    """Daily iid-normal factor returns spanning the quarter range plus a tail.

    The tail (``extra_quarters``) exists so tests can build an information set
    whose public data runs past the last published quarter — the ragged edge
    the module is about.
    """
    end_quarter = start_quarter + n_quarters - 1 + extra_quarters
    index = pd.bdate_range(start_quarter.start_time, end_quarter.end_time)
    names = list(factor_vols)
    k = len(names)
    daily_sd = np.array([factor_vols[n] for n in names]) / np.sqrt(_BDAYS_PER_YEAR)

    if k == 1:
        z = rng.standard_normal((len(index), 1))
    else:
        corr = np.full((k, k), float(factor_corr))
        np.fill_diagonal(corr, 1.0)
        chol = np.linalg.cholesky(corr)
        z = rng.standard_normal((len(index), k)) @ chol.T
    return pd.DataFrame(z * daily_sd, index=index, columns=names)


def _compound(frame: pd.DataFrame, periods: pd.Index) -> pd.DataFrame:
    """Compound ``frame`` within each group of ``periods`` as ``Π(1+r)−1``."""
    log1p = np.log1p(frame)
    out = np.expm1(log1p.groupby(periods, sort=True).sum(min_count=1))
    return out


def _quarterly_factors(daily: pd.DataFrame) -> pd.DataFrame:
    out = _compound(daily, pd.PeriodIndex(daily.index, freq="Q"))
    out.index = pd.PeriodIndex(out.index, freq="Q")
    out.index.name = None
    return out


def _monthly_factors(daily: pd.DataFrame) -> pd.DataFrame:
    out = _compound(daily, pd.PeriodIndex(daily.index, freq="M"))
    out.index = pd.PeriodIndex(out.index, freq="M").to_timestamp(how="end").normalize()
    out.index.name = None
    return out


def _smooth(true_returns: np.ndarray, lam: float) -> np.ndarray:
    r"""Apply the AR(1) appraisal filter :math:`s_t = (1-\lambda)r_t + \lambda s_{t-1}`.

    Initialised at ``s_0 = r_0`` so the filter starts in its own steady state
    rather than at zero (a zero start injects a spurious transient that biases
    λ recovery on short samples).
    """
    s = np.empty_like(true_returns)
    s[0] = true_returns[0]
    for t in range(1, len(true_returns)):
        s[t] = (1.0 - lam) * true_returns[t] + lam * s[t - 1]
    return s


def _build_vintages(
    reported: pd.Series,
    *,
    publication_lag_days: int,
    n_releases: int,
    revision_sd: float,
    revision_decay: float,
    release_spacing_days: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series]:
    r"""Generate a vintage table whose final release equals ``reported``.

    Release ``k`` (1-indexed) of quarter ``Q`` is published at
    ``Q.end + publication_lag + (k-1)·release_spacing`` and carries value
    ``s_Q + δ_k`` where

    .. math::
        \delta_k = \sigma_{rev}\,\rho^{\,k-1} z_k \ (k < K), \qquad \delta_K = 0 .

    The decaying noise reproduces the real pattern — a first print based on a
    partial set of fund NAVs, converging over the next few releases as
    late reporters arrive — and makes the *revised* series genuinely easier to
    predict than the first print, which is the effect
    ``evaluation_target='first_print'`` exists to avoid flattering.

    Returns
    -------
    vintages : pd.DataFrame
        Columns ``[quarter, release_date, value]``.
    first_print : pd.Series
        Release 1 of each quarter.
    """
    if n_releases < 1:
        raise ValueError("n_releases must be >= 1")
    quarters = pd.PeriodIndex(reported.index, freq="Q")
    rows: list[dict[str, Any]] = []
    first: dict[pd.Period, float] = {}
    values = reported.to_numpy()
    for i in range(len(quarters)):
        q, s_q = quarters[i], values[i]
        base = q.end_time.normalize() + pd.Timedelta(days=publication_lag_days)
        for k in range(1, n_releases + 1):
            if k < n_releases:
                delta = revision_sd * (revision_decay ** (k - 1)) * rng.standard_normal()
            else:
                delta = 0.0
            value = float(s_q + delta)
            release = base + pd.Timedelta(days=release_spacing_days * (k - 1))
            rows.append(
                {"quarter": q, "release_date": release, "value": value}
            )
            if k == 1:
                first[q] = value
    vintages = pd.DataFrame(rows, columns=["quarter", "release_date", "value"])
    first_print = pd.Series(first, name="first_print")
    first_print.index = pd.PeriodIndex(first_print.index, freq="Q")
    return vintages, first_print


# ──────────────────────────────────────────────────────────────────
# DGP 1 — linear smoothing
# ──────────────────────────────────────────────────────────────────


def make_linear_smoothing_dataset(
    *,
    n_quarters: int = 100,
    start_quarter: str = "2001Q1",
    lam: float = 0.6,
    beta: dict[str, float] | None = None,
    alpha: float = 0.005,
    sigma_eps: float = 0.025,
    factor_vols: dict[str, float] | None = None,
    factor_corr: float = 0.3,
    publication_lag_days: int = _DEFAULT_PUBLICATION_LAG_DAYS,
    n_releases: int = 3,
    revision_sd: float = 0.004,
    revision_decay: float = 0.5,
    release_spacing_days: int = 91,
    extra_quarters: int = 2,
    seed: int | np.random.Generator | None = 20261007,
) -> NowcastDataset:
    r"""Linear smoothing DGP — the reduced form of the library's AR(1) model.

    .. code-block:: text

        daily factor returns  →  F_Q = Π(1+r_d) − 1
        r_Q = α + Σ_k β_k F_{k,Q} + ε_Q,     ε_Q ~ N(0, σ_ε²)
        s_Q = (1 − λ) r_Q + λ s_{Q−1}

    Substituting the factor equation into the filter gives the reduced form the
    smoothing regression estimates::

        s_Q = (1−λ)α + Σ_k (1−λ)β_k F_{k,Q} + λ s_{Q−1} + (1−λ)ε_Q

    so the recovery targets are ``b_k ≈ (1−λ)β_k`` and ``c_1 ≈ λ`` — this is
    what ``test_models.py::test_recovery_*`` asserts.

    Parameters
    ----------
    n_quarters
        Number of quarters of reported history. The default ~100 is the real
        sample size the module is designed for, not a convenience.
    start_quarter
        First reference quarter.
    lam
        True smoothing parameter λ.
    beta, alpha, sigma_eps
        True factor loadings (keyed by factor name), intercept and
        idiosyncratic volatility of the *latent* quarterly return.
    factor_vols
        Annualised volatility per factor. Keys must match ``beta``.
    factor_corr
        Pairwise correlation between factors when there is more than one.
    publication_lag_days, n_releases, revision_sd, revision_decay,
    release_spacing_days
        Publication and revision process — see :func:`_build_vintages`.
    extra_quarters
        Quarters of public factor data generated *beyond* the last reported
        quarter, so tests can exercise the ragged edge.
    seed
        Seed or ``Generator``.

    Returns
    -------
    NowcastDataset
    """
    rng = _as_rng(seed)
    beta = dict(beta) if beta is not None else {"equity_market": 1.10}
    factor_vols = (
        dict(factor_vols)
        if factor_vols is not None
        else {name: 0.16 for name in beta}
    )
    if set(beta) != set(factor_vols):
        raise ValueError(
            f"beta keys {sorted(beta)} must match factor_vols keys "
            f"{sorted(factor_vols)}."
        )

    q0 = pd.Period(start_quarter, freq="Q")
    daily = _daily_factor_panel(
        q0, n_quarters, factor_vols, factor_corr, rng, extra_quarters=extra_quarters
    )
    factors_q = _quarterly_factors(daily)
    quarters = pd.period_range(q0, periods=n_quarters, freq="Q")
    F = factors_q.loc[quarters, list(beta)]

    systematic = F.to_numpy() @ np.array([beta[c] for c in beta])
    eps = rng.normal(scale=sigma_eps, size=n_quarters)
    r = alpha + systematic + eps
    s = _smooth(r, lam)

    reported = pd.Series(s, index=quarters, name="reported")
    true_returns = pd.Series(r, index=quarters, name="true_return")
    vintages, first_print = _build_vintages(
        reported,
        publication_lag_days=publication_lag_days,
        n_releases=n_releases,
        revision_sd=revision_sd,
        revision_decay=revision_decay,
        release_spacing_days=release_spacing_days,
        rng=rng,
    )
    return NowcastDataset(
        reported=reported,
        first_print=first_print,
        vintages=vintages,
        true_returns=true_returns,
        daily_factors=daily,
        monthly_factors=_monthly_factors(daily),
        quarterly_factors=factors_q,
        early_signals=None,
        params={
            "dgp": "linear_smoothing",
            "lam": lam,
            "beta": dict(beta),
            "alpha": alpha,
            "sigma_eps": sigma_eps,
            "factor_vols": dict(factor_vols),
            "publication_lag_days": publication_lag_days,
            "n_releases": n_releases,
            "revision_sd": revision_sd,
            "n_quarters": n_quarters,
            # Reduced-form targets implied by the structural parameters.
            "reduced_form_b": {k: (1.0 - lam) * v for k, v in beta.items()},
            "reduced_form_c1": lam,
            "reduced_form_a": (1.0 - lam) * alpha,
        },
    )


# ──────────────────────────────────────────────────────────────────
# DGP 2 — path-dependent marking
# ──────────────────────────────────────────────────────────────────


def _within_quarter_path_statistic(
    daily: pd.DataFrame,
    quarters: pd.PeriodIndex,
    columns: list[str],
    weighting: Literal["average", "recency"],
) -> pd.DataFrame:
    r"""Path statistic the appraiser marks against, per quarter.

    Let :math:`X_u` be the cumulative log level of a factor within the quarter,
    rebased so :math:`X_0 = 0`, with :math:`u \in [0, 1]` the rescaled time.

    ``'average'``
        :math:`A_Q = \int_0^1 X_u \, du` — the average level over the quarter.
        An appraiser using a trailing average of comparable prices behaves this
        way.
    ``'recency'``
        :math:`A_Q = \int_0^1 2u \, dX_u` — moves late in the quarter count
        double those at the start.

    Both are level-2 signature terms of the path ``(time, X)``: the first is
    :math:`S^{(2,1)}`, the second is :math:`2 S^{(1,2)}`. Both are *linear*
    signature terms (each non-time channel appears once), so
    ``keep_sigs='linear'`` at ``level=2`` spans them, while ``level=1`` — which
    sees only the quarter-end increment :math:`X_1` — cannot. That asymmetry is
    exactly what ``test_models.py::test_path_dependence`` checks.

    Returns
    -------
    pd.DataFrame
        One column per factor, indexed by quarter, in *simple-return* units
        (``expm1`` of the log statistic) so it is comparable to ``F_Q``.
    """
    log1p = np.log1p(daily.loc[:, columns])
    out: dict[pd.Period, np.ndarray] = {}
    groups = log1p.groupby(pd.PeriodIndex(daily.index, freq="Q"), sort=True)
    for q, block in groups:
        if q not in quarters:
            continue
        # Cumulative log level within the quarter, rebased to 0 at the start.
        level = block.cumsum().to_numpy()
        n = level.shape[0]
        if n == 0:  # pragma: no cover - defensive
            continue
        u = (np.arange(1, n + 1)) / n  # rescaled time of each observation
        if weighting == "average":
            # ∫ X du by the trapezoid rule on the rebased path (X(0) = 0).
            padded = np.vstack([np.zeros((1, level.shape[1])), level])
            u_pad = np.concatenate([[0.0], u])
            stat = integrate.trapezoid(padded, x=u_pad, axis=0)
        elif weighting == "recency":
            # ∫ 2u dX_u, with dX the per-observation log increments.
            increments = block.to_numpy()
            stat = (2.0 * u[:, None] * increments).sum(axis=0)
        else:
            raise ValueError(
                f"weighting must be 'average' or 'recency', got {weighting!r}."
            )
        out[q] = stat
    frame = pd.DataFrame(out, index=columns).T
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    return np.expm1(frame)


def make_path_dependent_dataset(
    *,
    weighting: Literal["average", "recency"] = "average",
    n_quarters: int = 100,
    start_quarter: str = "2001Q1",
    lam: float = 0.6,
    beta: dict[str, float] | None = None,
    alpha: float = 0.005,
    sigma_eps: float = 0.012,
    factor_vols: dict[str, float] | None = None,
    publication_lag_days: int = _DEFAULT_PUBLICATION_LAG_DAYS,
    n_releases: int = 3,
    revision_sd: float = 0.004,
    revision_decay: float = 0.5,
    release_spacing_days: int = 91,
    extra_quarters: int = 2,
    seed: int | np.random.Generator | None = 20261008,
) -> NowcastDataset:
    r"""Path-dependent DGP — the appraiser marks against the within-quarter path.

    Identical to :func:`make_linear_smoothing_dataset` except that the latent
    return responds to a *path statistic* of the public market over the quarter
    rather than to the quarter-end compounded return:

    .. code-block:: text

        r_Q = α + Σ_k β_k A_{k,Q} + ε_Q      A = within-quarter path statistic
        s_Q = (1 − λ) r_Q + λ s_{Q−1}

    See :func:`_within_quarter_path_statistic` for the two statistics and why a
    level-2 linear signature spans them while ``F_Q`` alone does not.

    ``sigma_eps`` defaults lower than in the linear DGP: the path statistic has
    smaller variance than the quarter-end return (averaging damps it), so a
    comparable signal-to-noise ratio needs a smaller idiosyncratic term. Without
    that, the path signal is buried and the test would be measuring noise.

    Parameters
    ----------
    weighting
        ``'average'`` (default) or ``'recency'`` — see
        :func:`_within_quarter_path_statistic`.
    n_quarters, start_quarter, lam, beta, alpha, sigma_eps, factor_vols
        As in :func:`make_linear_smoothing_dataset`.
    publication_lag_days, n_releases, revision_sd, revision_decay,
    release_spacing_days, extra_quarters, seed
        As in :func:`make_linear_smoothing_dataset`.

    Returns
    -------
    NowcastDataset
        ``params['path_statistic']`` holds the realised ``A_Q`` frame, so a test
        can confirm the signal is really there before concluding a model failed
        to find it.
    """
    rng = _as_rng(seed)
    beta = dict(beta) if beta is not None else {"equity_market": 1.10}
    factor_vols = (
        dict(factor_vols)
        if factor_vols is not None
        else {name: 0.16 for name in beta}
    )
    q0 = pd.Period(start_quarter, freq="Q")
    daily = _daily_factor_panel(
        q0, n_quarters, factor_vols, 0.3, rng, extra_quarters=extra_quarters
    )
    factors_q = _quarterly_factors(daily)
    quarters = pd.period_range(q0, periods=n_quarters, freq="Q")
    A = _within_quarter_path_statistic(daily, quarters, list(beta), weighting)
    A = A.loc[quarters]

    systematic = A.to_numpy() @ np.array([beta[c] for c in beta])
    eps = rng.normal(scale=sigma_eps, size=n_quarters)
    r = alpha + systematic + eps
    s = _smooth(r, lam)

    reported = pd.Series(s, index=quarters, name="reported")
    vintages, first_print = _build_vintages(
        reported,
        publication_lag_days=publication_lag_days,
        n_releases=n_releases,
        revision_sd=revision_sd,
        revision_decay=revision_decay,
        release_spacing_days=release_spacing_days,
        rng=rng,
    )
    return NowcastDataset(
        reported=reported,
        first_print=first_print,
        vintages=vintages,
        true_returns=pd.Series(r, index=quarters, name="true_return"),
        daily_factors=daily,
        monthly_factors=_monthly_factors(daily),
        quarterly_factors=factors_q,
        early_signals=None,
        params={
            "dgp": "path_dependent",
            "weighting": weighting,
            "lam": lam,
            "beta": dict(beta),
            "alpha": alpha,
            "sigma_eps": sigma_eps,
            "factor_vols": dict(factor_vols),
            "publication_lag_days": publication_lag_days,
            "n_releases": n_releases,
            "revision_sd": revision_sd,
            "n_quarters": n_quarters,
            "path_statistic": A,
        },
    )


# ──────────────────────────────────────────────────────────────────
# DGP 3 — early signals
# ──────────────────────────────────────────────────────────────────


def make_early_signal_dataset(
    *,
    n_quarters: int = 100,
    signal_correlation: float = 0.7,
    signals_per_quarter: int = 4,
    signal_vol: float = 0.09,
    seed: int | np.random.Generator | None = 20261009,
    **linear_kwargs: Any,
) -> NowcastDataset:
    r"""Linear DGP plus an irregularly observed proxy for private marks.

    Adds a ``listed_pe_nav`` series whose observations land on random dates
    *inside* each quarter and whose value is correlated with that quarter's
    latent true return:

    .. code-block:: text

        signal_obs = ρ · (r_Q − α) / σ_r · σ_signal + √(1 − ρ²) · σ_signal · η

    Because the observations fall inside the quarter, a nowcaster at an
    ``as_of`` shortly after quarter end can see them while the index print is
    still months away — the situation in which an early signal is worth
    anything. The dates are irregular and the count per quarter varies, which
    is the data shape signature methods handle without resampling.

    Parameters
    ----------
    n_quarters
        Quarters of reported history.
    signal_correlation
        ρ between the signal observations and the latent true return of their
        quarter. ``0.0`` gives a pure-noise channel, useful for confirming the
        model does not reward itself for using it.
    signals_per_quarter
        Mean number of observations per quarter; the realised count is Poisson,
        floored at 1, so coverage is genuinely ragged.
    signal_vol
        Standard deviation of the signal series.
    seed
        Seed or ``Generator``.
    **linear_kwargs
        Forwarded to :func:`make_linear_smoothing_dataset`.

    Returns
    -------
    NowcastDataset
        With ``early_signals`` populated.
    """
    rng = _as_rng(seed)
    base = make_linear_smoothing_dataset(
        n_quarters=n_quarters, seed=rng, **linear_kwargs
    )
    r = base.true_returns
    r_centred = (r - r.mean()) / r.std(ddof=1)

    rows: list[tuple[pd.Timestamp, float]] = []
    rho = float(signal_correlation)
    signal_quarters = pd.PeriodIndex(r.index, freq="Q")
    z_values = r_centred.to_numpy()
    for i in range(len(signal_quarters)):
        q, z = signal_quarters[i], z_values[i]
        n_obs = max(1, int(rng.poisson(signals_per_quarter)))
        days = pd.bdate_range(q.start_time, q.end_time)
        if len(days) == 0:  # pragma: no cover - defensive
            continue
        picks = rng.choice(len(days), size=min(n_obs, len(days)), replace=False)
        for i in sorted(picks):
            noise = rng.standard_normal()
            value = signal_vol * (rho * z + np.sqrt(max(1.0 - rho**2, 0.0)) * noise)
            rows.append((days[i], float(value)))

    rows.sort(key=lambda t: t[0])
    early = pd.DataFrame(
        {"listed_pe_nav": [v for _, v in rows]},
        index=pd.DatetimeIndex([d for d, _ in rows], name=None),
    )
    params = dict(base.params)
    params.update(
        {
            "dgp": "early_signal",
            "signal_correlation": rho,
            "signals_per_quarter": signals_per_quarter,
            "signal_vol": signal_vol,
        }
    )
    return NowcastDataset(
        reported=base.reported,
        first_print=base.first_print,
        vintages=base.vintages,
        true_returns=base.true_returns,
        daily_factors=base.daily_factors,
        monthly_factors=base.monthly_factors,
        quarterly_factors=base.quarterly_factors,
        early_signals=early,
        params=params,
    )


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def linear_dataset() -> NowcastDataset:
    """100 quarters of the linear smoothing DGP, λ=0.6, one factor."""
    return make_linear_smoothing_dataset()


@pytest.fixture(scope="session")
def motivating_dataset() -> NowcastDataset:
    """The module spec's motivating case, as data.

    103 quarters ending 2026Q3, with daily public factors running past
    2026-10-07. At ``as_of = 2026-10-07`` and a 100-day publication lag the last
    published quarter is 2026Q1, leaving 2026Q2 at horizon 1 and 2026Q3 at
    horizon 2 — and factor data beyond ``as_of`` for the leakage test to
    perturb.
    """
    return make_linear_smoothing_dataset(
        n_quarters=103, start_quarter="2001Q1", seed=20261011
    )


@pytest.fixture(scope="session")
def linear_dataset_two_factor() -> NowcastDataset:
    """Linear DGP with two correlated factors."""
    return make_linear_smoothing_dataset(
        beta={"equity_market": 1.10, "credit_proxy": 0.35},
        factor_vols={"equity_market": 0.16, "credit_proxy": 0.09},
        seed=20261010,
    )


@pytest.fixture(scope="session")
def path_dataset() -> NowcastDataset:
    """100 quarters of the path-dependent (within-quarter average) DGP."""
    return make_path_dependent_dataset()


@pytest.fixture(scope="session")
def early_signal_dataset() -> NowcastDataset:
    """Linear DGP plus an irregular listed-PE-NAV proxy series."""
    return make_early_signal_dataset()


@pytest.fixture
def make_linear():
    """Factory fixture for :func:`make_linear_smoothing_dataset`."""
    return make_linear_smoothing_dataset


@pytest.fixture
def make_path():
    """Factory fixture for :func:`make_path_dependent_dataset`."""
    return make_path_dependent_dataset


@pytest.fixture
def make_early_signal():
    """Factory fixture for :func:`make_early_signal_dataset`."""
    return make_early_signal_dataset
