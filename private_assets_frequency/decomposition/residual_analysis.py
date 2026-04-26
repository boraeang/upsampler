r"""
Residual diagnostics and distribution fitting (Stage 2).

This module characterises the idiosyncratic residual series ``ε_t`` produced
by :mod:`factor_model`. The pipeline uses the diagnostics to detect
mis-specified factor models (auto-correlated residuals → missing AR factor;
ARCH effects → missing volatility regime) and the fitted distribution to
draw daily idiosyncratic shocks in Stage 4 simulation mode.

Per the spec (Stage 2 spec point 3):

    * Ljung-Box for residual autocorrelation
    * ARCH-LM for residual heteroscedasticity
    * Jarque-Bera for normality
    * Fit normal, Student-t, skewed-t (Hansen 1994); pick best by AIC

Implementation notes
--------------------
* Ljung-Box and ARCH-LM use :mod:`statsmodels.stats.diagnostic` rather than
  rolling our own — these are well-tested.
* Hansen-skewed-t is implemented from scratch; scipy doesn't ship it. The
  parametrisation follows Hansen (1994) directly: skewness ``λ ∈ (-1, 1)``
  and degrees of freedom ``η ∈ (2, ∞)``. Standardised PDF is piecewise on
  the threshold ``z = -a/b`` with location & scale recovered by
  ``r_t = μ + σ·z``.
* :func:`fit_best_distribution` returns the candidate with the lowest AIC
  out of {normal, Student-t, skewed-t}; tied AIC favours the simpler model.

References
----------
.. [1] Hansen (1994) — "Autoregressive conditional density estimation,"
       *International Economic Review* 35: 705-730.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# ──────────────────────────────────────────────────────────────────
# Diagnostic test results
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LjungBoxResult:
    """Output of the Ljung-Box autocorrelation test at one or more lags."""

    lags: tuple[int, ...]
    statistic: tuple[float, ...]
    p_value: tuple[float, ...]

    def passes_at(self, lag: int, alpha: float = 0.05) -> bool:
        """``True`` if the null of no autocorrelation is *not* rejected at ``lag``."""
        if lag not in self.lags:
            raise KeyError(f"lag {lag} not in tested lags {self.lags!r}.")
        idx = self.lags.index(lag)
        return self.p_value[idx] > alpha


@dataclass(frozen=True)
class ArchLMResult:
    """Outcome of an Engle ARCH-LM test for conditional heteroscedasticity."""

    statistic: float
    p_value: float

    def has_arch(self, alpha: float = 0.05) -> bool:
        """``True`` if the null of homoscedasticity is rejected at level ``alpha``."""
        return self.p_value <= alpha


@dataclass(frozen=True)
class JarqueBeraResult:
    """Outcome of a Jarque-Bera normality test, plus skew and excess kurtosis."""

    statistic: float
    p_value: float
    skewness: float
    kurtosis: float  # excess kurtosis (i.e. minus 3)

    def passes(self, alpha: float = 0.05) -> bool:
        """``True`` if the null of normality is *not* rejected at level ``alpha``."""
        return self.p_value > alpha


# ──────────────────────────────────────────────────────────────────
# Diagnostic functions
# ──────────────────────────────────────────────────────────────────


def ljung_box(
    residuals: pd.Series | np.ndarray,
    lags: tuple[int, ...] = (5, 10, 20),
) -> LjungBoxResult:
    """Ljung-Box test at multiple lags.

    Returns the test statistic and p-value at each ``lag`` in ``lags``.
    """
    arr = _as_array(residuals)
    n = len(arr)
    valid = tuple(L for L in lags if L < n)
    if not valid:
        raise ValueError(f"all requested lags >= n={n}; cannot run Ljung-Box.")
    df = acorr_ljungbox(arr, lags=list(valid), return_df=True)
    return LjungBoxResult(
        lags=valid,
        statistic=tuple(float(x) for x in df["lb_stat"].to_numpy()),
        p_value=tuple(float(x) for x in df["lb_pvalue"].to_numpy()),
    )


def arch_lm(
    residuals: pd.Series | np.ndarray,
    nlags: int = 5,
) -> ArchLMResult:
    """Engle's ARCH-LM test for conditional heteroscedasticity."""
    arr = _as_array(residuals)
    if len(arr) <= nlags + 1:
        raise ValueError(f"need > {nlags + 1} observations, got {len(arr)}.")
    stat, pval, _, _ = het_arch(arr, nlags=nlags)
    return ArchLMResult(statistic=float(stat), p_value=float(pval))


def jarque_bera(residuals: pd.Series | np.ndarray) -> JarqueBeraResult:
    """Jarque-Bera test for normality, plus skew and excess kurtosis."""
    arr = _as_array(residuals)
    if len(arr) < 8:
        raise ValueError(f"need >= 8 observations, got {len(arr)}.")
    jb = stats.jarque_bera(arr)
    return JarqueBeraResult(
        statistic=float(jb.statistic),
        p_value=float(jb.pvalue),
        skewness=float(stats.skew(arr)),
        kurtosis=float(stats.kurtosis(arr)),
    )


# ──────────────────────────────────────────────────────────────────
# Distribution fitting
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FittedDistribution:
    """A fitted parametric distribution with diagnostics.

    Attributes
    ----------
    name
        ``'normal'`` | ``'student_t'`` | ``'skewed_t'``
    params
        Mapping of parameter names to fitted values. Standard fields:
        ``loc``, ``scale``; ``df`` for Student-t; ``df`` and ``skew`` for
        Hansen skewed-t.
    log_likelihood
        Maximised log-likelihood at the fit.
    aic
        Akaike information criterion: ``2k - 2·log_likelihood``.
    bic
        Bayesian information criterion: ``k·log(n) - 2·log_likelihood``.
    n_obs
        Number of observations used in the fit.
    """

    name: str
    params: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int

    @property
    def n_params(self) -> int:
        return len(self.params)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Density at ``x``."""
        if self.name == "normal":
            return stats.norm.pdf(x, loc=self.params["loc"], scale=self.params["scale"])
        if self.name == "student_t":
            return stats.t.pdf(
                x,
                df=self.params["df"],
                loc=self.params["loc"],
                scale=self.params["scale"],
            )
        if self.name == "skewed_t":
            return _hansen_skewed_t_pdf(
                x,
                mu=self.params["loc"],
                sigma=self.params["scale"],
                eta=self.params["df"],
                lam=self.params["skew"],
            )
        raise ValueError(f"unknown distribution {self.name!r}")

    def sample(
        self,
        size: int,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        rng_g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        if self.name == "normal":
            return rng_g.normal(self.params["loc"], self.params["scale"], size=size)
        if self.name == "student_t":
            z = stats.t.rvs(self.params["df"], size=size, random_state=rng_g)
            return self.params["loc"] + self.params["scale"] * z
        if self.name == "skewed_t":
            return _hansen_skewed_t_sample(
                size=size,
                mu=self.params["loc"],
                sigma=self.params["scale"],
                eta=self.params["df"],
                lam=self.params["skew"],
                rng=rng_g,
            )
        raise ValueError(f"unknown distribution {self.name!r}")


def fit_normal(x: pd.Series | np.ndarray) -> FittedDistribution:
    """Fit a normal distribution by sample mean / unbiased std.

    Returns a :class:`FittedDistribution` with parameters ``loc`` and
    ``scale`` and the maximised log-likelihood / AIC / BIC.
    """
    arr = _as_array(x)
    n = len(arr)
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    ll = float(stats.norm.logpdf(arr, loc=mu, scale=sigma).sum())
    return _make_fitted("normal", {"loc": mu, "scale": sigma}, ll, n)


def fit_student_t(x: pd.Series | np.ndarray) -> FittedDistribution:
    """Fit a Student-t distribution via :func:`scipy.stats.t.fit`.

    Returns a :class:`FittedDistribution` with parameters ``df``, ``loc``,
    and ``scale``.
    """
    arr = _as_array(x)
    n = len(arr)
    df, loc, scale = stats.t.fit(arr)
    ll = float(stats.t.logpdf(arr, df=df, loc=loc, scale=scale).sum())
    return _make_fitted(
        "student_t", {"df": float(df), "loc": float(loc), "scale": float(scale)}, ll, n
    )


def fit_skewed_t(x: pd.Series | np.ndarray) -> FittedDistribution:
    """Hansen (1994) skewed-t MLE.

    Optimises ``(loc, scale, df, skew)`` jointly with a parametrisation
    that keeps the constraints (``scale > 0``, ``df > 2``, ``-1 < skew < 1``)
    automatic via softplus / tanh transforms.
    """
    arr = _as_array(x)
    n = len(arr)

    # Initial guess from Student-t fit. Clamp df so the inverse-softplus
    # transform `log(expm1(df - 2))` doesn't overflow when the underlying
    # data is essentially Gaussian (scipy can return df ≈ 1e6+).
    init_t = fit_student_t(arr)
    init_loc = init_t.params["loc"]
    init_scale = init_t.params["scale"]
    init_df = float(np.clip(init_t.params["df"], 4.0, 50.0))

    def neg_ll(theta_unc: np.ndarray) -> float:
        loc = theta_unc[0]
        # scale: clamp the log-scale to avoid overflow under wide Nelder-Mead steps
        scale = float(np.exp(np.clip(theta_unc[1], -50.0, 30.0)))
        # df: 2 + softplus(theta) — numerically stable via logaddexp
        df_unc = float(np.clip(theta_unc[2], -50.0, 50.0))
        df = 2.0 + float(np.logaddexp(0.0, df_unc))
        # skew: tanh(theta) → (-1, 1)
        skew = float(np.tanh(theta_unc[3]))
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            pdf = _hansen_skewed_t_pdf(arr, loc, scale, df, skew)
            log_pdf = np.log(np.where(pdf > 0.0, pdf, 1e-300))
        return -float(log_pdf.sum())

    # Inverse softplus: theta_unc s.t. 2 + softplus(theta_unc) = init_df.
    # softplus(x) = log(1 + exp(x)) → x = log(exp(y) - 1) where y = init_df - 2.
    # Use a stable form for large y: x ≈ y for y > 30.
    y = max(init_df - 2.0, 1e-3)
    init_df_unc = float(y if y > 30.0 else np.log(np.expm1(y)))
    init = np.array(
        [
            init_loc,
            np.log(max(init_scale, 1e-12)),
            init_df_unc,
            0.0,  # skew = 0
        ],
        dtype=float,
    )
    res = optimize.minimize(
        neg_ll, init, method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
    )
    loc = float(res.x[0])
    scale = float(np.exp(np.clip(res.x[1], -50.0, 30.0)))
    df_unc = float(np.clip(res.x[2], -50.0, 50.0))
    df = float(2.0 + np.logaddexp(0.0, df_unc))
    skew = float(np.tanh(res.x[3]))
    ll = -float(res.fun)
    return _make_fitted(
        "skewed_t",
        {"loc": loc, "scale": scale, "df": df, "skew": skew},
        ll,
        n,
    )


def fit_best_distribution(
    x: pd.Series | np.ndarray,
    *,
    candidates: tuple[str, ...] = ("normal", "student_t", "skewed_t"),
    criterion: Literal["aic", "bic"] = "aic",
) -> FittedDistribution:
    """Fit each candidate distribution and return the one with the lowest IC.

    Ties are broken by parameter count (simpler model wins).
    """
    fitted = []
    for name in candidates:
        if name == "normal":
            fitted.append(fit_normal(x))
        elif name == "student_t":
            fitted.append(fit_student_t(x))
        elif name == "skewed_t":
            fitted.append(fit_skewed_t(x))
        else:
            raise ValueError(f"unknown candidate {name!r}")
    fitted.sort(key=lambda f: (getattr(f, criterion), f.n_params))
    return fitted[0]


# ──────────────────────────────────────────────────────────────────
# Hansen 1994 skewed-t internals
# ──────────────────────────────────────────────────────────────────


def _hansen_skewed_t_constants(eta: float, lam: float) -> tuple[float, float, float]:
    """Compute (a, b, c) constants from Hansen (1994)."""
    # log c = log Γ((η+1)/2) − 0.5 log(π(η-2)) − log Γ(η/2)
    log_c = (
        gammaln((eta + 1.0) / 2.0)
        - 0.5 * np.log(np.pi * (eta - 2.0))
        - gammaln(eta / 2.0)
    )
    c = float(np.exp(log_c))
    a = 4.0 * lam * c * (eta - 2.0) / (eta - 1.0)
    b = float(np.sqrt(1.0 + 3.0 * lam**2 - a**2))
    return a, b, c


def _hansen_skewed_t_pdf(
    x: float | np.ndarray,
    mu: float,
    sigma: float,
    eta: float,
    lam: float,
) -> np.ndarray:
    """Hansen (1994) skewed-t density at ``x``.

    Parameters
    ----------
    x : float or array
    mu, sigma : location and scale
    eta : degrees of freedom (> 2)
    lam : skewness (-1 < lam < 1)
    """
    if not (eta > 2.0):
        return np.zeros_like(np.asarray(x, dtype=float))
    if not (-1.0 < lam < 1.0):
        return np.zeros_like(np.asarray(x, dtype=float))
    if sigma <= 0.0:
        return np.zeros_like(np.asarray(x, dtype=float))
    a, b, c = _hansen_skewed_t_constants(eta, lam)
    z = (np.asarray(x, dtype=float) - mu) / sigma
    threshold = -a / b
    denom = np.where(z < threshold, 1.0 - lam, 1.0 + lam)
    inner = ((b * z + a) / denom) ** 2 / (eta - 2.0)
    pdf_z = b * c * (1.0 + inner) ** (-(eta + 1.0) / 2.0)
    return pdf_z / sigma


def _hansen_skewed_t_sample(
    *,
    size: int,
    mu: float,
    sigma: float,
    eta: float,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from Hansen-skewed-t via inverse-CDF on a fine grid (good enough for Stage 4)."""
    # Build a CDF on a fine z-grid and invert via interp.
    z_grid = np.linspace(-15.0, 15.0, 4001)
    pdf = _hansen_skewed_t_pdf(z_grid, mu=0.0, sigma=1.0, eta=eta, lam=lam)
    cdf = np.cumsum(pdf) * (z_grid[1] - z_grid[0])
    cdf /= cdf[-1]
    u = rng.uniform(size=size)
    z = np.interp(u, cdf, z_grid)
    return mu + sigma * z


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _as_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def _make_fitted(
    name: str, params: dict[str, float], log_likelihood: float, n: int
) -> FittedDistribution:
    k = len(params)
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = k * np.log(n) - 2.0 * log_likelihood
    return FittedDistribution(
        name=name,
        params=params,
        log_likelihood=float(log_likelihood),
        aic=float(aic),
        bic=float(bic),
        n_obs=int(n),
    )


# ──────────────────────────────────────────────────────────────────
# Cross-strategy residual covariance
# ──────────────────────────────────────────────────────────────────


def cross_strategy_covariance(
    residuals_panel: pd.DataFrame,
    *,
    method: Literal["empirical", "shrinkage"] = "empirical",
    shrinkage_intensity: float = 0.1,
) -> pd.DataFrame:
    r"""Cross-strategy residual covariance matrix.

    Parameters
    ----------
    residuals_panel
        DataFrame whose columns are strategy names and rows are aligned residuals.
    method
        ``'empirical'`` returns the sample covariance; ``'shrinkage'`` returns
        a Ledoit-Wolf-style convex combination of the sample covariance and a
        diagonal target with magnitude ``shrinkage_intensity ∈ [0, 1]``.
    shrinkage_intensity
        Mixing weight on the diagonal target — ``0.0`` is pure empirical,
        ``1.0`` is fully diagonal.
    """
    if not isinstance(residuals_panel, pd.DataFrame):
        raise TypeError("residuals_panel must be a DataFrame.")
    if residuals_panel.isna().any().any():
        raise ValueError("residuals_panel has NaN values.")
    cov_emp = residuals_panel.cov()
    if method == "empirical":
        return cov_emp
    if method == "shrinkage":
        if not (0.0 <= shrinkage_intensity <= 1.0):
            raise ValueError(
                f"shrinkage_intensity must lie in [0, 1], got {shrinkage_intensity}"
            )
        diag = pd.DataFrame(
            np.diag(np.diag(cov_emp.to_numpy())),
            index=cov_emp.index,
            columns=cov_emp.columns,
        )
        return (1.0 - shrinkage_intensity) * cov_emp + shrinkage_intensity * diag
    raise ValueError(f"unknown method {method!r}")


__all__ = [
    "ArchLMResult",
    "FittedDistribution",
    "JarqueBeraResult",
    "LjungBoxResult",
    "arch_lm",
    "cross_strategy_covariance",
    "fit_best_distribution",
    "fit_normal",
    "fit_skewed_t",
    "fit_student_t",
    "jarque_bera",
    "ljung_box",
]
