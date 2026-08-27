r"""
Okunev & White (2003) generalized return-unsmoothing model (Stage 1).

Method
------
Okunev & White (2003), "Hedge Fund Risk Factors and Value at Risk of Credit
Trading Strategies" (Section III.A) generalize Geltner desmoothing from a
single lag to an arbitrary lag :math:`m`. The observed (reported) return is
assumed to be a linear combination of the true current return and lagged
*reported* returns (their Eq. 3):

.. math::

    r_{0,t} = (1 - \alpha)\, r_{m,t} + \sum_i \beta_i\, r_{0,t-i},
    \qquad (1 - \alpha) = \sum_i \beta_i .

The procedure inverts this **without assuming a time-series process** for the
true returns — the only assumption is that the observed autocorrelation is an
artifact of smoothing. It is therefore a **pure time-series method**: it uses
only the return series' own autocorrelation structure and does *not* use
factor returns at all (unlike :class:`RudinReparamSmoothing` or the AR(1)
Bayesian desmoother).

The building block is a Geltner-style adjustment at lag :math:`k` (Eq. 24):

.. math::

    r_{k,t} = \frac{r_{k-1,t} - c_k\, r_{k-1,t-k}}{1 - c_k},

where :math:`c_k` is chosen so the lag-:math:`k` autocorrelation of the
adjusted series hits a target :math:`d_k` (default 0). Setting
:math:`a_{k,k}(c_k) = d_k` yields a quadratic in :math:`c_k` (Eqs. 5-6, 13-14,
27) which we build from the autocorrelation condition and solve numerically
(see :func:`_solve_c`) rather than transcribing the paper's closed forms.

Re-cleaning
-----------
Removing autocorrelation at lag :math:`k` **reintroduces** autocorrelation at
lower lags (Eq. 19). The defining feature of Okunev-White over naive
sequential lag removal is the inner *re-cleaning* loop: after adjusting the
current target lag we re-remove every lower lag, and iterate until *all*
targeted lags fall below the tolerance simultaneously (Eqs. 20-22). Skipping
this step leaves residual autocorrelation at the lower lags — it is the single
most important correctness requirement of the method.

Observation loss
----------------
Each adjustment at lag :math:`k` references :math:`r_{t-k}` and therefore
loses the first :math:`k` observations of whatever series it is applied to.
The reconstructed series is returned on the trimmed (valid) index and the
cumulative number of observations lost is reported in the diagnostics. To keep
the loss bounded we *skip* an adjustment whenever the lag is already within
``tol`` of its target (a near-zero ``c`` that would consume observations for a
no-op); convergence is declared once no further adjustment is needed.

References
----------
Okunev, J., & White, D. (2003). Hedge Fund Risk Factors and Value at Risk of
Credit Trading Strategies. Working paper (SSRN 460641).
Geltner, D. (1991, 1993). Return unsmoothing for appraisal-based returns.
Brooks, C., & Kat, H. (2001). The statistical properties of hedge fund index
returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from ..core.config import FallbackPolicy
from ..core.protocols import DesmoothedResult, Frequency

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

_MIN_LAGS = 1
_MAX_LAGS = 8
_ALLOWED_ROOT_SELECTION = frozenset({"stable", "smaller", "larger"})
# Below this |c| an adjustment is treated as a no-op and skipped (avoids
# consuming observations to remove autocorrelation that is already ~0).
_C_SKIP = 1e-8
# Guard against the 1/(1-c) transform blowing up.
_ONE_MINUS_C_FLOOR = 1e-8


class _NoRealRoot(Exception):
    """Raised by :func:`_solve_c` when the real-root condition is violated.

    Signals that the requested target autocorrelation cannot be achieved at
    this lag by a real Geltner-style adjustment (paper's Eqs. 7 / 15 / 28).
    Caught by :meth:`OkunevWhiteSmoothing.fit` and routed through the
    configured :class:`FallbackPolicy`.
    """

    def __init__(self, lag: int, a_k: float, a_2k: float, d: float) -> None:
        self.lag = lag
        self.a_k = a_k
        self.a_2k = a_2k
        self.d = d
        super().__init__(
            f"No real root for the lag-{lag} adjustment: the real-root "
            f"condition (a_k - d)^2 <= (1 + a_2k - 2 d a_k)^2 / 4 is violated "
            f"with a_k={a_k:.4f}, a_2k={a_2k:.4f}, d={d:.4f}."
        )


# ──────────────────────────────────────────────────────────────────
# ACF helper
# ──────────────────────────────────────────────────────────────────


def _sample_acf(x: np.ndarray, lag: int) -> float:
    r"""Biased sample autocorrelation at ``lag``.

    Uses the standard biased estimator (denominator over all ``N``
    observations), matching the convention in
    :func:`desmoothing.geltner_classic._lag1_autocorr` and statsmodels'
    ``acf(..., adjusted=False)``:

    .. math::

        \hat a_k = \frac{\sum_{t=k}^{N-1} (x_t - \bar x)(x_{t-k} - \bar x)}
                        {\sum_{t=0}^{N-1} (x_t - \bar x)^2} .

    Parameters
    ----------
    x
        1-D series.
    lag
        Non-negative lag. ``lag == 0`` returns 1.0.

    Returns
    -------
    float
        The sample autocorrelation, or ``0.0`` for a (numerically) constant
        series or when ``lag >= len(x)``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if lag == 0:
        return 1.0
    if lag < 0:
        raise ValueError(f"lag must be non-negative, got {lag}")
    if lag >= n:
        return 0.0
    # Robust constant-series detection (float noise in mean() otherwise leaks).
    if float(np.ptp(x)) <= np.finfo(float).eps * max(1.0, float(np.abs(x).max())):
        return 0.0
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    if denom <= 0.0:
        return 0.0
    num = float(np.sum(xc[lag:] * xc[:-lag]))
    return num / denom


# ──────────────────────────────────────────────────────────────────
# Quadratic c-solver
# ──────────────────────────────────────────────────────────────────


def _solve_c(
    a_k: float,
    a_2k: float,
    d: float = 0.0,
    root_selection: str = "stable",
) -> float:
    r"""Solve for the adjustment coefficient ``c`` at a given lag.

    Builds the quadratic ``A c^2 + B c + C = 0`` from the condition that the
    lag-:math:`k` autocorrelation of the adjusted series equals the target
    :math:`d` (paper's Eq. 5 rearranged):

    .. math::

        A = a_k - d, \quad
        B = -(1 + a_{2k}) + 2 d\, a_k, \quad
        C = a_k - d,

    and returns the economically sensible root. Because :math:`A = C`, the two
    roots are reciprocals; the ``"stable"`` / ``"smaller"`` selection returns
    the root with the smaller magnitude (:math:`|c| \le 1`), which keeps
    :math:`1 - c` away from zero. For a genuine AR(1) input with
    :math:`a_{2k} = a_k^2` and :math:`d = 0` this yields :math:`c = a_k`
    (the Geltner special case).

    Parameters
    ----------
    a_k
        Sample autocorrelation at the lag being adjusted.
    a_2k
        Sample autocorrelation at twice that lag.
    d
        Target autocorrelation for the lag (default ``0.0`` = remove entirely).
    root_selection
        ``"stable"`` / ``"smaller"`` → smaller-magnitude root (default);
        ``"larger"`` → larger-magnitude root.

    Returns
    -------
    float
        The chosen ``c``. Returns ``0.0`` when the series is already at the
        target (``a_k == d``), i.e. no adjustment is needed.

    Raises
    ------
    _NoRealRoot
        If the discriminant is negative (real-root condition violated).
    ValueError
        If ``root_selection`` is not recognised, or the chosen root drives
        ``1 - c`` below the numerical floor.
    """
    if root_selection not in _ALLOWED_ROOT_SELECTION:
        raise ValueError(
            f"root_selection must be one of {sorted(_ALLOWED_ROOT_SELECTION)}, "
            f"got {root_selection!r}"
        )
    A = a_k - d
    B = -(1.0 + a_2k) + 2.0 * d * a_k
    C = a_k - d

    # Degenerate (already at target): A = C = 0 ⇒ the quadratic collapses and
    # c = 0 is the no-op solution.
    if abs(A) < 1e-15:
        return 0.0

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        # Distinguish genuine sign-negativity from float noise near zero.
        if disc < -1e-12 * max(1.0, B * B):
            raise _NoRealRoot(lag=-1, a_k=a_k, a_2k=a_2k, d=d)
        disc = 0.0

    sqrt_disc = float(np.sqrt(disc))
    root_plus = (-B + sqrt_disc) / (2.0 * A)
    root_minus = (-B - sqrt_disc) / (2.0 * A)

    if root_selection == "larger":
        chosen = root_plus if abs(root_plus) >= abs(root_minus) else root_minus
    else:  # "stable" / "smaller"
        chosen = root_plus if abs(root_plus) <= abs(root_minus) else root_minus

    if abs(1.0 - chosen) < _ONE_MINUS_C_FLOOR:
        raise ValueError(
            f"selected c={chosen:.6g} drives (1 - c) below the numerical floor "
            f"{_ONE_MINUS_C_FLOOR}; the adjustment would be unstable."
        )
    return float(chosen)


# ──────────────────────────────────────────────────────────────────
# Single Geltner-style adjustment
# ──────────────────────────────────────────────────────────────────


def _adjust(series: np.ndarray, lag: int, c: float) -> np.ndarray:
    r"""Apply one Geltner-style adjustment at ``lag`` (Eq. 24).

    .. math::

        r^{\text{new}}_t = \frac{r_t - c\, r_{t-lag}}{1 - c}, \quad t \ge lag.

    The first ``lag`` observations are consumed by the shift, so the returned
    array has length ``len(series) - lag``.

    Parameters
    ----------
    series
        Current 1-D return series.
    lag
        Positive lag of the adjustment.
    c
        Adjustment coefficient (from :func:`_solve_c`).

    Returns
    -------
    numpy.ndarray
        The adjusted series, ``lag`` observations shorter.
    """
    s = np.asarray(series, dtype=float)
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if s.size <= lag:
        raise ValueError(
            f"series length {s.size} too short for a lag-{lag} adjustment."
        )
    denom = 1.0 - c
    if abs(denom) < _ONE_MINUS_C_FLOOR:
        raise ValueError(f"(1 - c) = {denom:.3e} too close to zero.")
    return (s[lag:] - c * s[:-lag]) / denom


# ──────────────────────────────────────────────────────────────────
# The smoother
# ──────────────────────────────────────────────────────────────────


@dataclass
class OkunevWhiteSmoothing:
    r"""Generalized return-unsmoothing of Okunev & White (2003).

    An iterative extension of Geltner desmoothing that removes autocorrelation
    up to an arbitrary lag :math:`m` — not merely first-order — by repeatedly
    applying a Geltner-style adjustment at each lag and re-cleaning perturbed
    lower lags until all targeted autocorrelations fall below a threshold. Pure
    time-series method: uses only the return series' own autocorrelation
    structure, no factors. Especially suited to hedge fund indices (the
    paper's application), where higher-order serial correlation is common.

    Parameters
    ----------
    n_lags
        Number of autocorrelation orders to remove, :math:`m \in [1, 8]`.
        Default 4 (the paper eliminated the first four autocorrelations for
        hedge fund indices).
    target_acf
        Desired autocorrelation at lags ``1..n_lags``. A sequence of length
        ``n_lags`` (entry ``i`` targets lag ``i+1``). ``None`` (default) means
        all zeros — remove all autocorrelation up to lag ``n_lags``. Supply a
        non-zero vector to leave a deliberate residual at chosen lags.
    tol
        Absolute convergence-tolerance floor: a lag is "clean" once
        ``|acf(lag) - target| < max(tol, band)``. Default ``1e-4``.
    acf_significance
        Bartlett-style significance ``z``-score used to build the per-lag
        tolerance band ``band = z / sqrt(n_current)``. A lag is only adjusted
        when its sample autocorrelation is *statistically distinguishable* from
        its target; this stops the procedure chasing finite-sample sampling
        noise (which would over-adjust genuinely-uncorrelated data and consume
        observations for no benefit). Default ``1.96`` (≈95%). Set to ``0`` to
        fall back to the raw ``tol`` (in-sample zeroing of the sample ACF).
        This is a deliberate, documented small-sample refinement of the paper's
        idealised "drive the autocorrelation to zero" statement.
    max_iter
        Maximum outer re-cleaning iterations per target lag. Default 100.
    root_selection
        Which quadratic root to take: ``"stable"`` (smaller magnitude, the
        default and economically sensible choice), ``"smaller"`` (alias), or
        ``"larger"``.
    fallback_policy
        Behaviour when the real-root condition fails at some lag
        (:class:`FallbackPolicy`): ``STRICT`` raises, ``WARN`` skips that lag
        and continues, ``AUTO`` removes only the solvable lags and flags that
        the requested depth was not reached. Default ``WARN``.
    periods_per_year
        Annualisation factor. If ``None`` (default) it is inferred from the
        return index frequency (quarterly → 4, monthly → 12, weekly → 52,
        daily → 252).
    n_bootstrap
        If ``> 0``, number of moving-block bootstrap resamples used to build a
        confidence interval on the variance-inflation ratio (reported in
        ``posterior_summary``). Default 0 (skip — keeps ``fit`` fast and
        deterministic).
    bootstrap_seed
        Seed for the bootstrap resampler (only used when ``n_bootstrap > 0``).
    """

    n_lags: int = 4
    target_acf: "list[float] | tuple[float, ...] | np.ndarray | None" = None
    tol: float = 1e-4
    acf_significance: float = 1.96
    max_iter: int = 100
    patience: int = 8
    root_selection: str = "stable"
    fallback_policy: "FallbackPolicy | str" = FallbackPolicy.WARN
    periods_per_year: int | None = None
    n_bootstrap: int = 0
    bootstrap_seed: int | None = 0

    _target: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n_lags, (int, np.integer)):
            raise TypeError(f"n_lags must be int, got {type(self.n_lags).__name__}")
        if not (_MIN_LAGS <= self.n_lags <= _MAX_LAGS):
            raise ValueError(
                f"n_lags must lie in [{_MIN_LAGS}, {_MAX_LAGS}], got {self.n_lags}"
            )
        if self.root_selection not in _ALLOWED_ROOT_SELECTION:
            raise ValueError(
                f"root_selection must be one of {sorted(_ALLOWED_ROOT_SELECTION)}, "
                f"got {self.root_selection!r}"
            )
        if self.tol <= 0.0:
            raise ValueError(f"tol must be > 0, got {self.tol}")
        if self.acf_significance < 0.0:
            raise ValueError(
                f"acf_significance must be >= 0, got {self.acf_significance}"
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.periods_per_year is not None and self.periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year must be > 0 or None, got {self.periods_per_year}"
            )
        if self.n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be >= 0, got {self.n_bootstrap}")
        # Normalise the fallback policy to the enum.
        self.fallback_policy = (
            self.fallback_policy
            if isinstance(self.fallback_policy, FallbackPolicy)
            else FallbackPolicy(self.fallback_policy)
        )
        # Build the per-lag target vector, indexed 1..n_lags (index 0 unused).
        if self.target_acf is None:
            target = np.zeros(self.n_lags, dtype=float)
        else:
            target = np.asarray(self.target_acf, dtype=float).reshape(-1)
            if target.size != self.n_lags:
                raise ValueError(
                    f"target_acf must have length n_lags = {self.n_lags}, "
                    f"got {target.size}."
                )
        # Prepend a placeholder so self._target[lag] indexes lag directly.
        self._target = np.concatenate([[np.nan], target])

    # ── target accessor ───────────────────────────────────────────────

    def _target_for(self, lag: int) -> float:
        """Desired autocorrelation for ``lag`` (0.0 for lags beyond n_lags)."""
        if 1 <= lag < self._target.size:
            return float(self._target[lag])
        return 0.0

    def _band(self, n: int) -> float:
        """Per-lag tolerance band ``max(tol, z / sqrt(n))`` for a length-``n`` series.

        Autocorrelations within this band of their target are treated as
        already achieved (not adjusted), so the procedure does not chase
        finite-sample sampling noise.
        """
        n = max(int(n), 1)
        return max(self.tol, self.acf_significance / np.sqrt(n))

    # ── protocol: fit ─────────────────────────────────────────────────

    def fit(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame | None = None,
        priors: dict[str, Any] | None = None,
    ) -> DesmoothedResult:
        """Run the iterative unsmoothing.

        ``factor_returns`` and ``priors`` are accepted for
        :class:`SmoothingModel` protocol compatibility but are **not used** —
        this is a pure time-series method. An informational note is recorded
        in ``diagnostics['notes']`` if either is supplied.

        Parameters
        ----------
        observed_returns
            The smoothed (observed) return series at native frequency.
        factor_returns
            Ignored. Present for protocol compatibility.
        priors
            Ignored. Present for protocol compatibility.

        Returns
        -------
        DesmoothedResult
            ``true_returns`` is the reconstructed series on the trimmed valid
            index; ``smoothing_params`` holds the full ``c`` trace, per-lag net
            adjustment, implied smoothing kernel, ``n_lags`` and
            iterations-to-convergence; ``diagnostics`` holds ACF before/after,
            variance inflation (formula and empirical), annualised vols,
            Ljung-Box, and convergence flags.
        """
        if not isinstance(observed_returns, pd.Series):
            raise TypeError(
                f"observed_returns must be a pandas Series, got "
                f"{type(observed_returns).__name__}."
            )
        if observed_returns.isna().any():
            raise ValueError("observed_returns contains NaN.")
        r0 = observed_returns.to_numpy(dtype=float)
        n_obs = r0.size
        # Need enough data to estimate the lag-2m autocorrelation used by the
        # deepest solver, with a little slack for the trimming.
        min_needed = 2 * self.n_lags + 3
        if n_obs < min_needed:
            raise ValueError(
                f"observed_returns has {n_obs} observations; need >= {min_needed} "
                f"for n_lags={self.n_lags}."
            )

        notes: list[str] = []
        if factor_returns is not None and getattr(factor_returns, "empty", True) is False:
            notes.append(
                "factor_returns supplied but ignored — Okunev-White is a pure "
                "time-series method and does not use factors."
            )
        if priors:
            notes.append("priors supplied but ignored — this model uses no priors.")

        ppy = self._resolve_ppy(observed_returns.index)

        # ── run the core algorithm ────────────────────────────────────
        working, c_trace, per_lag, convergence = self._unsmooth(r0)
        n_lost = n_obs - working.size

        true_index = observed_returns.index[n_lost:]
        true_returns = pd.Series(working, index=true_index, name=observed_returns.name)

        # ── ACF before/after (lags 1..n_lags+2) ───────────────────────
        acf_lags = list(range(1, self.n_lags + 3))
        acf_before = {lag: _sample_acf(r0, lag) for lag in acf_lags}
        acf_after = {lag: _sample_acf(working, lag) for lag in acf_lags}

        # ── variance inflation ────────────────────────────────────────
        var_obs = float(np.var(r0, ddof=1))
        var_true = float(np.var(working, ddof=1))
        var_inflation_empirical = var_true / var_obs if var_obs > 0 else float("nan")
        # Formula (Eq. 29): product of per-adjustment factors.
        var_factor_product = 1.0
        for step in c_trace:
            var_factor_product *= step["var_factor"]
        var_inflation_formula = var_factor_product
        var_discrepancy = abs(var_inflation_formula - var_inflation_empirical)

        # ── implied smoothing kernel (generalised Eq. 23) ─────────────
        kernel = _implied_kernel(c_trace)

        # ── annualised vols ───────────────────────────────────────────
        vol_obs_ann = float(np.std(r0, ddof=1) * np.sqrt(ppy))
        vol_true_ann = float(np.std(working, ddof=1) * np.sqrt(ppy))

        # ── Ljung-Box on the unsmoothed series ────────────────────────
        lb_lag = min(self.n_lags, max(1, working.size // 4))
        lb = acorr_ljungbox(working, lags=[lb_lag], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])

        # ── convergence flags ─────────────────────────────────────────
        all_converged = all(info["converged"] for info in convergence.values())
        residual_acf = {lag: acf_after[lag] for lag in range(1, self.n_lags + 1)}
        after_band = self._band(working.size)
        target_met = {
            lag: bool(abs(acf_after[lag] - self._target_for(lag)) < after_band)
            for lag in range(1, self.n_lags + 1)
        }
        skipped_lags = sorted(
            lag for lag, info in convergence.items() if info.get("skipped", False)
        )
        if skipped_lags:
            notes.append(
                f"lags {skipped_lags} were skipped (real-root condition failed); "
                f"fallback_policy={self.fallback_policy.value}."
            )
        if self.fallback_policy is FallbackPolicy.AUTO and skipped_lags:
            notes.append(
                "AUTO fallback: requested unsmoothing depth was not fully "
                f"reached — removed lags "
                f"{sorted(set(range(1, self.n_lags + 1)) - set(skipped_lags))}."
            )

        # ── bootstrap CI on the variance-inflation ratio ──────────────
        bootstrap_summary = self._bootstrap_variance_inflation(r0)

        # ── assemble ──────────────────────────────────────────────────
        smoothing_params: dict[str, Any] = {
            "c_trace": c_trace,
            "param_array": np.array(
                [[step["lag"], step["c"]] for step in c_trace], dtype=float
            ).reshape(-1, 2),
            "per_lag_net_c": per_lag,
            "implied_kernel": kernel,
            "n_lags": int(self.n_lags),
            "target_acf": self._target[1:].copy(),
            "iterations_to_convergence": {
                lag: info["iterations"] for lag, info in convergence.items()
            },
            "n_observations_lost": int(n_lost),
        }

        posterior_summary: dict[str, Any] = {
            "estimator": "okunev_white_frequentist",
            "note": (
                "Frequentist time-series unsmoothing — there is no posterior. "
                "Point estimates only."
            ),
            "variance_inflation": {
                "formula": var_inflation_formula,
                "empirical": var_inflation_empirical,
            },
            "variance_inflation_bootstrap": bootstrap_summary,
        }

        diagnostics: dict[str, Any] = {
            "method": "okunev_white",
            "n_observations": int(n_obs),
            "n_observations_lost": int(n_lost),
            "n_effective": int(working.size),
            "periods_per_year": int(ppy),
            "acf_before": acf_before,
            "acf_after": acf_after,
            "residual_acf": residual_acf,
            "variance_inflation_formula": var_inflation_formula,
            "variance_inflation_empirical": var_inflation_empirical,
            "variance_inflation_discrepancy": var_discrepancy,
            "vol_annualised_observed": vol_obs_ann,
            "vol_annualised_unsmoothed": vol_true_ann,
            "ljung_box_stat": lb_stat,
            "ljung_box_pvalue": lb_pvalue,
            "ljung_box_lags_tested": lb_lag,
            "mean_observed": float(np.mean(r0)),
            "mean_unsmoothed": float(np.mean(working)),
            "n_adjustments": len(c_trace),
            "converged": bool(all_converged),
            "target_met": target_met,
            "skipped_lags": skipped_lags,
            "convergence_detail": convergence,
            "notes": notes,
        }

        return DesmoothedResult(
            true_returns=true_returns,
            smoothing_params=smoothing_params,
            posterior_summary=posterior_summary,
            diagnostics=diagnostics,
        )

    # ── core iterative algorithm ──────────────────────────────────────

    def _unsmooth(
        self, r0: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[int, float], dict[int, dict[str, Any]]]:
        """Iteratively remove autocorrelation up to ``n_lags``.

        Removes lag 1, then lag 2 (re-cleaning lag 1), then lag 3 (re-cleaning
        1-2), and so on. Each stage runs a re-cleaning cascade and keeps the
        *best-so-far* iterate (minimum residual autocorrelation over the lags
        targeted so far), because the coupled cascade can oscillate around a
        finite-sample floor rather than reaching ``tol`` exactly (the spec's
        convergence guard). Best-so-far also bounds observation loss.

        Returns the reconstructed (trimmed) series, the full ordered ``c``
        trace, the per-lag net adjustment, and per-target-lag convergence info.
        """
        working = r0.copy()
        c_trace: list[dict[str, Any]] = []
        convergence: dict[int, dict[str, Any]] = {}

        for target_lag in range(1, self.n_lags + 1):
            working, c_trace, info = self._run_stage(working, c_trace, target_lag)
            convergence[target_lag] = info

        per_lag = _net_per_lag(c_trace, self.n_lags)
        return working, c_trace, per_lag, convergence

    def _run_stage(
        self,
        working: np.ndarray,
        c_trace: list[dict[str, Any]],
        target_lag: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
        """Run one target-lag re-cleaning cascade, returning the best iterate.

        Parameters
        ----------
        working
            Current reconstructed series entering this stage.
        c_trace
            Adjustment trace accumulated so far (not mutated; a snapshot is
            extended and returned).
        target_lag
            The lag whose autocorrelation this stage removes (re-cleaning all
            lower lags each iteration).

        Returns
        -------
        (best_working, best_trace, info)
        """
        info: dict[str, Any] = {"iterations": 0, "converged": False, "skipped": False}

        def resid(series: np.ndarray) -> float:
            return max(
                abs(_sample_acf(series, k) - self._target_for(k))
                for k in range(1, target_lag + 1)
            )

        best_working = working
        best_trace = list(c_trace)
        best_resid = resid(working)
        if best_resid < self._band(working.size):
            info["converged"] = True
            info["residual"] = float(best_resid)
            return best_working, best_trace, info

        cur_working = working
        cur_trace = list(c_trace)
        stale = 0
        lags_to_process = [target_lag] + list(range(1, target_lag))

        for iteration in range(1, self.max_iter + 1):
            info["iterations"] = iteration
            iter_working = cur_working
            iter_trace = list(cur_trace)
            applied_any = False
            aborted = False

            for lag in lags_to_process:
                outcome = self._process_lag(iter_working, lag)
                if outcome is None:
                    continue  # already within tol at this lag / no-op
                if outcome == "no_real_root":
                    if self.fallback_policy is FallbackPolicy.STRICT:
                        raise ValueError(
                            f"Okunev-White: real-root condition failed at lag "
                            f"{lag} (Eq. 7/15/28). Cannot achieve the target "
                            f"autocorrelation with a real adjustment. "
                            f"fallback_policy=STRICT."
                        )
                    # WARN / AUTO: abandon this stage, keep the best-so-far.
                    info["skipped"] = True
                    aborted = True
                    break
                iter_working, step = outcome
                iter_trace.append(step)
                applied_any = True

            if aborted:
                break

            cur_working = iter_working
            cur_trace = iter_trace
            cur_resid = resid(cur_working)

            if cur_resid < best_resid - 1e-12:
                best_resid = cur_resid
                best_working = cur_working
                best_trace = list(cur_trace)
                stale = 0
            else:
                stale += 1

            if best_resid < self._band(best_working.size):
                info["converged"] = True
                break
            if not applied_any:
                # Nothing left to adjust within the band: floor reached.
                info["converged"] = True
                break
            if stale >= self.patience:
                # Oscillating around a finite-sample floor — stop, keep best.
                break

        info["converged"] = (
            bool(best_resid < self._band(best_working.size)) and not info["skipped"]
        )
        info["residual"] = float(best_resid)
        return best_working, best_trace, info

    def _process_lag(
        self,
        working: np.ndarray,
        lag: int,
    ) -> "None | str | tuple[np.ndarray, dict[str, Any]]":
        """Attempt one adjustment at ``lag`` on ``working``.

        Returns
        -------
        None
            The lag is already within ``tol`` of its target (no-op, no
            observation loss).
        ``"no_real_root"``
            The real-root condition failed (caller applies the fallback).
        ``(new_working, step)``
            The adjusted (trimmed) series and the recorded step dict.
        """
        d = self._target_for(lag)
        a_k = _sample_acf(working, lag)
        if abs(a_k - d) < self._band(working.size):
            return None
        # Need lag-2k autocorrelation; guard the series length.
        if working.size <= 2 * lag:
            return None
        a_2k = _sample_acf(working, 2 * lag)
        try:
            c = _solve_c(a_k, a_2k, d, self.root_selection)
        except _NoRealRoot:
            return "no_real_root"
        if abs(c) < _C_SKIP:
            return None
        new_working = _adjust(working, lag, c)
        var_factor = (1.0 + c * c - 2.0 * c * a_k) / ((1.0 - c) ** 2)
        step = {
            "lag": int(lag),
            "c": float(c),
            "a_pre": float(a_k),
            "a_2k_pre": float(a_2k),
            "target": float(d),
            "var_factor": float(var_factor),
        }
        return new_working, step

    # ── protocol: desmooth ────────────────────────────────────────────

    def desmooth(
        self,
        observed_returns: np.ndarray,
        smoothing_params: np.ndarray,
    ) -> np.ndarray:
        """Apply a known sequence of ``c``-adjustments to reconstruct returns.

        Parameters
        ----------
        observed_returns
            Observed return array (1-D, length ``N``).
        smoothing_params
            The ordered adjustment trace as a ``(k, 2)`` array whose rows are
            ``(lag, c)`` — e.g. ``result.smoothing_params['param_array']``. A
            list of ``(lag, c)`` pairs or the ``c_trace`` list of dicts is also
            accepted.

        Returns
        -------
        numpy.ndarray
            The reconstructed series. Each lag-``k`` step trims ``k`` leading
            observations, so the output is
            ``N - sum(lags)`` long.
        """
        steps = _coerce_param_steps(smoothing_params)
        r = np.asarray(observed_returns, dtype=float).reshape(-1)
        for lag, c in steps:
            if r.size <= lag:
                raise ValueError(
                    f"series too short ({r.size}) for a lag-{lag} adjustment during "
                    "desmooth; the supplied trace consumes more observations than "
                    "available."
                )
            r = _adjust(r, int(lag), float(c))
        return r

    # ── protocol: log_likelihood / log_prior ──────────────────────────

    def log_likelihood(
        self,
        smoothing_params: np.ndarray,
        observed_returns: np.ndarray,
        factor_returns: np.ndarray | None = None,
    ) -> float:
        """Gaussian log-likelihood of the unsmoothed residuals.

        Reconstructs the true-return series by applying ``smoothing_params``
        (a ``(lag, c)`` trace; see :meth:`desmooth`) and evaluates the
        Gaussian log-likelihood at the MLE mean and variance — i.e. treating
        the unsmoothed returns as iid Normal. Provided for cross-model
        comparison; ``factor_returns`` is ignored.
        """
        try:
            r = self.desmooth(observed_returns, smoothing_params)
        except (ValueError, TypeError):
            return -np.inf
        n = r.size
        if n < 2:
            return -np.inf
        var = float(np.var(r, ddof=0))
        if var <= 0.0:
            return -np.inf
        # Profile Gaussian ll at MLE (mean, var): -0.5 n (log(2π var) + 1).
        return float(-0.5 * n * (np.log(2.0 * np.pi * var) + 1.0))

    def log_prior(
        self,
        smoothing_params: np.ndarray,
        prior_config: dict[str, Any],
    ) -> float:
        """Return ``0.0`` — this model uses no priors. Present for protocol compliance."""
        return 0.0

    # ── internal helpers ──────────────────────────────────────────────

    def _resolve_ppy(self, index: pd.Index) -> int:
        if self.periods_per_year is not None:
            return int(self.periods_per_year)
        return _infer_periods_per_year(index)

    def _bootstrap_variance_inflation(self, r0: np.ndarray) -> dict[str, Any]:
        """Moving-block bootstrap CI on the empirical variance-inflation ratio.

        Returns a dict with ``ci_05``/``ci_95``/``mean``/``std``/``n`` when
        ``n_bootstrap > 0``, otherwise ``{'available': False}``.
        """
        if self.n_bootstrap <= 0:
            return {"available": False}
        n = r0.size
        block = max(self.n_lags + 1, int(round(n ** (1.0 / 3.0))))
        if block >= n:
            return {"available": False, "reason": "series too short for blocks"}
        rng = np.random.default_rng(self.bootstrap_seed)
        n_blocks = int(np.ceil(n / block))
        ratios: list[float] = []
        for _ in range(self.n_bootstrap):
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            sample = np.concatenate([r0[s : s + block] for s in starts])[:n]
            try:
                boot_working, _, _, _ = self._unsmooth(sample)
            except Exception:
                continue
            var_obs = float(np.var(sample, ddof=1))
            if var_obs <= 0.0:
                continue
            ratios.append(float(np.var(boot_working, ddof=1)) / var_obs)
        if not ratios:
            return {"available": False, "reason": "no successful resamples"}
        arr = np.asarray(ratios, dtype=float)
        return {
            "available": True,
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "ci_05": float(np.percentile(arr, 5)),
            "ci_95": float(np.percentile(arr, 95)),
            "n": int(arr.size),
            "block_length": int(block),
        }


# ──────────────────────────────────────────────────────────────────
# Free helpers
# ──────────────────────────────────────────────────────────────────


def _coerce_param_steps(smoothing_params: Any) -> list[tuple[int, float]]:
    """Normalise assorted ``smoothing_params`` encodings to a ``[(lag, c)]`` list."""
    if isinstance(smoothing_params, dict):
        if "param_array" in smoothing_params:
            arr = np.asarray(smoothing_params["param_array"], dtype=float).reshape(-1, 2)
            return [(int(row[0]), float(row[1])) for row in arr]
        if "c_trace" in smoothing_params:
            return [(int(s["lag"]), float(s["c"])) for s in smoothing_params["c_trace"]]
        raise ValueError(
            "smoothing_params dict must contain 'param_array' or 'c_trace'."
        )
    if isinstance(smoothing_params, (list, tuple)) and smoothing_params and isinstance(
        smoothing_params[0], dict
    ):
        return [(int(s["lag"]), float(s["c"])) for s in smoothing_params]
    arr = np.asarray(smoothing_params, dtype=float)
    if arr.ndim == 1:
        if arr.size == 0:
            return []
        if arr.size == 2:
            return [(int(arr[0]), float(arr[1]))]
        raise ValueError(
            "1-D smoothing_params must be a single (lag, c) pair; use a (k, 2) "
            "array for multiple adjustments."
        )
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [(int(row[0]), float(row[1])) for row in arr]
    raise ValueError(
        f"cannot interpret smoothing_params of shape {arr.shape} as (lag, c) rows."
    )


def _net_per_lag(c_trace: list[dict[str, Any]], n_lags: int) -> dict[int, float]:
    r"""Net adjustment per lag.

    The individual filters at a given lag ``k`` compose as
    :math:`\prod (1 - c_i L^k)/(1 - c_i)`; the "net" coefficient reported here
    is the single ``c`` that reproduces the *product of the gains*
    :math:`\prod 1/(1 - c_i)` at that lag, i.e.
    :math:`c_{\text{net}} = 1 - \prod_i (1 - c_i)`. This is a compact scalar
    summary (exact only when a single adjustment was applied at the lag).
    """
    out: dict[int, float] = {}
    for lag in range(1, n_lags + 1):
        gains = [1.0 - s["c"] for s in c_trace if s["lag"] == lag]
        if gains:
            prod = float(np.prod(gains))
            out[lag] = 1.0 - prod
    return out


def _implied_kernel(c_trace: list[dict[str, Any]]) -> dict[str, Any]:
    r"""Implied smoothing kernel — generalisation of the paper's Eq. 23.

    Every applied adjustment is an LTI filter
    :math:`(1 - c_i L^{k_i})/(1 - c_i)`, so the net transform from the observed
    series :math:`r_0` to the reconstructed series :math:`r_m` is the product
    of their transfer functions. Expanding the numerator product gives a finite
    polynomial :math:`P(L)` in the lag operator such that

    .. math::

        r_{m,t} = \frac{1}{\prod_i (1 - c_i)} \sum_k P_k\, r_{0,t-k},

    equivalently the paper's Eq. 23 form

    .. math::

        r_{0,t} = \Big(\prod_i (1 - c_i)\Big) r_{m,t} - \sum_{k \ge 1} P_k\, r_{0,t-k}.

    Returns
    -------
    dict
        ``observed_lag_weights`` — the polynomial coefficients :math:`P_k`
        (index = lag on :math:`r_0`); ``scale`` — :math:`1/\prod(1 - c_i)`;
        ``reconstruction`` — the normalised weights ``scale * P_k`` giving
        ``r_m,t = Σ_k w_k r_0,{t-k}``.
    """
    poly = np.array([1.0])  # coefficient on r0_{t-0}
    gain = 1.0
    for step in c_trace:
        lag = int(step["lag"])
        c = float(step["c"])
        factor = np.zeros(lag + 1)
        factor[0] = 1.0
        factor[lag] = -c
        poly = np.convolve(poly, factor)
        gain *= 1.0 - c
    scale = 1.0 / gain if gain != 0.0 else float("inf")
    reconstruction = scale * poly
    return {
        "observed_lag_weights": poly,
        "scale": float(scale),
        "reconstruction": reconstruction,
        "denominator_gain": float(gain),
    }


def _infer_periods_per_year(index: pd.Index) -> int:
    """Infer periods per year from a pandas index; default monthly (12)."""
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq or pd.infer_freq(index)
        if freq is not None:
            token = str(freq).upper()
            if token.startswith(("Q", "BQ")) or "Q" in token[:3]:
                return Frequency.QUARTERLY.periods_per_year
            if token.startswith(("M", "BM", "MS")) or "M" in token[:3]:
                return Frequency.MONTHLY.periods_per_year
            if token.startswith("W"):
                return Frequency.WEEKLY.periods_per_year
            if token.startswith(("B", "D", "C")):
                return Frequency.DAILY.periods_per_year
        if len(index) >= 2:
            spacings = np.diff(index.view("i8")) / 1e9 / 86400.0
            median_days = float(np.median(spacings))
            if median_days >= 80:
                return Frequency.QUARTERLY.periods_per_year
            if median_days >= 25:
                return Frequency.MONTHLY.periods_per_year
            if median_days >= 5:
                return Frequency.WEEKLY.periods_per_year
            return Frequency.DAILY.periods_per_year
    return Frequency.MONTHLY.periods_per_year


__all__ = ["OkunevWhiteSmoothing"]
