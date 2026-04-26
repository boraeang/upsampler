r"""
Threshold (regime-switching) AR(1) desmoother for private credit (Stage 1, Model 3).

Model
-----
The smoothing intensity λ depends on a binary regime indicator that the
caller supplies (typically a high-yield OAS level crossing a threshold)::

    s_t = (1 - λ_{regime_t}) · r_t + λ_{regime_t} · s_{t-1}     (smoothing)
    r_t = β · F_t + α + ε_t,    ε_t ∼ N(0, σ_ε²)               (factor model on truth)

with two distinct values ``λ_normal`` (regime = 0) and ``λ_stress``
(regime = 1). Empirically credit indices show two very different
appraisal regimes — loans carried at par for quarters in calm markets
(``λ_normal`` ≈ 0.6–0.9) and abruptly marked-to-reality during stress
(``λ_stress`` ≈ 0–0.3). A single-regime AR(1) averages those two and
ends up under-correcting in normal times and over-correcting in crises.

Bayesian estimation is a 2-D extension of :class:`AR1BayesianSmoother`:

    * 2-D grid over ``(λ_normal, λ_stress) ∈ [floor, ceil]²``.
    * For each pair, the desmoothed series ``r(λ_n, λ_s)`` uses the
      regime-aware inverse :math:`r_t = (s_t − λ_t s_{t-1}) / (1 − λ_t)`.
    * Conjugate NIG posterior on (β, α, σ²) given r.
    * Marginal log-likelihood includes the regime-aware Jacobian
      :math:`-n_n \log(1-λ_n) - n_s \log(1-λ_s)`, where ``n_n`` and ``n_s``
      count regime occurrences.

Stage 0 carry / MTM split
-------------------------
Per the spec, only the MTM component of credit returns is desmoothed; the
carry component bypasses Stage 1 entirely. This module operates on the
*MTM series only* — the caller (typically the pipeline runner) is
responsible for calling :func:`preprocessing.credit_decomposition.decompose_credit_return`
first and reattaching carry after disaggregation.

Identifiability diagnostics
---------------------------
* Each regime's ``λ`` 90 % credible interval width > 0.4 → warn.
* Either regime has fewer than ``min_regime_obs`` observations (default 8)
  → warn that the regime is underpowered.
* Posterior probability mass that ``λ_stress > λ_normal`` (which
  contradicts the model premise) → warn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma
from typing import Any

import numpy as np
import pandas as pd
from scipy import integrate

from ..core.config import BetaDist, InverseGammaPrior, NormalPrior
from ..core.protocols import DesmoothedResult


@dataclass
class ThresholdAR1Smoother:
    """Bayesian threshold AR(1) desmoother with exogenous regime indicator.

    Parameters
    ----------
    lambda_grid_size
        Resolution along each axis of the 2-D ``(λ_normal, λ_stress)`` grid.
        Default 30 → 900 candidates total.
    lambda_lower, lambda_upper
        Bounds on each ``λ``. Defaults ``[0.01, 0.95]`` keep the inverse
        ``1/(1-λ)`` numerically well-conditioned.
    ci_low, ci_high
        Credible-interval probabilities for the posterior summary.
    min_regime_obs
        Minimum observations required per regime before posterior moments
        are reported as informative; below this, a diagnostic warning fires.
    """

    lambda_grid_size: int = 30
    lambda_lower: float = 0.01
    lambda_upper: float = 0.95
    ci_low: float = 0.05
    ci_high: float = 0.95
    min_regime_obs: int = 8
    lambda_grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.lambda_grid_size < 5:
            raise ValueError(f"lambda_grid_size must be >= 5, got {self.lambda_grid_size}")
        if not (0.0 < self.lambda_lower < self.lambda_upper < 1.0):
            raise ValueError(
                f"require 0 < lambda_lower ({self.lambda_lower}) "
                f"< lambda_upper ({self.lambda_upper}) < 1."
            )
        if not (0.0 < self.ci_low < self.ci_high < 1.0):
            raise ValueError(
                f"require 0 < ci_low ({self.ci_low}) < ci_high ({self.ci_high}) < 1."
            )
        self.lambda_grid = np.linspace(
            self.lambda_lower, self.lambda_upper, self.lambda_grid_size
        )

    # ── SmoothingModel interface ────────────────────────────────────

    def fit(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
        priors: dict[str, Any],
    ) -> DesmoothedResult:
        """Run the Bayesian threshold-AR(1) fit on a regime-tagged series.

        Notes
        -----
        ``priors`` must contain a ``'regime_indicator'`` key whose value is
        a :class:`pandas.Series` of binary regime labels (0 = normal,
        1 = stress) aligned to ``observed_returns.index``. ``priors`` must
        also contain ``lambda_normal``, ``lambda_stress``, ``beta``,
        ``alpha``, ``sigma_eps``.
        """
        s_obs, X, regime, factor_names = self._validate_and_prepare(
            observed_returns, factor_returns, priors
        )
        priors_validated = self._validate_priors(priors, factor_names)
        mu_0, V_0_inv, a_0, b_0 = self._build_nig_prior(priors_validated, factor_names)

        d = X.shape[1]
        n = X.shape[0]
        a_n = a_0 + n / 2.0
        sign_V0, logdet_V0_inv = np.linalg.slogdet(V_0_inv)
        if sign_V0 <= 0.0:
            raise RuntimeError("prior precision V_0_inv is not positive definite.")
        log_det_V0 = -logdet_V0_inv

        # Regime counts on the lagged-pair window (t = 1..T-1)
        regime_lagged = regime[1:]
        n_normal = int((regime_lagged == 0).sum())
        n_stress = int((regime_lagged == 1).sum())

        s_t = s_obs[1:]
        s_lag = s_obs[:-1]

        L = self.lambda_grid_size
        log_marg = np.zeros((L, L))
        post_w_means = np.zeros((L, L, d))
        post_w_diag_var = np.zeros((L, L, d))
        post_b = np.zeros((L, L))

        for i, lam_n in enumerate(self.lambda_grid):
            for j, lam_s in enumerate(self.lambda_grid):
                lam_t = np.where(regime_lagged == 1, lam_s, lam_n)
                y_lam = (s_t - lam_t * s_lag) / (1.0 - lam_t)
                V_n_inv = V_0_inv + X.T @ X
                sign_Vn, logdet_Vn_inv = np.linalg.slogdet(V_n_inv)
                if sign_Vn <= 0.0:
                    raise RuntimeError(
                        f"posterior precision not PD at (λ_n,λ_s)=({lam_n:.3f},{lam_s:.3f})."
                    )
                log_det_Vn = -logdet_Vn_inv
                mu_n = np.linalg.solve(V_n_inv, V_0_inv @ mu_0 + X.T @ y_lam)
                sse_term = (
                    float(y_lam @ y_lam)
                    + float(mu_0 @ V_0_inv @ mu_0)
                    - float(mu_n @ V_n_inv @ mu_n)
                )
                b_n = b_0 + 0.5 * sse_term
                if b_n <= 0.0:
                    b_n = 1e-300
                # Regime-aware Jacobian: ∏_t 1/(1-λ_t) = (1-λ_n)^{-n_n} (1-λ_s)^{-n_s}
                log_marg[i, j] = (
                    -0.5 * n * np.log(2.0 * np.pi)
                    + 0.5 * (log_det_Vn - log_det_V0)
                    + a_0 * np.log(b_0)
                    - a_n * np.log(b_n)
                    + lgamma(a_n)
                    - lgamma(a_0)
                    - n_normal * np.log(1.0 - lam_n)
                    - n_stress * np.log(1.0 - lam_s)
                )
                post_w_means[i, j] = mu_n
                V_n = np.linalg.inv(V_n_inv)
                scale = b_n / (a_n - 1.0)
                post_w_diag_var[i, j] = np.diag(V_n) * scale
                post_b[i, j] = b_n

        log_prior_n = priors_validated["lambda_normal"].logpdf(self.lambda_grid)
        log_prior_s = priors_validated["lambda_stress"].logpdf(self.lambda_grid)
        log_post = log_marg + log_prior_n[:, None] + log_prior_s[None, :]
        log_post -= log_post.max()
        post_unnorm = np.exp(log_post)
        # Normalise via 2-D trapezoid
        norm = float(
            integrate.trapezoid(
                integrate.trapezoid(post_unnorm, self.lambda_grid, axis=1),
                self.lambda_grid,
            )
        )
        if not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("threshold-AR(1) posterior failed to normalise.")
        post = post_unnorm / norm

        # Marginals
        marg_n = integrate.trapezoid(post, self.lambda_grid, axis=1)  # P(λ_n | data)
        marg_s = integrate.trapezoid(post, self.lambda_grid, axis=0)  # P(λ_s | data)

        lam_n_mean, lam_n_std, lam_n_lo, lam_n_hi = _summarise_marginal(
            self.lambda_grid, marg_n, self.ci_low, self.ci_high
        )
        lam_s_mean, lam_s_std, lam_s_lo, lam_s_hi = _summarise_marginal(
            self.lambda_grid, marg_s, self.ci_low, self.ci_high
        )

        # Posterior mass that λ_stress > λ_normal (contradicts model premise)
        # via 2-D integration of post over the upper-triangular region.
        ln_grid, ls_grid = np.meshgrid(self.lambda_grid, self.lambda_grid, indexing="ij")
        contradict_mask = ls_grid > ln_grid
        prob_contradict = float(
            integrate.trapezoid(
                integrate.trapezoid(post * contradict_mask, self.lambda_grid, axis=1),
                self.lambda_grid,
            )
        )

        # w & σ² posterior moments via 2-D integration
        w_mean = _trapz2d(post[..., None] * post_w_means, self.lambda_grid)
        E_var = _trapz2d(post[..., None] * post_w_diag_var, self.lambda_grid)
        E_w_sq = _trapz2d(post[..., None] * post_w_means**2, self.lambda_grid)
        w_var = E_var + E_w_sq - w_mean**2
        w_std = np.sqrt(np.maximum(w_var, 0.0))
        beta_mean = w_mean[:-1]
        beta_std = w_std[:-1]
        alpha_mean = float(w_mean[-1])
        alpha_std = float(w_std[-1])
        sigma2_post_mean = _trapz2d(post * post_b / (a_n - 1.0), self.lambda_grid)
        sigma_eps_mean = float(np.sqrt(max(float(sigma2_post_mean), 0.0)))

        # Native-frequency desmoothing using posterior-mean (λ_n, λ_s)
        r_post = self.desmooth(
            s_obs,
            np.array([lam_n_mean, lam_s_mean]),
            regime_indicator=regime,
        )
        true_returns = pd.Series(
            r_post, index=observed_returns.index, name=observed_returns.name
        )

        diagnostics = self._diagnostics(
            lam_n_lo=lam_n_lo,
            lam_n_hi=lam_n_hi,
            lam_s_lo=lam_s_lo,
            lam_s_hi=lam_s_hi,
            n_normal=n_normal,
            n_stress=n_stress,
            prob_contradict=prob_contradict,
            n_obs=len(observed_returns),
        )

        smoothing_params = {
            "lambda_normal": lam_n_mean,
            "lambda_stress": lam_s_mean,
            "beta": dict(zip(factor_names, [float(b) for b in beta_mean])),
            "alpha": alpha_mean,
            "sigma_eps": sigma_eps_mean,
        }
        posterior_summary = {
            "lambda_normal": {
                "mean": lam_n_mean,
                "std": lam_n_std,
                f"ci_{int(self.ci_low * 100):02d}": lam_n_lo,
                f"ci_{int(self.ci_high * 100):02d}": lam_n_hi,
            },
            "lambda_stress": {
                "mean": lam_s_mean,
                "std": lam_s_std,
                f"ci_{int(self.ci_low * 100):02d}": lam_s_lo,
                f"ci_{int(self.ci_high * 100):02d}": lam_s_hi,
            },
            "beta": {
                name: {"mean": float(beta_mean[k]), "std": float(beta_std[k])}
                for k, name in enumerate(factor_names)
            },
            "alpha": {"mean": alpha_mean, "std": alpha_std},
            "sigma_eps": {"mean": sigma_eps_mean},
            "joint_density": post.copy(),
            "lambda_grid": self.lambda_grid.copy(),
            "marginal_normal": marg_n.copy(),
            "marginal_stress": marg_s.copy(),
        }
        return DesmoothedResult(
            true_returns=true_returns,
            smoothing_params=smoothing_params,
            posterior_summary=posterior_summary,
            diagnostics=diagnostics,
        )

    def log_likelihood(
        self,
        smoothing_params: np.ndarray,
        observed_returns: np.ndarray,
        factor_returns: np.ndarray,
    ) -> float:
        raise NotImplementedError(
            "log_likelihood for threshold AR(1) requires the regime indicator; "
            "use fit() with priors['regime_indicator'] supplied."
        )

    def log_prior(
        self,
        smoothing_params: np.ndarray,
        prior_config: dict[str, Any],
    ) -> float:
        lam = np.asarray(smoothing_params, dtype=float).reshape(-1)
        if lam.size != 2:
            return -np.inf
        lam_n, lam_s = float(lam[0]), float(lam[1])
        if not (0.0 <= lam_n <= 1.0 and 0.0 <= lam_s <= 1.0):
            return -np.inf
        prior_n = prior_config.get("lambda_normal")
        prior_s = prior_config.get("lambda_stress")
        if not (isinstance(prior_n, BetaDist) and isinstance(prior_s, BetaDist)):
            raise ValueError(
                "prior_config requires lambda_normal and lambda_stress as BetaDist."
            )
        return float(prior_n.logpdf(lam_n) + prior_s.logpdf(lam_s))

    def desmooth(
        self,
        observed_returns: np.ndarray,
        smoothing_params: np.ndarray,
        *,
        regime_indicator: np.ndarray | pd.Series,
    ) -> np.ndarray:
        r"""Apply :math:`r_t = (s_t - λ_t s_{t-1}) / (1 - λ_t)` with regime-dependent λ."""
        lam = np.asarray(smoothing_params, dtype=float).reshape(-1)
        if lam.size != 2:
            raise ValueError(
                f"smoothing_params must be (λ_normal, λ_stress) of length 2, got {lam.size}"
            )
        lam_n, lam_s = float(lam[0]), float(lam[1])
        if not (0.0 <= lam_n < 1.0 and 0.0 <= lam_s < 1.0):
            raise ValueError(f"λ values must lie in [0, 1), got ({lam_n}, {lam_s})")
        s = np.asarray(observed_returns, dtype=float)
        if isinstance(regime_indicator, pd.Series):
            regime_indicator = regime_indicator.to_numpy()
        regime = np.asarray(regime_indicator, dtype=int)
        if regime.shape != s.shape:
            raise ValueError(
                f"regime_indicator shape {regime.shape} ≠ observed shape {s.shape}"
            )
        r = np.empty_like(s)
        r[0] = s[0]
        lam_t = np.where(regime[1:] == 1, lam_s, lam_n)
        r[1:] = (s[1:] - lam_t * s[:-1]) / (1.0 - lam_t)
        return r

    # ── Internals ───────────────────────────────────────────────────

    def _validate_and_prepare(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
        priors: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        if not isinstance(observed_returns, pd.Series):
            raise TypeError(
                f"observed_returns must be a Series, got {type(observed_returns).__name__}"
            )
        if not isinstance(factor_returns, pd.DataFrame) or factor_returns.empty:
            raise TypeError(
                "factor_returns must be a non-empty DataFrame for the threshold AR(1) fit."
            )
        if observed_returns.isna().any():
            raise ValueError("observed_returns contains NaN.")
        regime = priors.get("regime_indicator")
        if not isinstance(regime, pd.Series):
            raise ValueError(
                "priors['regime_indicator'] must be a pandas Series of binary labels."
            )
        regime = regime.reindex(observed_returns.index)
        if regime.isna().any():
            raise ValueError("regime_indicator missing values after alignment.")
        regime_arr = regime.to_numpy(dtype=int)
        if not set(np.unique(regime_arr)).issubset({0, 1}):
            raise ValueError(
                f"regime_indicator must be binary (0 or 1); got values {set(np.unique(regime_arr))}"
            )

        factor_returns = factor_returns.reindex(observed_returns.index)
        if factor_returns.isna().any().any():
            raise ValueError("factor_returns has NaN after alignment.")
        n_obs = len(observed_returns)
        if n_obs < 12:
            raise ValueError(f"need >= 12 observations, got {n_obs}.")
        s = observed_returns.to_numpy(dtype=float)
        F = factor_returns.to_numpy(dtype=float)
        F_t = F[1:]
        X = np.hstack([F_t, np.ones((F_t.shape[0], 1))])
        return s, X, regime_arr, list(factor_returns.columns)

    def _validate_priors(
        self,
        priors: dict[str, Any],
        factor_names: list[str],
    ) -> dict[str, Any]:
        for required in ("lambda_normal", "lambda_stress", "beta", "alpha", "sigma_eps"):
            if required not in priors:
                raise ValueError(f"priors missing required key {required!r}")
        if not isinstance(priors["lambda_normal"], BetaDist):
            raise TypeError("priors['lambda_normal'] must be a BetaDist.")
        if not isinstance(priors["lambda_stress"], BetaDist):
            raise TypeError("priors['lambda_stress'] must be a BetaDist.")
        if not isinstance(priors["alpha"], NormalPrior):
            raise TypeError("priors['alpha'] must be a NormalPrior.")
        if not isinstance(priors["sigma_eps"], InverseGammaPrior):
            raise TypeError("priors['sigma_eps'] must be an InverseGammaPrior.")
        if not isinstance(priors["beta"], dict):
            raise TypeError("priors['beta'] must be a mapping.")
        missing = [f for f in factor_names if f not in priors["beta"]]
        if missing:
            raise ValueError(f"priors['beta'] missing entries for {missing!r}")
        for k in factor_names:
            if not isinstance(priors["beta"][k], NormalPrior):
                raise TypeError(f"priors['beta'][{k!r}] must be a NormalPrior.")
        if priors["sigma_eps"].alpha <= 1.0:
            raise ValueError(
                f"priors['sigma_eps'].alpha must be > 1; got {priors['sigma_eps'].alpha}"
            )
        return priors

    def _build_nig_prior(
        self,
        priors: dict[str, Any],
        factor_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        beta_priors = priors["beta"]
        alpha_prior = priors["alpha"]
        ig = priors["sigma_eps"]
        a_0 = float(ig.alpha)
        b_0 = float(ig.beta)
        E_sigma2 = b_0 / (a_0 - 1.0)
        sigma_marg = np.array(
            [beta_priors[k].std for k in factor_names] + [alpha_prior.std], dtype=float
        )
        mu_0 = np.array(
            [beta_priors[k].mean for k in factor_names] + [alpha_prior.mean], dtype=float
        )
        V_0_inv = np.diag(E_sigma2 / (sigma_marg**2))
        return mu_0, V_0_inv, a_0, b_0

    def _diagnostics(
        self,
        *,
        lam_n_lo: float,
        lam_n_hi: float,
        lam_s_lo: float,
        lam_s_hi: float,
        n_normal: int,
        n_stress: int,
        prob_contradict: float,
        n_obs: int,
    ) -> dict[str, Any]:
        warns: list[str] = []
        flags: dict[str, bool] = {}
        flags["normal_ci_too_wide"] = (lam_n_hi - lam_n_lo) > 0.4
        flags["stress_ci_too_wide"] = (lam_s_hi - lam_s_lo) > 0.4
        flags["normal_underpowered"] = n_normal < self.min_regime_obs
        flags["stress_underpowered"] = n_stress < self.min_regime_obs
        flags["regime_contradiction"] = prob_contradict > 0.2

        if flags["normal_ci_too_wide"]:
            warns.append(
                f"normal-regime λ has 90 % CI width "
                f"{lam_n_hi - lam_n_lo:.3f} > 0.4 — poorly identified."
            )
        if flags["stress_ci_too_wide"]:
            warns.append(
                f"stress-regime λ has 90 % CI width "
                f"{lam_s_hi - lam_s_lo:.3f} > 0.4 — poorly identified."
            )
        if flags["normal_underpowered"]:
            warns.append(
                f"normal regime has only {n_normal} observations — "
                "below min_regime_obs threshold; posterior is prior-driven."
            )
        if flags["stress_underpowered"]:
            warns.append(
                f"stress regime has only {n_stress} observations — "
                "below min_regime_obs threshold; posterior is prior-driven."
            )
        if flags["regime_contradiction"]:
            warns.append(
                f"posterior puts {prob_contradict:.1%} mass on λ_stress > λ_normal, "
                "contradicting the model premise (stress should mark faster). "
                "Consider re-checking the regime threshold."
            )

        return {
            "method": "threshold_ar1",
            "n_observations": int(n_obs),
            "n_normal": n_normal,
            "n_stress": n_stress,
            "prob_lambda_stress_gt_normal": prob_contradict,
            "lambda_normal_ci": (lam_n_lo, lam_n_hi),
            "lambda_stress_ci": (lam_s_lo, lam_s_hi),
            "flags": flags,
            "warnings": warns,
        }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _summarise_marginal(
    grid: np.ndarray,
    density: np.ndarray,
    ci_low: float,
    ci_high: float,
) -> tuple[float, float, float, float]:
    norm = float(integrate.trapezoid(density, grid))
    if norm <= 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pdf = density / norm
    mean = float(integrate.trapezoid(grid * pdf, grid))
    var = float(integrate.trapezoid((grid - mean) ** 2 * pdf, grid))
    cdf = integrate.cumulative_trapezoid(pdf, grid, initial=0.0)
    cdf /= cdf[-1]
    lo = float(np.interp(ci_low, cdf, grid))
    hi = float(np.interp(ci_high, cdf, grid))
    return mean, float(np.sqrt(max(var, 0.0))), lo, hi


def _trapz2d(values: np.ndarray, grid: np.ndarray) -> np.ndarray | float:
    """Integrate ``values`` over a 2-D regular grid via the trapezoidal rule.

    ``values`` must have shape ``(L, L, *trailing)``; the returned shape is
    ``trailing``.
    """
    inner = integrate.trapezoid(values, grid, axis=1)
    return integrate.trapezoid(inner, grid, axis=0)


__all__ = ["ThresholdAR1Smoother"]
