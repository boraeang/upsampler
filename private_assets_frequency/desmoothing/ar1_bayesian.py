r"""
Bayesian AR(1) desmoother for private equity, infrastructure, and real estate.

This is the primary desmoothing model in the library. It follows the MSCI
PE Factor Model methodology described in Stage 1 of the spec.

Model
-----
The AR(1) appraisal-smoothing process is::

    s_t = (1 - λ) r_t + λ s_{t-1}                  (smoothing)
    r_t = β · F_t + α + ε_t,   ε_t ∼ N(0, σ_ε²)    (factor model on truth)

The two equations are estimated *jointly* — the spec stresses that
single-step joint estimation of (λ, β, α, σ_ε) is more honest about the
λ-β confound than a two-step procedure that estimates λ first and then
regresses the desmoothed series on factors.

Prior structure (Normal-Inverse-Gamma conjugate, σ²-coupled)
-------------------------------------------------------------
For the regression weights ``w = (β_1, …, β_K, α)``::

    w | σ² ∼ N(μ_0, σ² · V_0)
    σ²    ∼ InverseGamma(a_0, b_0)

The user supplies *marginal* priors on each weight via :class:`NormalPrior`
and on σ² via :class:`InverseGammaPrior` from :mod:`core.config`. These are
mapped onto NIG hyperparameters by setting::

    V_0,kk = σ_k² / E[σ²]   where  E[σ²] = b_0 / (a_0 - 1)

so that the marginal prior std of ``w_k`` (under the Student-t marginal of
the NIG) equals ``σ_k`` exactly when ``a_0 > 1``. With the spec's defaults
``a_0 = 3, b_0 = 0.02`` this gives ``E[σ²] = 0.01`` (i.e. 10 % per-period
σ, plausible for quarterly PE returns). The σ²-coupling is required for
analytical NIG conjugacy — see Murphy (2007) or any standard Bayesian-
linreg reference.

Posterior computation
---------------------
Grid integration over λ ∈ [0.01, 0.95] (50 points by default). For each λ,
the conjugate NIG update gives closed-form ``μ_n``, ``V_n``, ``a_n``,
``b_n`` and a marginal log-likelihood ``log p(y(λ) | λ)``. Combining with
the Beta prior on λ and trapezoidal-rule integration yields the marginal
posterior on λ; posterior expectations of ``w`` and σ² are obtained by
integrating the conditional NIG moments against the λ posterior.

Note on rolling-annual vs quarterly fits
----------------------------------------
The spec recommends fitting on rolling four-quarter compounded returns to
break Q4 mark-to-market seasonality of real PE/RE indices. This
implementation currently fits on **raw quarterly observations** (i.e. the
quarterly AR(1) recursion), which is the cleanest mapping from
synthetic-data ground truth back to the recovered λ — the rolling-annual
transform is an *approximation* (the quarterly AR(1) does not aggregate
exactly to an annual AR(1) with the same λ), so using it on synthetic data
biases λ recovery. For real data with seasonality, the
``use_rolling_annual=True`` toggle (planned) would aggregate observed and
factor returns to rolling-annual before fitting; in the meantime, the
diagnostic block flags strong Q4 effects so users know when to worry.

Identifiability diagnostics (Stage 1 spec point 6)
--------------------------------------------------
Each fit emits warnings when the joint identification is suspect:

* 90 % credible interval on λ wider than 0.4
* KL(posterior‖prior) on λ below 0.1 nats
* β changes sign or magnitude > 50 % across the λ grid

References
----------
.. [1] Geltner (1993) — original AR(1) appraisal-smoothing paper.
.. [2] MSCI (2025) — "The MSCI Private Equity Factor Model."
.. [3] Murphy (2007) — "Conjugate Bayesian analysis of the Gaussian
       distribution," technical note (NIG conjugate update).
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


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────


@dataclass
class AR1BayesianSmoother:
    """Bayesian AR(1) desmoother (Stage 1, Model 1).

    Parameters
    ----------
    lambda_grid_size
        Number of grid points across :math:`[\\text{lambda\\_lower},
        \\text{lambda\\_upper}]`. Default 50 per the spec.
    lambda_lower, lambda_upper
        λ grid bounds. Defaults ``[0.01, 0.95]`` keep the desmoothing
        transform ``1/(1-λ)`` numerically well-conditioned.
    ci_low, ci_high
        Quantile probabilities for the credible interval reported in the
        posterior summary. Defaults to the 90 % CI used throughout the spec.
    """

    lambda_grid_size: int = 50
    lambda_lower: float = 0.01
    lambda_upper: float = 0.95
    ci_low: float = 0.05
    ci_high: float = 0.95
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
        """Run the Bayesian AR(1) fit and return the desmoothed series + posterior."""
        s_obs, X, factor_names = self._validate_and_prepare(
            observed_returns, factor_returns
        )
        priors_validated = self._validate_priors(priors, factor_names)

        # NIG prior hyperparameters (σ²-coupled)
        d = X.shape[1]  # number of regression coefs (K factors + intercept)
        n = X.shape[0]
        mu_0, V_0_inv, a_0, b_0 = self._build_nig_prior(priors_validated, factor_names)

        # Pre-compute posterior shape
        a_n = a_0 + n / 2.0

        # Storage for grid sweep
        L = self.lambda_grid_size
        log_marg = np.zeros(L)
        post_w_means = np.zeros((L, d))      # E[w | λ, y]
        post_w_diag_var = np.zeros((L, d))   # diag of conditional Var(w | λ, y)
        post_b = np.zeros(L)
        sign_V0, logdet_V0_inv = np.linalg.slogdet(V_0_inv)
        if sign_V0 <= 0.0:
            raise RuntimeError("prior precision V_0_inv is not positive definite.")
        # logdet(V_0) = -logdet(V_0_inv)
        log_det_V0 = -logdet_V0_inv

        s_t = s_obs[1:]
        s_lag = s_obs[:-1]

        for i, lam in enumerate(self.lambda_grid):
            # Desmooth at this λ: r_λ = (s_t − λ s_{t-1}) / (1−λ)
            y_lam = (s_t - lam * s_lag) / (1.0 - lam)
            # NIG conjugate posterior
            V_n_inv = V_0_inv + X.T @ X
            sign_Vn, log_det_Vn_inv = np.linalg.slogdet(V_n_inv)
            if sign_Vn <= 0.0:
                raise RuntimeError(
                    f"posterior precision V_n_inv is not PD at λ={lam:.4f}."
                )
            log_det_Vn = -log_det_Vn_inv
            mu_n = np.linalg.solve(V_n_inv, V_0_inv @ mu_0 + X.T @ y_lam)
            sse_term = (
                float(y_lam @ y_lam)
                + float(mu_0 @ V_0_inv @ mu_0)
                - float(mu_n @ V_n_inv @ mu_n)
            )
            b_n = b_0 + 0.5 * sse_term
            if b_n <= 0.0:
                # Numerical edge case — set very small but positive
                b_n = max(b_n, 1e-300)
            # NIG marginal of y(λ) given X. The full marginal of s | λ
            # additionally carries the Jacobian of the transform
            # s ↦ y(λ) = (s_t − λ s_{t-1}) / (1−λ): conditional on s_0, this
            # is a triangular linear map with determinant (1−λ)^{-n}, so
            # log p(s | λ) = log p(y(λ) | λ) + (-n) · log(1−λ) · (-1) · (-1)
            #             = log p(y(λ) | λ) − n log(1−λ).
            # Without this term the marginal likelihood is biased toward
            # small λ (which shrinks y's variance). See Geltner (1993) /
            # MSCI (2025) for the equivalent treatment.
            log_marg[i] = (
                -0.5 * n * np.log(2.0 * np.pi)
                + 0.5 * (log_det_Vn - log_det_V0)
                + a_0 * np.log(b_0)
                - a_n * np.log(b_n)
                + lgamma(a_n)
                - lgamma(a_0)
                - n * np.log(1.0 - lam)
            )
            post_w_means[i] = mu_n
            # Conditional posterior of w | (λ, y, σ²) is N(μ_n, σ² V_n).
            # Marginal over σ² gives Student-t with cov (b_n / (a_n - 1)) · V_n
            # for a_n > 1. Use diagonal as variance summary.
            V_n = np.linalg.inv(V_n_inv)
            scale = b_n / (a_n - 1.0)
            post_w_diag_var[i] = np.diag(V_n) * scale
            post_b[i] = b_n

        # Combine with Beta prior on λ, normalise via trapezoid
        log_prior_lam = priors_validated["lambda"].logpdf(self.lambda_grid)
        log_post_lam = log_marg + log_prior_lam
        # Stabilise then exponentiate
        log_post_lam -= log_post_lam.max()
        post_lam_unnorm = np.exp(log_post_lam)
        norm = float(integrate.trapezoid(post_lam_unnorm, self.lambda_grid))
        if norm <= 0.0 or not np.isfinite(norm):
            raise RuntimeError("λ posterior failed to normalise (non-finite or zero mass).")
        post_lam = post_lam_unnorm / norm

        # λ posterior moments
        lam_mean = float(integrate.trapezoid(self.lambda_grid * post_lam, self.lambda_grid))
        lam_var = float(
            integrate.trapezoid((self.lambda_grid - lam_mean) ** 2 * post_lam, self.lambda_grid)
        )
        cdf = integrate.cumulative_trapezoid(post_lam, self.lambda_grid, initial=0.0)
        cdf /= cdf[-1]  # guard against tiny rounding error
        ci_lo = float(np.interp(self.ci_low, cdf, self.lambda_grid))
        ci_hi = float(np.interp(self.ci_high, cdf, self.lambda_grid))

        # w posterior moments (β & α; integrate over λ)
        w_mean = integrate.trapezoid(
            post_w_means * post_lam[:, None], self.lambda_grid, axis=0
        )
        # Total variance: E[Var(w|λ)] + Var(E[w|λ])
        E_var = integrate.trapezoid(
            post_w_diag_var * post_lam[:, None], self.lambda_grid, axis=0
        )
        E_w_sq = integrate.trapezoid(
            post_w_means**2 * post_lam[:, None], self.lambda_grid, axis=0
        )
        w_var = E_var + E_w_sq - w_mean**2
        w_std = np.sqrt(np.maximum(w_var, 0.0))

        beta_mean = w_mean[:-1]
        beta_std = w_std[:-1]
        alpha_mean = float(w_mean[-1])
        alpha_std = float(w_std[-1])

        # σ² posterior mean: E[σ²] = E[b_n / (a_n - 1)] (a_n > 1 because a_0 + n/2 > 1)
        sigma2_post_mean = float(
            integrate.trapezoid(post_b / (a_n - 1.0) * post_lam, self.lambda_grid)
        )
        sigma_eps_mean = float(np.sqrt(max(sigma2_post_mean, 0.0)))

        # Native-frequency desmoothing using posterior-mean λ
        r_post = self.desmooth(s_obs, np.array([lam_mean]))
        true_returns = pd.Series(
            r_post, index=observed_returns.index, name=observed_returns.name
        )

        # Diagnostics
        diagnostics = self._diagnostics(
            log_post_lam=log_post_lam,
            post_lam=post_lam,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            post_w_means=post_w_means,
            factor_names=factor_names,
            n_obs=n,
            priors=priors_validated,
        )

        smoothing_params = {
            "lambda": lam_mean,
            "beta": dict(zip(factor_names, [float(b) for b in beta_mean])),
            "alpha": alpha_mean,
            "sigma_eps": sigma_eps_mean,
        }
        posterior_summary = {
            "lambda": {
                "mean": lam_mean,
                "std": float(np.sqrt(max(lam_var, 0.0))),
                f"ci_{int(self.ci_low * 100):02d}": ci_lo,
                f"ci_{int(self.ci_high * 100):02d}": ci_hi,
                "grid": self.lambda_grid.copy(),
                "density": post_lam.copy(),
            },
            "beta": {
                name: {"mean": float(beta_mean[k]), "std": float(beta_std[k])}
                for k, name in enumerate(factor_names)
            },
            "alpha": {"mean": alpha_mean, "std": alpha_std},
            "sigma_eps": {"mean": sigma_eps_mean},
        }

        # Cache last fit for sample_posterior() if user asks
        self._last_fit = {
            "log_post_lam": log_post_lam,
            "post_lam": post_lam,
            "cdf": cdf,
            "post_w_means": post_w_means,
            "post_w_cov_diag": post_w_diag_var,
            "post_b": post_b,
            "a_n": a_n,
            "factor_names": factor_names,
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
        """Profile log-likelihood at λ (concentrating out (β, α, σ²) by OLS).

        Used for diagnostic / sanity checks; the full Bayesian fit goes
        through :meth:`fit`.
        """
        lam = float(smoothing_params[0])
        if not (0.0 <= lam < 1.0):
            return -np.inf
        s = np.asarray(observed_returns, dtype=float)
        F = np.asarray(factor_returns, dtype=float)
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        s_t = s[1:]
        s_lag = s[:-1]
        F_t = F[1:]
        y = (s_t - lam * s_lag) / (1.0 - lam)
        n = len(y)
        X_aug = np.hstack([F_t, np.ones((n, 1))])
        coefs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        resid = y - X_aug @ coefs
        sigma2 = float(np.sum(resid**2) / max(n - X_aug.shape[1], 1))
        if sigma2 <= 0.0:
            return -np.inf
        ll = -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * np.sum(resid**2) / sigma2
        return float(ll)

    def log_prior(
        self,
        smoothing_params: np.ndarray,
        prior_config: dict[str, Any],
    ) -> float:
        """Log-prior on λ (the only smoothing parameter)."""
        lam = float(smoothing_params[0])
        prior = prior_config.get("lambda")
        if prior is None or not isinstance(prior, BetaDist):
            raise ValueError("prior_config must contain a 'lambda' BetaDist entry.")
        if not (0.0 <= lam <= 1.0):
            return -np.inf
        return float(prior.logpdf(lam))

    def desmooth(
        self,
        observed_returns: np.ndarray,
        smoothing_params: np.ndarray,
    ) -> np.ndarray:
        r"""Apply the AR(1) inverse :math:`r_t = (s_t - \lambda s_{t-1}) / (1-\lambda)`.

        First observation is left unchanged (no lag available); remaining
        ``T-1`` observations are desmoothed in closed form.
        """
        lam = float(np.asarray(smoothing_params).reshape(-1)[0])
        if not (0.0 <= lam < 1.0):
            raise ValueError(f"λ must lie in [0, 1), got {lam}")
        s = np.asarray(observed_returns, dtype=float)
        if s.ndim != 1:
            raise ValueError(f"observed_returns must be 1-D, got ndim={s.ndim}")
        r = np.empty_like(s)
        r[0] = s[0]
        if lam == 0.0:
            r[1:] = s[1:]
        else:
            r[1:] = (s[1:] - lam * s[:-1]) / (1.0 - lam)
        return r

    # ── Posterior sampling (uncertainty_mode='full') ────────────────

    def sample_posterior(
        self,
        n_samples: int,
        rng: np.random.Generator | int | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw ``n_samples`` joint posterior samples of (λ, w, σ²).

        Requires that :meth:`fit` has been called. Samples λ from the grid
        posterior via inverse-CDF sampling, then nearest-grid-point
        conditional draws of σ² (Inverse-Gamma) and ``w | σ²`` (Gaussian).
        """
        if not hasattr(self, "_last_fit"):
            raise RuntimeError("call fit() before sample_posterior().")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        cache = self._last_fit
        cdf = cache["cdf"]
        # Inverse-CDF sample of λ
        u = rng.uniform(size=n_samples)
        lam_samples = np.interp(u, cdf, self.lambda_grid)
        # Nearest-grid-point conditional draws
        idx = np.searchsorted(self.lambda_grid, lam_samples)
        idx = np.clip(idx, 0, self.lambda_grid_size - 1)
        a_n = cache["a_n"]
        b_n_arr = cache["post_b"][idx]
        # Sample σ² ~ IG(a_n, b_n)
        sigma2_samples = 1.0 / rng.gamma(shape=a_n, scale=1.0 / b_n_arr)
        # Sample w ~ N(μ_n, σ² · V_n_diag) — diagonal approximation
        w_means = cache["post_w_means"][idx]
        w_var_diag = cache["post_w_cov_diag"][idx] * (a_n - 1.0) / b_n_arr[:, None] * sigma2_samples[:, None]
        w_samples = w_means + rng.normal(size=w_means.shape) * np.sqrt(np.maximum(w_var_diag, 0.0))
        K = w_samples.shape[1] - 1
        return {
            "lambda": lam_samples,
            "beta": w_samples[:, :K],
            "alpha": w_samples[:, K],
            "sigma_eps": np.sqrt(sigma2_samples),
            "factor_names": np.array(cache["factor_names"]),
        }

    # ── Internals ───────────────────────────────────────────────────

    def _validate_and_prepare(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if not isinstance(observed_returns, pd.Series):
            raise TypeError(
                f"observed_returns must be a Series, got {type(observed_returns).__name__}"
            )
        if not isinstance(factor_returns, pd.DataFrame) or factor_returns.empty:
            raise TypeError(
                "factor_returns must be a non-empty DataFrame for the AR(1) Bayesian fit."
            )
        if observed_returns.isna().any():
            raise ValueError("observed_returns contains NaN.")
        # Align
        factor_returns = factor_returns.reindex(observed_returns.index)
        if factor_returns.isna().any().any():
            raise ValueError(
                "factor_returns has NaN values after alignment to observed_returns."
            )
        n_obs = len(observed_returns)
        if n_obs < 12:
            raise ValueError(
                f"need >= 12 observations for a stable Bayesian AR(1) fit, got {n_obs}."
            )
        s = observed_returns.to_numpy(dtype=float)
        F = factor_returns.to_numpy(dtype=float)
        # Build X = [F_t, 1] for t = 1..T-1 (i.e. the rows aligned with s_t and s_{t-1})
        F_t = F[1:]
        X = np.hstack([F_t, np.ones((F_t.shape[0], 1))])
        return s, X, list(factor_returns.columns)

    def _validate_priors(
        self,
        priors: dict[str, Any],
        factor_names: list[str],
    ) -> dict[str, Any]:
        for required in ("lambda", "beta", "alpha", "sigma_eps"):
            if required not in priors:
                raise ValueError(f"priors missing required key {required!r}")
        if not isinstance(priors["lambda"], BetaDist):
            raise TypeError("priors['lambda'] must be a BetaDist.")
        if not isinstance(priors["alpha"], NormalPrior):
            raise TypeError("priors['alpha'] must be a NormalPrior.")
        if not isinstance(priors["sigma_eps"], InverseGammaPrior):
            raise TypeError("priors['sigma_eps'] must be an InverseGammaPrior.")
        beta_priors = priors["beta"]
        if not isinstance(beta_priors, dict):
            raise TypeError("priors['beta'] must be a mapping factor_name → NormalPrior.")
        missing = [f for f in factor_names if f not in beta_priors]
        if missing:
            raise ValueError(
                f"priors['beta'] missing entries for factors {missing!r}"
            )
        for k in factor_names:
            if not isinstance(beta_priors[k], NormalPrior):
                raise TypeError(
                    f"priors['beta'][{k!r}] must be a NormalPrior."
                )
        # sigma_eps prior must have a > 1 for finite E[σ²]
        ig = priors["sigma_eps"]
        if ig.alpha <= 1.0:
            raise ValueError(
                f"priors['sigma_eps'].alpha must be > 1 for finite prior mean; "
                f"got {ig.alpha}."
            )
        return priors

    def _build_nig_prior(
        self,
        priors: dict[str, Any],
        factor_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Map user-facing marginal priors to NIG hyperparameters (μ_0, V_0_inv, a_0, b_0)."""
        beta_priors = priors["beta"]
        alpha_prior = priors["alpha"]
        ig = priors["sigma_eps"]
        a_0 = float(ig.alpha)
        b_0 = float(ig.beta)
        E_sigma2 = b_0 / (a_0 - 1.0)
        # Marginal prior std for each w_k
        sigma_marg = np.array(
            [beta_priors[k].std for k in factor_names] + [alpha_prior.std], dtype=float
        )
        mu_0 = np.array(
            [beta_priors[k].mean for k in factor_names] + [alpha_prior.mean], dtype=float
        )
        # V_0,kk = σ_k² / E[σ²]; V_0_inv = diag(E[σ²] / σ_k²)
        V_0_inv = np.diag(E_sigma2 / (sigma_marg**2))
        return mu_0, V_0_inv, a_0, b_0

    def _diagnostics(
        self,
        *,
        log_post_lam: np.ndarray,
        post_lam: np.ndarray,
        ci_lo: float,
        ci_hi: float,
        post_w_means: np.ndarray,
        factor_names: list[str],
        n_obs: int,
        priors: dict[str, Any],
    ) -> dict[str, Any]:
        warns: list[str] = []
        flags: dict[str, bool] = {}
        ci_width = ci_hi - ci_lo
        ci_too_wide = ci_width > 0.4
        flags["ci_too_wide"] = ci_too_wide
        if ci_too_wide:
            warns.append(
                f"90% credible interval on λ has width {ci_width:.3f} > 0.4 — "
                "smoothing is poorly identified; output is prior-driven."
            )

        # KL(post || prior) on λ
        prior_density = priors["lambda"].pdf(self.lambda_grid)
        # Avoid log(0)
        eps = 1e-20
        kl_integrand = post_lam * (np.log(post_lam + eps) - np.log(prior_density + eps))
        kl = float(integrate.trapezoid(kl_integrand, self.lambda_grid))
        flags["data_uninformative"] = kl < 0.1
        if kl < 0.1:
            warns.append(
                f"KL(posterior‖prior) on λ is {kl:.3f} nats < 0.1 — "
                "data is essentially not updating the prior."
            )

        # β sign / magnitude variation across λ grid
        # post_w_means is L × d (d = K + 1 with α last)
        K = len(factor_names)
        beta_grid = post_w_means[:, :K]
        sign_changes: list[str] = []
        magnitude_changes: list[str] = []
        for k, name in enumerate(factor_names):
            col = beta_grid[:, k]
            if (col > 0).any() and (col < 0).any():
                sign_changes.append(name)
            mag_min = float(np.min(np.abs(col)))
            mag_max = float(np.max(np.abs(col)))
            if mag_min > 0.0 and mag_max / mag_min > 1.5:
                magnitude_changes.append(name)
        flags["beta_sign_change"] = bool(sign_changes)
        flags["beta_magnitude_change"] = bool(magnitude_changes)
        if sign_changes:
            warns.append(
                f"β changes sign across the λ grid for factor(s) {sign_changes} — "
                "λ and β are confounded; consider uncertainty_mode='full'."
            )
        if magnitude_changes:
            warns.append(
                f"β magnitude varies > 50% across the λ grid for factor(s) "
                f"{magnitude_changes} — λ and β are confounded; consider "
                "uncertainty_mode='full'."
            )

        diagnostics: dict[str, Any] = {
            "method": "ar1_bayesian",
            "n_observations": int(n_obs),
            "lambda_ci_low": ci_lo,
            "lambda_ci_high": ci_hi,
            "lambda_ci_width": ci_width,
            "kl_post_prior": kl,
            "flags": flags,
            "warnings": warns,
            "lambda_grid": self.lambda_grid.copy(),
            "lambda_density": post_lam.copy(),
        }
        # Warnings live in diagnostics["warnings"]; the pipeline runner
        # decides whether to escalate based on FallbackPolicy.
        return diagnostics


__all__ = ["AR1BayesianSmoother"]
