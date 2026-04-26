r"""
Getmansky-Lo-Makarov MA(q) desmoother for hedge-fund returns (Stage 1, Model 2).

Model
-----
The reported hedge-fund index return is a weighted MA(q) of the underlying
true monthly return::

    s_t = θ_0 r_t + θ_1 r_{t-1} + ... + θ_q r_{t-q}     (smoothing)
    Σ_{j=0..q} θ_j = 1,   θ_j ≥ 0                       (Getmansky constraints)
    r_t = β · F_t + α + ε_t,  ε_t ∼ N(0, σ_ε²)         (factor model on truth)

The recovered series is :math:`r = \Theta^{-1} s` where Θ is a banded lower-
triangular Toeplitz matrix with θ on the first ``q+1`` diagonals. We solve
this system efficiently with :func:`scipy.signal.lfilter`.

Bayesian estimation
-------------------
For each candidate θ on the simplex grid, the conditional posterior of
``(β, α, σ²)`` is the standard Normal-Inverse-Gamma update applied to the
desmoothed series :math:`r_\theta = \Theta^{-1} s` regressed on the factor
matrix. The marginal log-likelihood of the *observed* series :math:`s` adds
the Jacobian factor :math:`-n \log θ_0` (since :math:`\det \Theta = θ_0^n`),
which is required to make the comparison across θ candidates honest.

Prior structure
~~~~~~~~~~~~~~~
* ``θ`` lives on the (q+1)-simplex with ordering :math:`θ_0 ≥ θ_1 ≥ … ≥ θ_q`
  and a hard floor :math:`θ_0 ≥ \text{theta\_0\_floor}` (default 0.2 per
  Stage 1 spec point 4).
* Density on the ordered region: Dirichlet(c_0, …, c_q) with default
  decreasing concentrations ``(5, 3, 2, 1)`` truncated to length ``q+1``.
  Combined with the ordering indicator, this captures the spec's
  "ordered Dirichlet / stick-breaking" prior on θ.
* ``β`` and ``α`` carry user-supplied :class:`NormalPrior` densities;
  ``σ_ε²`` carries an :class:`InverseGammaPrior`. These are mapped onto the
  σ²-coupled NIG hyperparameters exactly as in :class:`AR1BayesianSmoother`,
  so the marginal prior std on each weight equals the user's input.

Grid integration is feasible for ``q ≤ 3``. For ``q ≥ 4`` the spec
recommends Laplace approximation; this implementation raises
``NotImplementedError`` for ``q ≥ 4`` and defers Laplace to a follow-up.

Identifiability diagnostics
---------------------------
* ``cond(Θ) > cond_warning_threshold`` (default 100) → warn that the
  inversion is amplifying noise.
* Posterior-mean ``θ_0 < theta_0_warning_threshold`` (default 0.3) → warn
  that the contemporaneous weight is borderline; recommend reducing ``q``
  or applying the auto-fallback in :class:`FallbackPolicy.AUTO`.
* Standard NIG diagnostics (β, α, σ_ε posterior summaries).

References
----------
.. [1] Getmansky, Lo, Makarov (2004) — "An econometric model of serial
       correlation and illiquidity in hedge fund returns," *J. Financial
       Economics* 74: 529-609.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal
from scipy.special import gammaln

from ..core.config import InverseGammaPrior, NormalPrior
from ..core.protocols import DesmoothedResult


@dataclass
class MAGLMSmoother:
    """Bayesian Getmansky-Lo-Makarov MA(q) desmoother.

    Parameters
    ----------
    q
        Number of MA lags (so the weight vector has length ``q+1``).
        Supported range is ``[1, 3]``; ``q ≥ 4`` will raise
        ``NotImplementedError`` (Laplace approximation pending).
    theta_0_floor
        Hard lower bound on ``θ_0`` over the simplex grid. Default 0.2 per
        Stage 1 spec point 4.
    n_grid_per_axis
        Resolution of the grid along each free axis. Default 12 — gives
        a few hundred candidate ``θ`` vectors after the ordering /
        sum-to-one filter for ``q ≤ 3``.
    cond_warning_threshold
        Condition number above which a diagnostic warning fires
        (Stage 1 spec point 4).
    theta_0_warning_threshold
        Posterior-mean ``θ_0`` below which the same warning fires.
    dirichlet_concentrations
        Concentration parameters for the Dirichlet prior over θ. Defaults
        to the leading ``q+1`` entries of ``(5, 3, 2, 1)``.
    """

    q: int = 2
    theta_0_floor: float = 0.2
    n_grid_per_axis: int = 12
    cond_warning_threshold: float = 100.0
    theta_0_warning_threshold: float = 0.3
    dirichlet_concentrations: tuple[float, ...] | None = None
    theta_grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.q, int) or self.q < 1:
            raise ValueError(f"q must be an int >= 1, got {self.q!r}")
        if self.q > 3:
            raise NotImplementedError(
                f"q={self.q}: grid integration only supported for q ≤ 3. "
                "Laplace approximation for q ≥ 4 is pending."
            )
        if not (0.0 <= self.theta_0_floor < 1.0):
            raise ValueError(
                f"theta_0_floor must lie in [0, 1), got {self.theta_0_floor}"
            )
        if self.n_grid_per_axis < 4:
            raise ValueError(f"n_grid_per_axis must be >= 4, got {self.n_grid_per_axis}")
        if self.dirichlet_concentrations is None:
            default = (5.0, 3.0, 2.0, 1.0)
            self.dirichlet_concentrations = default[: self.q + 1]
        else:
            if len(self.dirichlet_concentrations) != self.q + 1:
                raise ValueError(
                    f"dirichlet_concentrations must have length q+1={self.q + 1}, "
                    f"got {len(self.dirichlet_concentrations)}"
                )
            if any(c <= 0.0 for c in self.dirichlet_concentrations):
                raise ValueError("dirichlet_concentrations entries must all be > 0.")
        # Pre-compute the candidate grid
        self.theta_grid = self._enumerate_thetas()

    # ── SmoothingModel interface ────────────────────────────────────

    def fit(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
        priors: dict[str, Any],
    ) -> DesmoothedResult:
        """Run the Bayesian MA(q) fit and return the desmoothed series + posterior."""
        s_obs, X, factor_names = self._validate_and_prepare(observed_returns, factor_returns)
        priors_validated = self._validate_priors(priors, factor_names)
        mu_0, V_0_inv, a_0, b_0 = self._build_nig_prior(priors_validated, factor_names)

        d = X.shape[1]
        n = X.shape[0]
        a_n = a_0 + n / 2.0
        sign_V0, logdet_V0_inv = np.linalg.slogdet(V_0_inv)
        if sign_V0 <= 0.0:
            raise RuntimeError("prior precision V_0_inv is not positive definite.")
        log_det_V0 = -logdet_V0_inv

        thetas = self.theta_grid
        n_cand = thetas.shape[0]
        if n_cand == 0:
            raise RuntimeError(
                "no candidate θ vectors enumerated; check theta_0_floor / n_grid_per_axis."
            )
        log_marg = np.zeros(n_cand)
        log_prior = np.zeros(n_cand)
        post_w_means = np.zeros((n_cand, d))
        post_w_diag_var = np.zeros((n_cand, d))
        post_b = np.zeros(n_cand)

        for i, theta in enumerate(thetas):
            # Desmooth via lfilter — solves Θ r = s in O(n·q)
            r_theta = signal.lfilter([1.0], theta, s_obs)
            # Drop the first q observations (boundary effects from r_{t<0} = 0)
            r_eff = r_theta[self.q :]
            X_eff = X[self.q :]
            n_eff = len(r_eff)
            # NIG posterior on (β, α, σ²)
            V_n_inv = V_0_inv + X_eff.T @ X_eff
            sign_Vn, logdet_Vn_inv = np.linalg.slogdet(V_n_inv)
            if sign_Vn <= 0.0:
                raise RuntimeError(
                    f"posterior precision V_n_inv not PD at θ={theta!r}."
                )
            log_det_Vn = -logdet_Vn_inv
            mu_n = np.linalg.solve(V_n_inv, V_0_inv @ mu_0 + X_eff.T @ r_eff)
            sse_term = (
                float(r_eff @ r_eff)
                + float(mu_0 @ V_0_inv @ mu_0)
                - float(mu_n @ V_n_inv @ mu_n)
            )
            a_n_eff = a_0 + n_eff / 2.0
            b_n = b_0 + 0.5 * sse_term
            if b_n <= 0.0:
                b_n = 1e-300
            # NIG marginal on r_eff plus Jacobian term -n log θ_0
            log_marg[i] = (
                -0.5 * n_eff * np.log(2.0 * np.pi)
                + 0.5 * (log_det_Vn - log_det_V0)
                + a_0 * np.log(b_0)
                - a_n_eff * np.log(b_n)
                + lgamma(a_n_eff)
                - lgamma(a_0)
                - n_eff * np.log(theta[0])
            )
            post_w_means[i] = mu_n
            V_n = np.linalg.inv(V_n_inv)
            scale = b_n / (a_n_eff - 1.0) if a_n_eff > 1.0 else b_n
            post_w_diag_var[i] = np.diag(V_n) * scale
            post_b[i] = b_n
            log_prior[i] = self._log_prior_density(theta)

        log_post = log_marg + log_prior
        log_post -= log_post.max()
        post_unnorm = np.exp(log_post)
        norm = float(post_unnorm.sum())
        if not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("MA(q) θ posterior failed to normalise.")
        post = post_unnorm / norm

        # Posterior moments
        theta_post_mean = (thetas * post[:, None]).sum(axis=0)
        # Re-normalise to ensure sum-to-one (numerical hygiene)
        theta_post_mean = theta_post_mean / theta_post_mean.sum()
        theta_post_var = ((thetas - theta_post_mean) ** 2 * post[:, None]).sum(axis=0)
        theta_post_std = np.sqrt(np.maximum(theta_post_var, 0.0))

        # w / σ² posterior moments via integration over θ posterior
        w_mean = (post_w_means * post[:, None]).sum(axis=0)
        E_var = (post_w_diag_var * post[:, None]).sum(axis=0)
        E_w_sq = (post_w_means**2 * post[:, None]).sum(axis=0)
        w_var = E_var + E_w_sq - w_mean**2
        w_std = np.sqrt(np.maximum(w_var, 0.0))
        beta_mean = w_mean[:-1]
        beta_std = w_std[:-1]
        alpha_mean = float(w_mean[-1])
        alpha_std = float(w_std[-1])
        sigma2_post_mean = float(((post_b / (a_n - 1.0)) * post).sum())
        sigma_eps_mean = float(np.sqrt(max(sigma2_post_mean, 0.0)))

        # Native-frequency desmoothing using posterior-mean θ
        r_post = signal.lfilter([1.0], theta_post_mean, s_obs)
        true_returns = pd.Series(
            r_post, index=observed_returns.index, name=observed_returns.name
        )

        # Diagnostics
        diagnostics = self._diagnostics(
            theta_post=theta_post_mean,
            n_observations=n,
            factor_names=factor_names,
        )

        smoothing_params = {
            "theta": theta_post_mean.copy(),
            "beta": dict(zip(factor_names, [float(b) for b in beta_mean])),
            "alpha": alpha_mean,
            "sigma_eps": sigma_eps_mean,
        }
        posterior_summary = {
            "theta": {
                "mean": theta_post_mean.copy(),
                "std": theta_post_std.copy(),
                "grid": thetas.copy(),
                "density": post.copy(),
            },
            "beta": {
                name: {"mean": float(beta_mean[k]), "std": float(beta_std[k])}
                for k, name in enumerate(factor_names)
            },
            "alpha": {"mean": alpha_mean, "std": alpha_std},
            "sigma_eps": {"mean": sigma_eps_mean},
        }
        # Cache for sample_posterior
        self._last_fit = {
            "post": post,
            "thetas": thetas,
            "post_w_means": post_w_means,
            "post_w_diag_var": post_w_diag_var,
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
        """Profile log-likelihood at θ (concentrating out (β, α, σ²) by OLS).

        ``smoothing_params`` should be the θ vector of length ``q+1``.
        """
        theta = np.asarray(smoothing_params, dtype=float).reshape(-1)
        if theta.size != self.q + 1:
            return -np.inf
        if not _theta_valid(theta, self.theta_0_floor):
            return -np.inf
        s = np.asarray(observed_returns, dtype=float)
        F = np.asarray(factor_returns, dtype=float)
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        r = signal.lfilter([1.0], theta, s)
        # Drop the first q boundary observations
        r_eff = r[self.q :]
        F_eff = F[self.q :]
        n_eff = len(r_eff)
        X_aug = np.hstack([F_eff, np.ones((n_eff, 1))])
        coefs, *_ = np.linalg.lstsq(X_aug, r_eff, rcond=None)
        resid = r_eff - X_aug @ coefs
        sigma2 = float(np.sum(resid**2) / max(n_eff - X_aug.shape[1], 1))
        if sigma2 <= 0.0:
            return -np.inf
        ll = (
            -0.5 * n_eff * np.log(2.0 * np.pi * sigma2)
            - 0.5 * np.sum(resid**2) / sigma2
            - n_eff * np.log(theta[0])
        )
        return float(ll)

    def log_prior(
        self,
        smoothing_params: np.ndarray,
        prior_config: dict[str, Any],
    ) -> float:
        """Log of the (unnormalised) ordered Dirichlet density on θ."""
        theta = np.asarray(smoothing_params, dtype=float).reshape(-1)
        if theta.size != self.q + 1:
            return -np.inf
        if not _theta_valid(theta, self.theta_0_floor):
            return -np.inf
        return float(self._log_prior_density(theta))

    def desmooth(
        self,
        observed_returns: np.ndarray,
        smoothing_params: np.ndarray,
    ) -> np.ndarray:
        r"""Apply the MA(q) inverse :math:`r = \Theta^{-1} s` via :func:`scipy.signal.lfilter`."""
        theta = np.asarray(smoothing_params, dtype=float).reshape(-1)
        if theta.size != self.q + 1:
            raise ValueError(
                f"smoothing_params must have length q+1={self.q + 1}, got {theta.size}"
            )
        if theta[0] <= 0.0:
            raise ValueError(f"theta[0] must be > 0 to invert Θ, got {theta[0]}")
        s = np.asarray(observed_returns, dtype=float)
        if s.ndim != 1:
            raise ValueError(f"observed_returns must be 1-D, got ndim={s.ndim}")
        return signal.lfilter([1.0], theta, s)

    # ── Posterior sampling ──────────────────────────────────────────

    def sample_posterior(
        self,
        n_samples: int,
        rng: np.random.Generator | int | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw ``n_samples`` joint posterior samples of (θ, β, α, σ²).

        Samples θ from the discrete posterior over the grid; conditional on
        each θ draw, samples σ² from the NIG conditional and ``w | σ²``
        from a Gaussian using the diagonal NIG covariance.
        """
        if not hasattr(self, "_last_fit"):
            raise RuntimeError("call fit() before sample_posterior().")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        cache = self._last_fit
        post = cache["post"]
        thetas = cache["thetas"]
        idx = rng.choice(len(post), size=n_samples, p=post)
        theta_samples = thetas[idx]
        a_n = cache["a_n"]
        b_n_arr = cache["post_b"][idx]
        sigma2 = 1.0 / rng.gamma(shape=a_n, scale=1.0 / b_n_arr)
        w_means = cache["post_w_means"][idx]
        scale_old = b_n_arr / (a_n - 1.0)
        # post_w_diag_var = V_n_diag * scale_old → V_n_diag = post_w_diag_var / scale_old
        V_n_diag = cache["post_w_diag_var"][idx] / scale_old[:, None]
        w_var = V_n_diag * sigma2[:, None]
        w_samples = w_means + rng.normal(size=w_means.shape) * np.sqrt(np.maximum(w_var, 0.0))
        K = w_samples.shape[1] - 1
        return {
            "theta": theta_samples,
            "beta": w_samples[:, :K],
            "alpha": w_samples[:, K],
            "sigma_eps": np.sqrt(sigma2),
            "factor_names": np.array(cache["factor_names"]),
        }

    # ── Internals ───────────────────────────────────────────────────

    def _enumerate_thetas(self) -> np.ndarray:
        """Enumerate ordered, sum-to-one θ vectors with θ_0 ≥ floor."""
        n = self.n_grid_per_axis
        floor = self.theta_0_floor
        q = self.q

        thetas: list[tuple[float, ...]] = []
        if q == 1:
            for t0 in np.linspace(max(floor, 0.5), 1.0, n):
                t1 = 1.0 - t0
                if t1 < 0.0 or t1 > t0:
                    continue
                thetas.append((float(t0), float(t1)))
        elif q == 2:
            for t0 in np.linspace(max(floor, 1.0 / 3.0), 1.0, n):
                if t0 >= 1.0:
                    thetas.append((1.0, 0.0, 0.0))
                    continue
                # Ordered: t1 ∈ [(1-t0)/2, t0]; also t1 ≤ 1-t0 (so t2 ≥ 0); also t1 ≥ t2 = 1-t0-t1
                t1_lo = max((1.0 - t0) / 2.0, 0.0)
                t1_hi = min(t0, 1.0 - t0)
                if t1_hi < t1_lo:
                    continue
                for t1 in np.linspace(t1_lo, t1_hi, n):
                    t2 = 1.0 - t0 - t1
                    if t2 < 0.0 or t2 > t1 + 1e-12:
                        continue
                    t2 = max(t2, 0.0)
                    thetas.append((float(t0), float(t1), float(t2)))
        elif q == 3:
            for t0 in np.linspace(max(floor, 0.25), 1.0, n):
                if t0 >= 1.0:
                    thetas.append((1.0, 0.0, 0.0, 0.0))
                    continue
                # Lower bound on t1 such that ordered with sum=1 is feasible:
                # remaining = 1 - t0, distributed over t1 ≥ t2 ≥ t3 each ≤ t0,
                # so t1 ≥ (1 - t0)/3 (if t0 large) or some larger value.
                t1_lo = max((1.0 - t0) / 3.0, 0.0)
                t1_hi = min(t0, 1.0 - t0)
                if t1_hi < t1_lo:
                    continue
                for t1 in np.linspace(t1_lo, t1_hi, n):
                    rem = 1.0 - t0 - t1
                    # rem distributed to (t2, t3) with t2 ≥ t3 ≥ 0 and t2 ≤ t1
                    if rem < 0.0:
                        continue
                    t2_lo = max(rem / 2.0, 0.0)
                    t2_hi = min(t1, rem)
                    if t2_hi < t2_lo:
                        continue
                    for t2 in np.linspace(t2_lo, t2_hi, n):
                        t3 = rem - t2
                        if t3 < -1e-12 or t3 > t2 + 1e-12:
                            continue
                        t3 = max(t3, 0.0)
                        thetas.append((float(t0), float(t1), float(t2), float(t3)))
        return np.asarray(thetas, dtype=float)

    def _log_prior_density(self, theta: np.ndarray) -> float:
        """Log Dirichlet density at θ (unnormalised). Caller restricts to ordered region."""
        c = np.asarray(self.dirichlet_concentrations, dtype=float)
        # Dirichlet log density: Σ (c_j - 1) log θ_j + log normalisation constant
        # Norm const: Γ(Σ c) / Π Γ(c_j) — same for all θ in the simplex, so we can
        # drop it; but we keep it here because it doesn't hurt and aids future
        # comparisons across different c.
        log_z = gammaln(c.sum()) - gammaln(c).sum()
        with np.errstate(divide="ignore"):
            log_kernel = np.where(theta > 0.0, (c - 1.0) * np.log(theta + 1e-300), 0.0).sum()
        return float(log_z + log_kernel)

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
                "factor_returns must be a non-empty DataFrame for the MA(q) fit."
            )
        if observed_returns.isna().any():
            raise ValueError("observed_returns contains NaN.")
        factor_returns = factor_returns.reindex(observed_returns.index)
        if factor_returns.isna().any().any():
            raise ValueError("factor_returns has NaN after alignment.")
        n_obs = len(observed_returns)
        if n_obs < 4 * (self.q + 1):
            raise ValueError(
                f"need >= {4 * (self.q + 1)} observations for MA({self.q}) fit, got {n_obs}."
            )
        s = observed_returns.to_numpy(dtype=float)
        F = factor_returns.to_numpy(dtype=float)
        X = np.hstack([F, np.ones((n_obs, 1))])
        return s, X, list(factor_returns.columns)

    def _validate_priors(
        self,
        priors: dict[str, Any],
        factor_names: list[str],
    ) -> dict[str, Any]:
        for required in ("beta", "alpha", "sigma_eps"):
            if required not in priors:
                raise ValueError(f"priors missing required key {required!r}")
        if not isinstance(priors["alpha"], NormalPrior):
            raise TypeError("priors['alpha'] must be a NormalPrior.")
        if not isinstance(priors["sigma_eps"], InverseGammaPrior):
            raise TypeError("priors['sigma_eps'] must be an InverseGammaPrior.")
        if not isinstance(priors["beta"], dict):
            raise TypeError("priors['beta'] must be a dict factor_name → NormalPrior.")
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
        theta_post: np.ndarray,
        n_observations: int,
        factor_names: list[str],
    ) -> dict[str, Any]:
        warns: list[str] = []
        flags: dict[str, bool] = {}
        # Build the Θ matrix at posterior mean to compute condition number.
        # For very large n use a smaller block to keep cond() fast.
        n_check = min(n_observations, 250)
        Theta_mat = _build_ma_matrix(theta_post, n_check)
        cond_num = float(np.linalg.cond(Theta_mat))
        flags["condition_number_high"] = cond_num > self.cond_warning_threshold
        flags["theta_0_low"] = float(theta_post[0]) < self.theta_0_warning_threshold
        if flags["condition_number_high"]:
            warns.append(
                f"Θ matrix at posterior mean has condition number {cond_num:.1f} > "
                f"{self.cond_warning_threshold:.0f}; inversion is amplifying noise. "
                "Consider reducing q or applying FallbackPolicy.AUTO."
            )
        if flags["theta_0_low"]:
            warns.append(
                f"posterior θ_0 = {float(theta_post[0]):.3f} below the "
                f"{self.theta_0_warning_threshold:.2f} stability threshold; "
                "MA inversion is borderline. Consider reducing q."
            )
        # Check non-monotonic posterior (ordering)
        if (np.diff(theta_post) > 1e-6).any():
            warns.append(
                f"posterior θ is non-monotone: {theta_post.round(4)!r}; "
                "this is unexpected under the ordered-Dirichlet prior."
            )
        return {
            "method": "ma_glm",
            "q": self.q,
            "n_observations": int(n_observations),
            "theta_post_mean": theta_post.copy(),
            "condition_number": cond_num,
            "flags": flags,
            "warnings": warns,
        }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _theta_valid(theta: np.ndarray, floor: float) -> bool:
    """Validate that θ is non-negative, sums to one (within tol), ordered, θ_0 ≥ floor."""
    if (theta < -1e-9).any():
        return False
    if abs(float(theta.sum()) - 1.0) > 1e-6:
        return False
    if float(theta[0]) < floor - 1e-9:
        return False
    return bool((np.diff(theta) <= 1e-9).all())


def _build_ma_matrix(theta: np.ndarray, n: int) -> np.ndarray:
    """Build the n × n banded lower-triangular Toeplitz matrix with θ on the bands."""
    Theta = np.zeros((n, n))
    q = len(theta) - 1
    for j in range(q + 1):
        if j == 0:
            np.fill_diagonal(Theta, theta[0])
        else:
            np.fill_diagonal(Theta[j:, : n - j], theta[j])
    return Theta


__all__ = ["MAGLMSmoother"]
