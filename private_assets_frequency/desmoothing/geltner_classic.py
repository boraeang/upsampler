r"""
Classic (non-Bayesian) Geltner AR(1) desmoothing.

This is the original method from Geltner (1993, "Estimating Market Values
from Appraised Values without Assuming an Efficient Market"). It serves two
purposes in this library:

    1. **Baseline comparator** — sanity check for the Bayesian AR(1) results.
       If the Geltner point estimate of λ disagrees materially with the
       Bayesian posterior mean, something is suspect.
    2. **Fallback** — when the Bayesian AR(1) λ posterior is flat
       (un-identified), the pipeline falls back to Geltner-classic with a
       conservative λ override (default 0.5). See ``FallbackPolicy`` and
       ``pipeline/fallback.py``.

Method
------
Under the AR(1) smoothing model :math:`s_t = (1-\lambda) r_t + \lambda
s_{t-1}` with iid true returns :math:`r_t`, the lag-1 autocovariance of the
observed series satisfies (in steady state)::

    Cov(s_t, s_{t-1}) = λ · Var(s_t)

so :math:`\hat\lambda = \rho_1`, the sample lag-1 autocorrelation of the
observed returns. The desmoothed series is then::

    r_t = (s_t - \hat\lambda · s_{t-1}) / (1 - \hat\lambda)

The factor regression is OLS — priors are ignored (they exist only to keep
the protocol surface uniform with the Bayesian models). The class accepts an
optional ``lambda_override`` so the pipeline's fallback path can pin a
conservative value without re-estimating.

References
----------
.. [1] Geltner (1993), "Estimating Market Values from Appraised Values
       without Assuming an Efficient Market," *Journal of Real Estate
       Finance and Economics* 8: 325-345.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.protocols import DesmoothedResult

_LAMBDA_FLOOR = 0.01
_LAMBDA_CEIL = 0.95


@dataclass
class GeltnerClassicSmoother:
    """Non-Bayesian AR(1) desmoother using sample lag-1 autocorrelation.

    Parameters
    ----------
    lambda_override
        If provided, skips estimation and uses this value of λ. Used by the
        pipeline's ``FallbackPolicy.AUTO`` when the Bayesian fit is
        un-identified — the recommended override is 0.5 per the spec
        ("fall back to classic Geltner with a conservative λ = 0.5").
    lambda_floor, lambda_ceil
        Bounds applied to the estimated λ before desmoothing — guarantees
        the ``1/(1-λ)`` transform never blows up. Defaults match the AR(1)
        Bayesian grid bounds.
    """

    lambda_override: float | None = None
    lambda_floor: float = _LAMBDA_FLOOR
    lambda_ceil: float = _LAMBDA_CEIL

    def __post_init__(self) -> None:
        if not (0.0 <= self.lambda_floor < self.lambda_ceil < 1.0):
            raise ValueError(
                f"require 0 <= lambda_floor ({self.lambda_floor}) < "
                f"lambda_ceil ({self.lambda_ceil}) < 1."
            )
        if self.lambda_override is not None and not (
            self.lambda_floor <= self.lambda_override <= self.lambda_ceil
        ):
            raise ValueError(
                f"lambda_override={self.lambda_override} outside "
                f"[{self.lambda_floor}, {self.lambda_ceil}]."
            )

    # ── SmoothingModel interface ────────────────────────────────────

    def fit(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
        priors: dict[str, Any],
    ) -> DesmoothedResult:
        """Estimate λ, optionally regress on factors, return desmoothed series.

        ``priors`` is accepted for protocol compatibility but ignored.
        """
        if not isinstance(observed_returns, pd.Series):
            raise TypeError(
                f"observed_returns must be a Series, got {type(observed_returns).__name__}"
            )
        if observed_returns.isna().any():
            raise ValueError("observed_returns contains NaN.")
        if len(observed_returns) < 4:
            raise ValueError(
                f"observed_returns has only {len(observed_returns)} obs; need >= 4."
            )

        s_arr = observed_returns.to_numpy(dtype=float)
        n_obs = len(s_arr)

        # Estimate λ via lag-1 autocorrelation, or use override
        was_clipped = False
        if self.lambda_override is not None:
            lam_raw = float(self.lambda_override)
            lam = lam_raw
        else:
            lam_raw = _lag1_autocorr(s_arr)
            lam = float(np.clip(lam_raw, self.lambda_floor, self.lambda_ceil))
            was_clipped = lam != lam_raw

        # Desmooth
        r_arr = self.desmooth(s_arr, np.array([lam]))
        true_returns = pd.Series(r_arr, index=observed_returns.index, name=observed_returns.name)

        # Factor regression on the desmoothed series (OLS)
        beta: dict[str, float] = {}
        alpha: float = 0.0
        sigma_eps: float = float("nan")
        r_squared: float = float("nan")
        if factor_returns is not None and not factor_returns.empty:
            factor_returns = factor_returns.reindex(observed_returns.index)
            if factor_returns.isna().any().any():
                raise ValueError("factor_returns has NaN values after alignment.")
            X = factor_returns.to_numpy(dtype=float)
            X_aug = np.hstack([X, np.ones((n_obs, 1))])
            try:
                coefs, residuals_ss, rank, _ = np.linalg.lstsq(X_aug, r_arr, rcond=None)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(f"OLS factor regression failed: {exc}") from exc
            beta_arr = coefs[:-1]
            alpha = float(coefs[-1])
            beta = {col: float(b) for col, b in zip(factor_returns.columns, beta_arr)}
            resid = r_arr - X_aug @ coefs
            df = max(n_obs - rank, 1)
            sigma_eps = float(np.sqrt(np.sum(resid**2) / df))
            ss_tot = float(np.sum((r_arr - r_arr.mean()) ** 2))
            r_squared = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0.0 else float("nan")

        # Diagnostics
        warnings: list[str] = []
        if was_clipped:
            warnings.append(
                f"raw lag-1 autocorrelation {lam_raw:.4f} clipped to "
                f"[{self.lambda_floor}, {self.lambda_ceil}]."
            )
        if self.lambda_override is None and abs(lam_raw) < 0.05:
            warnings.append(
                "estimated λ is near zero — observed series shows no detectable "
                "autocorrelation; consider 'no_smoothing' instead."
            )

        diagnostics: dict[str, Any] = {
            "method": "geltner_classic",
            "n_observations": n_obs,
            "lambda_raw": float(lam_raw),
            "lambda_clipped": bool(was_clipped),
            "lambda_overridden": self.lambda_override is not None,
            "factor_r_squared": r_squared,
            "warnings": warnings,
        }

        smoothing_params: dict[str, Any] = {
            "lambda": float(lam),
        }
        if beta:
            smoothing_params["beta"] = beta
            smoothing_params["alpha"] = alpha
            smoothing_params["sigma_eps"] = sigma_eps

        # No posterior — point estimate only
        posterior_summary: dict[str, Any] = {
            "lambda": {"point": float(lam)},
        }
        if beta:
            posterior_summary["beta"] = {k: {"point": v} for k, v in beta.items()}
            posterior_summary["alpha"] = {"point": alpha}

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
        """Gaussian-AR(1) conditional log-likelihood of observed returns given λ.

        Under :math:`s_t = (1-\\lambda) r_t + \\lambda s_{t-1}` with
        :math:`r_t \\sim \\mathcal{N}(\\mu, \\sigma^2)` iid, conditional on
        :math:`s_0` the observed series is Gaussian with::

            E[s_t | s_{t-1}]  = λ s_{t-1} + (1-λ) μ
            Var(s_t | s_{t-1}) = (1-λ)^2 σ^2

        ``μ`` and ``σ^2`` are concentrated out by their conditional MLEs,
        so the returned value is the profile log-likelihood at λ.
        """
        lam = float(smoothing_params[0])
        s = np.asarray(observed_returns, dtype=float)
        if not (0.0 <= lam < 1.0):
            return -np.inf
        s_t = s[1:]
        s_lag = s[:-1]
        n = len(s_t)
        # MLE for μ and σ² conditional on λ
        residuals = s_t - lam * s_lag
        mu_hat = residuals.mean() / (1.0 - lam)
        sse = float(np.sum((residuals - (1.0 - lam) * mu_hat) ** 2))
        sigma2_hat = sse / (n * (1.0 - lam) ** 2)
        if sigma2_hat <= 0.0:
            return -np.inf
        ll = (
            -0.5 * n * np.log(2.0 * np.pi)
            - n * np.log((1.0 - lam) * np.sqrt(sigma2_hat))
            - 0.5 * sse / ((1.0 - lam) ** 2 * sigma2_hat)
        )
        return float(ll)

    def log_prior(
        self,
        smoothing_params: np.ndarray,
        prior_config: dict[str, Any],
    ) -> float:
        """Improper uniform prior on λ over ``[lambda_floor, lambda_ceil]``."""
        lam = float(smoothing_params[0])
        if self.lambda_floor <= lam <= self.lambda_ceil:
            return 0.0
        return -np.inf

    def desmooth(
        self,
        observed_returns: np.ndarray,
        smoothing_params: np.ndarray,
    ) -> np.ndarray:
        r"""Apply the AR(1) inverse :math:`r_t = (s_t - \lambda s_{t-1}) / (1-\lambda)`.

        The first observation is left unchanged (no lag available); this is
        a conventional choice. The remaining ``T-1`` observations are
        desmoothed in closed form.
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


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _lag1_autocorr(x: np.ndarray) -> float:
    """Sample lag-1 autocorrelation, mean-centred and normalised by sample variance.

    Uses the standard ``ρ_1 = Σ(x_t - x̄)(x_{t-1} - x̄) / Σ(x_t - x̄)^2`` form.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("need >= 2 observations for lag-1 autocorrelation.")
    # Detect a constant series robustly — float-imprecision in `mean()` gives
    # tiny but non-zero `x_centered` values for which num/denom is ill-defined.
    if float(np.ptp(x)) <= np.finfo(float).eps * max(1.0, float(np.abs(x).max())):
        return 0.0
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered**2))
    if denom <= 0.0:
        return 0.0
    num = float(np.sum(x_centered[1:] * x_centered[:-1]))
    return num / denom


__all__ = ["GeltnerClassicSmoother"]
