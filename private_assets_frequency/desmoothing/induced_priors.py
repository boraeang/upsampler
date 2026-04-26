"""
Cross-sectional induced-priors iteration (Stage 1 spec point 4).

The single-strategy AR(1) Bayesian fit is fragile when the time-series
alone is uninformative (short series, high noise). The spec's mitigation:
pool information across peer strategies of the *same* asset class — fit
each strategy independently with the user-supplied priors, take the
empirical cross-sectional distribution of the recovered parameters as a
new prior, refit, and iterate to convergence. This is a hierarchical-Bayes
shrinkage in two-stage form.

The iteration converges quickly in practice (a handful of iterations) and
pulls outlier estimates back toward the peer-group mean while leaving
well-identified strategies essentially unchanged.

This module provides:

    * :class:`InducedPriorsRunner` — the iteration loop with
      ``max_iter`` / ``tol`` parameters and convergence diagnostics.

The runner is generic over the smoothing model (anything implementing the
:class:`SmoothingModel` protocol) and only updates the prior *means*;
prior std defaults to the user's input but can be tightened toward the
cross-sectional spread via the ``shrink_std`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..core.config import BetaDist, NormalPrior
from ..core.protocols import DesmoothedResult, SmoothingModel


@dataclass
class InducedPriorsResult:
    """Output of :meth:`InducedPriorsRunner.run`.

    Attributes
    ----------
    fits
        Mapping ``{strategy_name: DesmoothedResult}`` from the *final*
        iteration.
    priors
        The induced priors at convergence (one mapping per strategy).
    n_iterations
        Number of iterations actually performed.
    converged
        ``True`` if convergence was reached within ``max_iter``.
    history
        List of per-iteration cross-sectional summaries — the mean and
        std (across strategies) of each scalar parameter that the runner
        tracks. Useful for diagnostic plots.
    """

    fits: dict[str, DesmoothedResult]
    priors: dict[str, dict[str, Any]]
    n_iterations: int
    converged: bool
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InducedPriorsRunner:
    """Cross-sectional shrinkage iteration over a peer group of strategies.

    Parameters
    ----------
    smoother_factory
        Callable returning a fresh :class:`SmoothingModel` instance per
        strategy per iteration. Each call should return a *new* object so
        per-strategy state (e.g. cached posteriors) does not leak across
        the loop.
    max_iter
        Hard cap on iterations (Stage 1 spec point 4: stop after 10 with a
        warning).
    tol
        Convergence threshold on the maximum change in cross-sectional
        prior means between iterations.
    shrink_std
        If ``True``, also update prior std towards the cross-sectional
        spread (capped above by the original user-supplied std). Default
        ``False`` — only means shrink.
    """

    smoother_factory: Callable[[], SmoothingModel]
    max_iter: int = 10
    tol: float = 0.01
    shrink_std: bool = False

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if not (0.0 < self.tol < 1.0):
            raise ValueError(f"tol must be in (0, 1), got {self.tol}")

    def run(
        self,
        datasets: dict[str, tuple[pd.Series, pd.DataFrame]],
        initial_priors: dict[str, dict[str, Any]],
    ) -> InducedPriorsResult:
        """Run the induced-priors iteration to convergence.

        Parameters
        ----------
        datasets
            Mapping ``{strategy_name: (observed_returns, factor_returns)}``.
            Strategies should be peers within the same asset class — pooling
            across asset classes is not meaningful.
        initial_priors
            Per-strategy priors used in iteration 0.

        Returns
        -------
        InducedPriorsResult
        """
        if not datasets:
            raise ValueError("datasets must contain at least one strategy.")
        if set(datasets) != set(initial_priors):
            raise ValueError(
                f"datasets and initial_priors keys differ: "
                f"datasets={set(datasets)}, priors={set(initial_priors)}"
            )

        priors: dict[str, dict[str, Any]] = {
            name: dict(p) for name, p in initial_priors.items()
        }
        history: list[dict[str, Any]] = []
        last_summary: dict[str, float] | None = None
        fits: dict[str, DesmoothedResult] = {}
        converged = False

        for it in range(self.max_iter):
            fits = {}
            for name, (obs, fac) in datasets.items():
                smoother = self.smoother_factory()
                fits[name] = smoother.fit(obs, fac, priors[name])
            cross_summary = self._cross_sectional_summary(fits)
            history.append(cross_summary)

            if last_summary is not None:
                max_delta = max(
                    abs(cross_summary[k] - last_summary.get(k, np.nan))
                    for k in cross_summary
                    if k.endswith("_mean") and not np.isnan(cross_summary[k])
                )
                if max_delta < self.tol:
                    converged = True
                    break
            last_summary = cross_summary

            # Update priors using the cross-sectional summary
            priors = {
                name: self._update_priors(priors[name], cross_summary, initial_priors[name])
                for name in datasets
            }

        return InducedPriorsResult(
            fits=fits,
            priors=priors,
            n_iterations=it + 1,
            converged=converged,
            history=history,
        )

    # ── helpers ────────────────────────────────────────────────────

    def _cross_sectional_summary(
        self,
        fits: dict[str, DesmoothedResult],
    ) -> dict[str, float]:
        """Mean and std (across strategies) of each scalar fit parameter."""
        # Collect per-strategy scalars
        lambdas: list[float] = []
        alphas: list[float] = []
        beta_per_factor: dict[str, list[float]] = {}
        for fit in fits.values():
            params = fit.smoothing_params
            if "lambda" in params:
                lambdas.append(float(params["lambda"]))
            if "alpha" in params:
                alphas.append(float(params["alpha"]))
            beta = params.get("beta")
            if isinstance(beta, dict):
                for f_name, b in beta.items():
                    beta_per_factor.setdefault(f_name, []).append(float(b))
        summary: dict[str, float] = {}
        if lambdas:
            summary["lambda_mean"] = float(np.mean(lambdas))
            summary["lambda_std"] = float(np.std(lambdas, ddof=1)) if len(lambdas) > 1 else 0.0
        if alphas:
            summary["alpha_mean"] = float(np.mean(alphas))
            summary["alpha_std"] = float(np.std(alphas, ddof=1)) if len(alphas) > 1 else 0.0
        for f_name, vals in beta_per_factor.items():
            summary[f"beta_{f_name}_mean"] = float(np.mean(vals))
            summary[f"beta_{f_name}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        return summary

    def _update_priors(
        self,
        current: dict[str, Any],
        cross_summary: dict[str, float],
        original: dict[str, Any],
    ) -> dict[str, Any]:
        """Update prior means (and optionally stds) using the cross-sectional summary."""
        out = dict(current)
        # λ via Beta moments — match cross-sectional mean / variance to BetaDist(a, b)
        if "lambda" in current and "lambda_mean" in cross_summary:
            mu = cross_summary["lambda_mean"]
            var_cross = cross_summary.get("lambda_std", 0.0) ** 2
            # Convert BetaDist parameters from mean/var; fall back to current var if 0
            current_lambda = current["lambda"]
            if isinstance(current_lambda, BetaDist):
                var = var_cross if var_cross > 0.0 else current_lambda.variance
                a, b = _beta_from_mean_var(mu, var)
                out["lambda"] = BetaDist(a, b)

        # α: cross-sectional mean replaces prior mean; std unchanged unless shrink_std
        if "alpha" in current and "alpha_mean" in cross_summary:
            current_alpha = current["alpha"]
            if isinstance(current_alpha, NormalPrior):
                new_mean = cross_summary["alpha_mean"]
                if self.shrink_std:
                    cross_std = cross_summary.get("alpha_std", current_alpha.std)
                    new_std = min(cross_std, original["alpha"].std) if cross_std > 0 else current_alpha.std
                else:
                    new_std = current_alpha.std
                out["alpha"] = NormalPrior(new_mean, new_std)

        # β per factor
        if "beta" in current and isinstance(current["beta"], dict):
            new_beta: dict[str, NormalPrior] = {}
            for f_name, prior in current["beta"].items():
                key_mean = f"beta_{f_name}_mean"
                if isinstance(prior, NormalPrior) and key_mean in cross_summary:
                    new_mean = cross_summary[key_mean]
                    if self.shrink_std:
                        cross_std = cross_summary.get(f"beta_{f_name}_std", prior.std)
                        new_std = (
                            min(cross_std, original["beta"][f_name].std)
                            if cross_std > 0 else prior.std
                        )
                    else:
                        new_std = prior.std
                    new_beta[f_name] = NormalPrior(new_mean, new_std)
                else:
                    new_beta[f_name] = prior
            out["beta"] = new_beta
        return out


def _beta_from_mean_var(mean: float, var: float) -> tuple[float, float]:
    """Convert a (mean, variance) into BetaDist parameters ``(a, b)``.

    Falls back to a Beta(2, 2) shape if the moments are infeasible
    (mean outside (0, 1), or var > mean·(1-mean)).
    """
    if not (0.0 < mean < 1.0):
        return 2.0, 2.0
    max_var = mean * (1.0 - mean)
    if var <= 0.0 or var >= max_var:
        return 2.0, 2.0
    common = (mean * (1.0 - mean) / var) - 1.0
    a = mean * common
    b = (1.0 - mean) * common
    if a <= 0.0 or b <= 0.0:
        return 2.0, 2.0
    return float(a), float(b)


__all__ = ["InducedPriorsResult", "InducedPriorsRunner"]
