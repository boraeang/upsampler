"""
Bundled residual diagnostic suite.

Wraps :mod:`decomposition.residual_analysis` into a single
:func:`run_residual_diagnostics` call that produces a uniform
:class:`DiagnosticBundle` for the pipeline result. The bundle is what's
exposed under ``result.diagnostics`` for inspection / reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..decomposition.residual_analysis import (
    ArchLMResult,
    FittedDistribution,
    JarqueBeraResult,
    LjungBoxResult,
    arch_lm,
    fit_best_distribution,
    jarque_bera,
    ljung_box,
)


@dataclass(frozen=True)
class DiagnosticBundle:
    """Aggregated residual diagnostics for one strategy.

    Attributes
    ----------
    ljung_box
        Ljung-Box test results at lags ``(5, 10, 20)`` (truncated to the
        available series length).
    arch_lm
        ARCH-LM test result at ``nlags=5``.
    jarque_bera
        Jarque-Bera test result with skew + kurtosis.
    fitted_distribution
        Best-fit residual distribution (lowest AIC) out of normal /
        Student-t / skewed-t.
    n_observations
        Number of residual observations used.
    warnings
        Free-form messages emitted by the diagnostic checks (e.g. ARCH
        effects detected; non-normal residuals).
    """

    ljung_box: LjungBoxResult
    arch_lm: ArchLMResult
    jarque_bera: JarqueBeraResult
    fitted_distribution: FittedDistribution
    n_observations: int
    warnings: tuple[str, ...]


def run_residual_diagnostics(
    residuals: pd.Series | np.ndarray,
    *,
    ljung_box_lags: tuple[int, ...] = (5, 10, 20),
    arch_lm_nlags: int = 5,
    distribution_candidates: tuple[str, ...] = ("normal", "student_t", "skewed_t"),
    alpha: float = 0.05,
) -> DiagnosticBundle:
    """Run the bundled residual diagnostic suite.

    Parameters
    ----------
    residuals
        Residual series from Stage 2 (or similar).
    ljung_box_lags
        Lags at which to test for autocorrelation. Lags >= len(residuals)
        are silently dropped.
    arch_lm_nlags
        Number of lags for ARCH-LM.
    distribution_candidates
        Distribution families to try; the lowest-AIC one is returned.
    alpha
        Significance level used for warning generation.
    """
    arr = (
        residuals.to_numpy(dtype=float)
        if isinstance(residuals, pd.Series)
        else np.asarray(residuals, dtype=float)
    )
    n = len(arr)
    valid_lags = tuple(L for L in ljung_box_lags if L < n)
    if not valid_lags:
        valid_lags = (max(1, n // 4),)
    lb = ljung_box(arr, lags=valid_lags)
    arch_nlags = min(arch_lm_nlags, max(n - 2, 1))
    al = arch_lm(arr, nlags=arch_nlags)
    jb = jarque_bera(arr)
    fitted = fit_best_distribution(arr, candidates=distribution_candidates)

    warns: list[str] = []
    for lag, p in zip(lb.lags, lb.p_value):
        if p <= alpha:
            warns.append(
                f"residual autocorrelation detected at lag {lag} (p={p:.3f}) — "
                "factor model may be missing a serially correlated factor."
            )
            break
    if al.has_arch(alpha=alpha):
        warns.append(
            f"ARCH effects detected (p={al.p_value:.3f}) — residual variance is "
            "time-varying; consider GARCH-style residual model in Stage 4 simulation."
        )
    if not jb.passes(alpha=alpha):
        warns.append(
            f"residuals are non-normal (JB p={jb.p_value:.3f}, skew={jb.skewness:.2f}, "
            f"excess kurtosis={jb.kurtosis:.2f}) — best fit is "
            f"{fitted.name!r}; use this distribution for Stage 4 simulation."
        )

    return DiagnosticBundle(
        ljung_box=lb,
        arch_lm=al,
        jarque_bera=jb,
        fitted_distribution=fitted,
        n_observations=int(n),
        warnings=tuple(warns),
    )


__all__ = ["DiagnosticBundle", "run_residual_diagnostics"]
