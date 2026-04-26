"""
Type protocols and shared result types for the ``private_assets_frequency`` library.

This module is the contract layer: every desmoothing model implements
:class:`SmoothingModel`, every stage emits a :class:`StageValidationResult`,
every pipeline returns objects whose shapes are pinned here. Keeping these
definitions in one place (and free of heavy dependencies on the rest of the
library) is what makes the desmoothing layer pluggable.

References
----------
.. [1] Geltner (1993) — "Estimating Market Values from Appraised Values without
       Assuming an Efficient Market."
.. [2] Getmansky, Lo, Makarov (2004) — "An econometric model of serial
       correlation and illiquidity in hedge fund returns."
.. [3] Shepard (2014), MSCI (2025) — Bayesian desmoothing methodology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# String-valued enums for stable serialisation across versions
# ──────────────────────────────────────────────────────────────────


class Frequency(str, Enum):
    """Native frequency of an asset-class return series."""

    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"

    @property
    def periods_per_year(self) -> int:
        return _PERIODS_PER_YEAR[self]


_PERIODS_PER_YEAR: dict["Frequency", int] = {
    Frequency.QUARTERLY: 4,
    Frequency.MONTHLY: 12,
    Frequency.WEEKLY: 52,
    Frequency.DAILY: 252,  # business days
}


class SmoothingModelKind(str, Enum):
    """Identifier strings for the supported smoothing models."""

    AR1_BAYESIAN = "ar1_bayesian"
    MA_GLM = "ma_glm"
    THRESHOLD_AR1 = "threshold_ar1"
    NO_SMOOTHING = "no_smoothing"
    GELTNER_CLASSIC = "geltner_classic"


class PreprocessingKind(str, Enum):
    """Stage 0 preprocessing options."""

    NONE = "none"
    CARRY_MTM_DECOMPOSITION = "carry_mtm_decomposition"
    REPORTING_LAG_ADJUSTMENT = "reporting_lag_adjustment"


class DisaggregationMethod(str, Enum):
    """Stage 3 disaggregation method."""

    CHOW_LIN = "chow_lin"
    FERNANDEZ = "fernandez"
    LITTERMAN = "litterman"


class AggregationType(str, Enum):
    """Temporal aggregation constraint for returns."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class UncertaintyMode(str, Enum):
    """Whether the pipeline propagates the smoothing-parameter posterior."""

    POINT = "point"
    FULL = "full"


class CorrelationMethod(str, Enum):
    """Cross-strategy correlation estimator."""

    EMPIRICAL = "empirical"
    SHRINKAGE = "shrinkage"


class ValidationStatus(str, Enum):
    """Inter-stage validation outcome."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


# ──────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────


class DesmoothedResult(NamedTuple):
    """Output of :meth:`SmoothingModel.fit`.

    Attributes
    ----------
    true_returns
        The desmoothed return series (point estimate or posterior mean,
        depending on the model). Indexed at the *native* frequency of the
        observed series passed to :meth:`SmoothingModel.fit`.
    smoothing_params
        Mapping containing the point estimate (typically posterior mean) of
        each smoothing parameter, e.g. ``{'lambda': 0.62}`` for AR(1) or
        ``{'theta': np.array([0.55, 0.30, 0.15])}`` for MA(q).
    posterior_summary
        Marginal posterior summaries for each smoothing parameter, e.g.::

            {'lambda': {'mean': 0.62, 'std': 0.08, 'ci_05': 0.48, 'ci_95': 0.76},
             'beta':   {...},
             'sigma_eps': {...}}
    diagnostics
        Model-specific diagnostics: identifiability flags, KL(post‖prior),
        effective sample size, condition numbers, Stage-1→2 validation, etc.
    """

    true_returns: pd.Series
    smoothing_params: dict[str, Any]
    posterior_summary: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class StageValidationResult:
    """Result of an inter-stage validation gate.

    Attributes
    ----------
    stage
        Human-readable name of the stage producing the validation, e.g.
        ``'Stage 1 → Stage 2'``.
    status
        ``PASS`` / ``WARN`` / ``FAIL``.
    checks
        Mapping ``{check_name: passed}`` for each individual check that ran.
    messages
        Free-form messages emitted by failing or warning checks.
    metadata
        Extra context (numbers, ratios) useful for debugging.
    """

    stage: str
    status: ValidationStatus
    checks: dict[str, bool] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is ValidationStatus.PASS

    @property
    def has_warnings(self) -> bool:
        return self.status is not ValidationStatus.PASS

    @property
    def is_failure(self) -> bool:
        return self.status is ValidationStatus.FAIL


# ──────────────────────────────────────────────────────────────────
# SmoothingModel protocol
# ──────────────────────────────────────────────────────────────────


@runtime_checkable
class SmoothingModel(Protocol):
    """Structural protocol every Stage-1 desmoothing model implements.

    Any class implementing :meth:`fit`, :meth:`log_likelihood`,
    :meth:`log_prior`, and :meth:`desmooth` with the signatures below is
    compatible with the pipeline. ``@runtime_checkable`` allows
    ``isinstance(x, SmoothingModel)`` for diagnostic logging — note that this
    only checks for *attribute presence*, not signature compatibility.

    Notes
    -----
    All four methods may assume that the caller has already aligned and
    validated the inputs. Models should *not* mutate their inputs in place.
    """

    def fit(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
        priors: dict[str, Any],
    ) -> DesmoothedResult:
        """Jointly estimate smoothing parameters and factor betas.

        Parameters
        ----------
        observed_returns
            The smoothed (observed) return series at the native frequency.
        factor_returns
            Factor return matrix aligned to ``observed_returns.index``.
        priors
            Mapping of prior specifications. Concrete keys depend on the
            model: AR(1) expects ``{'lambda', 'beta', 'alpha', 'sigma_eps'}``;
            MA(q) expects ``{'theta', 'beta', 'alpha', 'sigma_eps'}``; etc.

        Returns
        -------
        DesmoothedResult
        """
        ...

    def log_likelihood(
        self,
        smoothing_params: np.ndarray,
        observed_returns: np.ndarray,
        factor_returns: np.ndarray,
    ) -> float:
        """Conditional log-likelihood of ``observed_returns`` given parameters."""
        ...

    def log_prior(
        self,
        smoothing_params: np.ndarray,
        prior_config: dict[str, Any],
    ) -> float:
        """Log-prior density evaluated at ``smoothing_params``."""
        ...

    def desmooth(
        self,
        observed_returns: np.ndarray,
        smoothing_params: np.ndarray,
    ) -> np.ndarray:
        """Recover an estimate of the latent true returns given parameters."""
        ...


# ──────────────────────────────────────────────────────────────────
# Prior protocol (for type-checking; concrete impls live in core.config)
# ──────────────────────────────────────────────────────────────────


@runtime_checkable
class Prior(Protocol):
    """A univariate prior distribution exposing logpdf, sample, and mean."""

    def logpdf(self, x: float | np.ndarray) -> float | np.ndarray: ...

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        rng: np.random.Generator | int | None = None,
    ) -> float | np.ndarray: ...

    @property
    def mean(self) -> float: ...


__all__ = [
    "AggregationType",
    "CorrelationMethod",
    "DesmoothedResult",
    "DisaggregationMethod",
    "Frequency",
    "PreprocessingKind",
    "Prior",
    "SmoothingModel",
    "SmoothingModelKind",
    "StageValidationResult",
    "UncertaintyMode",
    "ValidationStatus",
]
