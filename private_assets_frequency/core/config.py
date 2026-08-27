"""
Configuration objects for ``private_assets_frequency``.

This module owns the user-facing knobs:

    * :class:`NormalPrior` and :class:`BetaDist` — concrete prior distributions
      backing the Bayesian desmoothers.
    * :class:`InverseGammaPrior` — extra prior for the idiosyncratic variance
      σ_ε in the AR(1) and threshold AR(1) models.
    * :class:`AssetClassConfig` — the bundle that fully specifies a single
      strategy: smoothing model, preprocessing, priors, factors, public proxy.
    * :class:`FallbackPolicy` — strict / warn / auto degradation behaviour.

Design notes
------------
All prior classes are immutable (``frozen=True``) so instances can be safely
shared across pipeline runs. They expose a uniform interface (``logpdf``,
``sample``, ``mean``) so the desmoothing layer can treat them
polymorphically.

``AssetClassConfig`` is also frozen and uses ``dataclasses.field`` defaults
for collection types. Validation lives in ``__post_init__``: it cross-checks
that the priors required by the chosen smoothing model are actually present.
The intent is to fail loudly at construction time rather than at fit time
when partial state is harder to reason about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from .protocols import (
    Frequency,
    PreprocessingKind,
    SmoothingModelKind,
)

# ──────────────────────────────────────────────────────────────────
# Fallback policy
# ──────────────────────────────────────────────────────────────────


class FallbackPolicy(str, Enum):
    """How the pipeline reacts to identifiability / sanity warnings.

    * ``STRICT`` — any warning is escalated to an exception.
    * ``WARN``   — warnings are surfaced via :mod:`warnings` and stored on
      the result object; the pipeline continues with the suspect output.
    * ``AUTO``   — automatic mitigations defined per stage are applied
      silently (e.g. fall back to classic Geltner if the Bayesian λ posterior
      is flat). Warnings are still recorded but execution does not pause.
    """

    STRICT = "strict"
    WARN = "warn"
    AUTO = "auto"


# ──────────────────────────────────────────────────────────────────
# Prior distributions
# ──────────────────────────────────────────────────────────────────


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


@dataclass(frozen=True)
class NormalPrior:
    """Univariate normal prior :math:`x \\sim \\mathcal{N}(\\mu, \\sigma^2)`.

    Used for factor betas and the alpha intercept.
    """

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError(f"NormalPrior mean must be finite, got {self.mean}")
        if not (math.isfinite(self.std) and self.std > 0):
            raise ValueError(f"NormalPrior std must be positive and finite, got {self.std}")

    def logpdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return stats.norm.logpdf(x, loc=self.mean, scale=self.std)

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        rng: np.random.Generator | int | None = None,
    ) -> float | np.ndarray:
        out = _as_rng(rng).normal(self.mean, self.std, size=size)
        return out

    @property
    def variance(self) -> float:
        return self.std**2


@dataclass(frozen=True)
class BetaDist:
    """Beta distribution :math:`x \\sim \\mathrm{Beta}(a, b)` on :math:`[0, 1]`.

    Used as a prior for the AR(1) smoothing parameter λ and for the Threshold
    AR(1) regime-specific λ values. The default support is the full interval
    ``[0, 1]``; for the AR(1) desmoother the grid is constructed on
    ``[lower, upper]`` to avoid the numerical blow-up at λ → 1, but the prior
    itself is simply Beta — clipping happens at the integration layer.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        for name, value in (("a", self.a), ("b", self.b)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"BetaDist parameter {name} must be positive, got {value}")

    def logpdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return stats.beta.logpdf(x, self.a, self.b)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return stats.beta.pdf(x, self.a, self.b)

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        rng: np.random.Generator | int | None = None,
    ) -> float | np.ndarray:
        return _as_rng(rng).beta(self.a, self.b, size=size)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def variance(self) -> float:
        s = self.a + self.b
        return (self.a * self.b) / (s * s * (s + 1.0))


@dataclass(frozen=True)
class InverseGammaPrior:
    """Inverse-Gamma prior with shape ``alpha`` and scale ``beta``.

    Used as the conjugate prior for the AR(1) idiosyncratic variance
    :math:`\\sigma_\\varepsilon^2`. The spec specifies
    :math:`\\sigma_\\varepsilon \\sim \\text{Inverse-Gamma}(3, 0.02)`.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"InverseGammaPrior parameter {name} must be positive, got {value}")

    def logpdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return stats.invgamma.logpdf(x, self.alpha, scale=self.beta)

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        rng: np.random.Generator | int | None = None,
    ) -> float | np.ndarray:
        return stats.invgamma.rvs(self.alpha, scale=self.beta, size=size, random_state=_as_rng(rng))

    @property
    def mean(self) -> float:
        if self.alpha <= 1.0:
            return float("inf")
        return self.beta / (self.alpha - 1.0)


# ──────────────────────────────────────────────────────────────────
# AssetClassConfig
# ──────────────────────────────────────────────────────────────────

# Models that consume each preset field
_REQUIRES_LAMBDA: frozenset[str] = frozenset(
    {SmoothingModelKind.AR1_BAYESIAN.value, SmoothingModelKind.GELTNER_CLASSIC.value}
)
_REQUIRES_THRESHOLD: frozenset[str] = frozenset({SmoothingModelKind.THRESHOLD_AR1.value})
_REQUIRES_MA: frozenset[str] = frozenset({SmoothingModelKind.MA_GLM.value})
# Factor-free models: no beta_priors required (pure time-series desmoothers).
_FACTOR_FREE: frozenset[str] = frozenset(
    {SmoothingModelKind.NO_SMOOTHING.value, SmoothingModelKind.OKUNEV_WHITE.value}
)


def _coerce_str_enum(
    value: str | Enum | None,
    enum_cls: type[Enum],
    field_name: str,
) -> str | None:
    """Validate that ``value`` is a member of ``enum_cls``; return canonical str.

    Accepts the enum instance, its string value, or ``None``. Raises
    ``ValueError`` for unknown values.
    """
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value.value
    try:
        return enum_cls(value).value
    except ValueError as exc:
        valid = ", ".join(repr(m.value) for m in enum_cls)
        raise ValueError(
            f"{field_name}={value!r} is not a valid {enum_cls.__name__}; valid: {valid}"
        ) from exc


@dataclass(frozen=True)
class AssetClassConfig:
    """Full configuration for one strategy / asset class.

    The same dataclass is reused across all five asset classes; fields not
    relevant to a given smoothing model default to ``None`` and are validated
    at construction time. See ``pipeline/presets.py`` for built-in instances.

    Required fields
    ---------------
    asset_class
        Free-form string label (``'private_equity'``, ``'hedge_fund'``, etc.).
        Used only for logging and result grouping.
    smoothing_model
        Identifier from :class:`SmoothingModelKind`.
    native_frequency
        Identifier from :class:`Frequency`.

    Optional fields
    ---------------
    preprocessing
        Stage 0 preprocessor identifier from :class:`PreprocessingKind`. ``None``
        is shorthand for ``"none"`` (pass-through).
    beta_priors, alpha_prior, sigma_eps_prior
        Priors for the factor regression. Required by every Bayesian model.
    lambda_prior
        Beta prior on λ for AR(1) and Geltner-classic models.
    lambda_prior_normal, lambda_prior_stress, regime_indicator, regime_threshold
        Threshold-AR(1) specific. ``regime_indicator`` is the *name* of the
        indicator series the user must supply at pipeline-run time;
        ``regime_threshold`` is the cutoff defining the stress regime.
    ma_lags, theta_prior
        MA(q) specific. ``ma_lags`` is q (number of lagged terms, so the MA
        weight vector has length ``ma_lags + 1``). ``theta_prior`` selects the
        constraint structure on θ (``'ordered_dirichlet'`` or
        ``'stick_breaking'``).
    reporting_lag_months
        Stage 0 reporting-lag adjustment for hedge funds. Zero means no shift.
    default_factors
        Names of the factor return columns that the strategy expects. The
        pipeline runner uses these to subset ``factor_returns_*`` arguments.
    public_proxy
        Name of the public benchmark used for sanity anchors.
    notes
        Free-form text shown in result summaries.
    """

    asset_class: str
    smoothing_model: str
    native_frequency: str

    preprocessing: str | None = None

    beta_priors: dict[str, NormalPrior] = field(default_factory=dict)
    alpha_prior: NormalPrior | None = None
    sigma_eps_prior: InverseGammaPrior | None = None

    # AR(1) / Geltner
    lambda_prior: BetaDist | None = None

    # Threshold AR(1)
    lambda_prior_normal: BetaDist | None = None
    lambda_prior_stress: BetaDist | None = None
    regime_indicator: str | None = None
    regime_threshold: float | None = None

    # MA(q)
    ma_lags: int | None = None
    theta_prior: str | None = None

    # Hedge funds
    reporting_lag_months: int = 0

    # Factor / proxy metadata
    default_factors: tuple[str, ...] = ()
    public_proxy: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        # Normalise enum-valued strings up-front
        object.__setattr__(
            self,
            "smoothing_model",
            _coerce_str_enum(self.smoothing_model, SmoothingModelKind, "smoothing_model"),
        )
        object.__setattr__(
            self,
            "native_frequency",
            _coerce_str_enum(self.native_frequency, Frequency, "native_frequency"),
        )
        if self.preprocessing is not None:
            object.__setattr__(
                self,
                "preprocessing",
                _coerce_str_enum(self.preprocessing, PreprocessingKind, "preprocessing"),
            )

        # Coerce default_factors to tuple for hashability / immutability
        if not isinstance(self.default_factors, tuple):
            object.__setattr__(self, "default_factors", tuple(self.default_factors))

        # Validate prior coverage by model
        model = self.smoothing_model
        if model in _REQUIRES_LAMBDA:
            if self.lambda_prior is None:
                raise ValueError(
                    f"smoothing_model={model!r} requires lambda_prior (BetaDist)."
                )
        if model in _REQUIRES_THRESHOLD:
            missing = [
                name
                for name, value in (
                    ("lambda_prior_normal", self.lambda_prior_normal),
                    ("lambda_prior_stress", self.lambda_prior_stress),
                    ("regime_indicator", self.regime_indicator),
                    ("regime_threshold", self.regime_threshold),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"smoothing_model={model!r} requires fields {missing}."
                )
        if model in _REQUIRES_MA:
            if self.ma_lags is None or self.ma_lags < 1:
                raise ValueError(
                    f"smoothing_model={model!r} requires ma_lags >= 1, got {self.ma_lags!r}."
                )

        # Beta priors must be a non-empty mapping for any factor-based model.
        # Factor-free desmoothers (no_smoothing, okunev_white) are exempt.
        if model not in _FACTOR_FREE and not self.beta_priors:
            raise ValueError(
                f"smoothing_model={model!r} requires non-empty beta_priors."
            )

        # Reporting lag adjustment must have a non-negative value
        if self.reporting_lag_months < 0:
            raise ValueError(
                f"reporting_lag_months must be >= 0, got {self.reporting_lag_months}."
            )

        # If preprocessing is reporting_lag_adjustment, lag must be >= 1
        if (
            self.preprocessing == PreprocessingKind.REPORTING_LAG_ADJUSTMENT.value
            and self.reporting_lag_months < 1
        ):
            raise ValueError(
                "preprocessing='reporting_lag_adjustment' requires reporting_lag_months >= 1."
            )

    # ── convenience accessors ───────────────────────────────────────────

    @property
    def smoothing_kind(self) -> SmoothingModelKind:
        return SmoothingModelKind(self.smoothing_model)

    @property
    def frequency(self) -> Frequency:
        return Frequency(self.native_frequency)

    def with_overrides(self, **changes: Any) -> "AssetClassConfig":
        """Return a copy of this config with the given fields replaced.

        Convenience for users who want to start from a preset and tweak one
        prior without spelling out every field.
        """
        from dataclasses import replace

        return replace(self, **changes)


__all__ = [
    "AssetClassConfig",
    "BetaDist",
    "FallbackPolicy",
    "InverseGammaPrior",
    "NormalPrior",
]
