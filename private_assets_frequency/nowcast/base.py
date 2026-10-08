r"""
The nowcaster contract: :class:`Nowcaster`, :class:`NowcastResult`.

Every model in ``nowcast/models/`` implements :class:`Nowcaster`, so the
evaluation harness and the pipeline integration can treat them
interchangeably. The protocol is deliberately narrow — two methods — and both
of them take an :class:`~private_assets_frequency.nowcast.information_set.InformationSet`
rather than raw data, which is what makes the no-leakage guarantee structural
rather than a matter of discipline.

Two nowcast targets
-------------------
``target='reported'`` (default)
    The smoothed return the index provider will publish for quarter ``Q``. This
    is the genuine nowcasting problem: the quantity exists, is unknown, and
    will be revealed.

``target='true'``
    The unsmoothed economic return. This is **not** a nowcast in the same
    sense. Its predictable part is just the systematic component
    :math:`\alpha + \beta \cdot F_Q` of the factor model, and the idiosyncratic
    part has expectation zero by construction. The resulting interval is
    dominated by idiosyncratic risk, which no amount of additional public data
    reduces — so a poor ``target='true'`` R² is the expected result, not a
    model failure. Comparing accuracy across the two targets compares two
    different questions.

References
----------
.. [1] Cohen, Mantoan, Nesheim, de Paula, Turrell & Yang (2023) —
       "Nowcasting using regression on signatures," arXiv:2305.10256v3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .information_set import InformationSet

__all__ = [
    "INTERVAL_LEVELS",
    "NowcastResult",
    "NowcastTarget",
    "Nowcaster",
    "make_nowcast_result",
]


NowcastTarget = Literal["reported", "true"]

INTERVAL_LEVELS: tuple[float, float] = (0.80, 0.95)
"""Nominal coverage of the two intervals every :class:`NowcastResult` carries."""


# ──────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NowcastResult:
    """One model's nowcast of one quarter, with its predictive distribution.

    Attributes
    ----------
    quarter
        The quarter being nowcast.
    horizon
        Quarters between the last published quarter and ``quarter``; ``1`` for
        the first unpublished quarter.
    as_of
        Information-set date the nowcast was made at. Together with ``quarter``
        this pins down exactly what was knowable.
    model
        ``Nowcaster.name`` of the producing model.
    target
        ``'reported'`` or ``'true'`` — see the module docstring.
    point
        Point nowcast. For the linear models this is the conditional mean
        obtained by chaining point estimates through the recursion, which
        coincides with the mean of ``draws`` up to simulation error.
    draws
        Predictive draws, shape ``(n_draws,)``.
    interval_80, interval_95
        ``(lower, upper)`` quantiles of ``draws`` at 80 % and 95 %.
    interval_method
        How the predictive distribution was built — e.g.
        ``'empirical_recursive'``, ``'empirical_direct'``, ``'parametric'``.
        Suffixed with ``'+in_sample_inflated'`` when the residual pool fell
        back to inflated in-sample residuals.
    diagnostics
        Features used, ``n_train``, feature budget, residual-pool provenance,
        fallback events, data mode inherited from the information set.
    """

    quarter: pd.Period
    horizon: int
    as_of: pd.Timestamp
    model: str
    target: NowcastTarget
    point: float
    draws: np.ndarray
    interval_80: tuple[float, float]
    interval_95: tuple[float, float]
    interval_method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        """Number of predictive draws."""
        return int(self.draws.size)

    @property
    def draw_mean(self) -> float:
        """Mean of the predictive draws."""
        return float(np.mean(self.draws))

    @property
    def draw_std(self) -> float:
        """Standard deviation of the predictive draws."""
        return float(np.std(self.draws, ddof=1))

    def interval(self, level: float) -> tuple[float, float]:
        """Predictive interval at an arbitrary nominal coverage.

        Parameters
        ----------
        level
            Nominal coverage in ``(0, 1)``, e.g. ``0.90``.

        Returns
        -------
        tuple[float, float]
            Equal-tailed ``(lower, upper)`` quantiles of ``draws``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level!r}.")
        tail = (1.0 - level) / 2.0
        lo, hi = np.quantile(self.draws, [tail, 1.0 - tail])
        return float(lo), float(hi)

    def covers(self, actual: float, level: float = 0.80) -> bool:
        """Whether the interval at ``level`` contains ``actual``.

        Parameters
        ----------
        actual
            Realised value.
        level
            Nominal coverage.

        Returns
        -------
        bool
        """
        lo, hi = self.interval(level)
        return bool(lo <= actual <= hi)

    def to_row(self) -> dict[str, Any]:
        """Flatten to a dict suitable for a results table (``draws`` omitted)."""
        return {
            "quarter": self.quarter,
            "horizon": self.horizon,
            "as_of": self.as_of,
            "model": self.model,
            "target": self.target,
            "point": self.point,
            "lower_80": self.interval_80[0],
            "upper_80": self.interval_80[1],
            "lower_95": self.interval_95[0],
            "upper_95": self.interval_95[1],
            "interval_method": self.interval_method,
            "n_train": self.diagnostics.get("n_train"),
            "data_mode": self.diagnostics.get("data_mode"),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lo, hi = self.interval_80
        return (
            f"NowcastResult({self.model}, {self.quarter}, h={self.horizon}, "
            f"target={self.target!r}, point={self.point:+.4f}, "
            f"80%=[{lo:+.4f}, {hi:+.4f}], as_of={self.as_of.date()})"
        )


def make_nowcast_result(
    *,
    quarter: pd.Period,
    horizon: int,
    as_of: pd.Timestamp,
    model: str,
    target: NowcastTarget,
    point: float,
    draws: np.ndarray,
    interval_method: str,
    diagnostics: dict[str, Any] | None = None,
) -> NowcastResult:
    """Build a :class:`NowcastResult`, deriving the intervals from ``draws``.

    Parameters
    ----------
    quarter, horizon, as_of, model, target, point, interval_method
        Passed through to :class:`NowcastResult`.
    draws
        Predictive draws; copied and made read-only so a result cannot be
        mutated after the fact.
    diagnostics
        Optional diagnostics mapping.

    Returns
    -------
    NowcastResult

    Raises
    ------
    ValueError
        If ``draws`` is empty or contains non-finite values, or ``point`` is not
        finite. A silently NaN nowcast is worse than a loud failure.
    """
    arr = np.asarray(draws, dtype=float).reshape(-1).copy()
    if arr.size == 0:
        raise ValueError(f"{model}: predictive draws for {quarter} are empty.")
    if not np.all(np.isfinite(arr)):
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(
            f"{model}: {n_bad} of {arr.size} predictive draws for {quarter} are "
            "non-finite."
        )
    if not np.isfinite(point):
        raise ValueError(f"{model}: point nowcast for {quarter} is not finite.")
    arr.flags.writeable = False

    lo80, hi80 = np.quantile(arr, [0.10, 0.90])
    lo95, hi95 = np.quantile(arr, [0.025, 0.975])
    return NowcastResult(
        quarter=quarter,
        horizon=int(horizon),
        as_of=as_of,
        model=model,
        target=target,
        point=float(point),
        draws=arr,
        interval_80=(float(lo80), float(hi80)),
        interval_95=(float(lo95), float(hi95)),
        interval_method=interval_method,
        diagnostics=dict(diagnostics or {}),
    )


# ──────────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────────


@runtime_checkable
class Nowcaster(Protocol):
    """Structural protocol every nowcasting model implements.

    As with
    :class:`~private_assets_frequency.core.protocols.SmoothingModel`,
    ``@runtime_checkable`` only checks attribute presence, not signature
    compatibility — it is for diagnostic logging, not validation.

    Notes
    -----
    Implementations must satisfy three invariants, each of which is tested:

    1. **Information-set purity.** ``fit`` and ``predict`` derive everything
       from the ``info`` they are handed — coefficients, scalers, residual
       pools, hyperparameters. Nothing is cached across ``as_of`` dates and no
       state survives a re-``fit``.
    2. **Reproducibility.** Given the same ``info``, ``quarters``, ``n_draws``
       and ``rng`` seed, ``predict`` returns bit-for-bit identical draws.
    3. **Fit/predict agreement.** ``predict`` raises if its ``info`` has a
       different ``as_of`` than the one ``fit`` saw, since using a later
       information set with earlier coefficients is a leak.
    """

    name: str

    def fit(self, info: InformationSet) -> Nowcaster:
        """Estimate everything the model needs from ``info`` alone.

        Parameters
        ----------
        info
            The information set at the nowcast date.

        Returns
        -------
        Nowcaster
            ``self``, fitted.
        """
        ...

    def predict(
        self,
        info: InformationSet,
        quarters: list[pd.Period],
        n_draws: int = 2000,
        rng: np.random.Generator | None = None,
    ) -> list[NowcastResult]:
        """Nowcast each quarter in ``quarters``, with predictive draws.

        Parameters
        ----------
        info
            The same information set ``fit`` was called with.
        quarters
            Target quarters. Multi-quarter horizons are handled recursively, in
            horizon order, regardless of the order given here; results come back
            in the order requested.
        n_draws
            Number of predictive draws per quarter.
        rng
            Generator for the predictive simulation.

        Returns
        -------
        list[NowcastResult]
        """
        ...
