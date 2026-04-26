"""
FallbackPolicy execution helper.

The :class:`FallbackPolicy` enum lives in :mod:`core.config` so it can be a
field of :class:`AssetClassConfig` without creating a circular import. This
module provides the *execution* logic: a small handler that consumes
warnings / stage validation results and acts on them according to the
configured policy.

Policies (per Stage 1 spec point "Fallback Mechanisms"):

    * ``STRICT`` — any warning escalates to ``PipelineFailure``.
    * ``WARN``   — warnings emit via :mod:`warnings` and accumulate on the
      pipeline result.
    * ``AUTO``   — automatic mitigations are applied silently, and only
      surface as messages on the result. Stages have to opt in to a
      specific auto-fix; otherwise we degrade to ``WARN`` semantics.

The handler is deliberately stateless beyond accumulation — concrete
auto-fix routines live in the per-stage modules (e.g. fall back to
:class:`GeltnerClassicSmoother` when the Bayesian fit is un-identified).
This module just provides the policy decision and a uniform recording API.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable

from ..core.config import FallbackPolicy
from ..core.protocols import StageValidationResult, ValidationStatus


class PipelineFailure(RuntimeError):
    """Raised in ``FallbackPolicy.STRICT`` mode when any warning is recorded."""


@dataclass
class FallbackHandler:
    """Stateful collector that decides what to do with stage warnings.

    Attributes
    ----------
    policy
        The active :class:`FallbackPolicy`.
    accumulated_warnings
        Free-form warning messages recorded across stages.
    accumulated_validations
        Inter-stage :class:`StageValidationResult` objects recorded.
    """

    policy: FallbackPolicy = FallbackPolicy.WARN
    accumulated_warnings: list[str] = field(default_factory=list)
    accumulated_validations: list[StageValidationResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.policy = (
            FallbackPolicy(self.policy)
            if not isinstance(self.policy, FallbackPolicy)
            else self.policy
        )

    # ── Warning ingest ──────────────────────────────────────────────

    def warn(self, message: str, *, source: str | None = None) -> None:
        """Record a free-form warning under the active policy.

        STRICT escalates to :class:`PipelineFailure`; WARN and AUTO simply
        accumulate the message on this handler. Callers expose the
        accumulated list via :attr:`PipelineResult.warnings`; deliberately
        *not* emitting to :mod:`warnings` so that downstream test
        configurations with ``filterwarnings='error'`` don't masquerade
        warnings as exceptions.
        """
        prefixed = f"[{source}] {message}" if source else message
        self.accumulated_warnings.append(prefixed)
        if self.policy is FallbackPolicy.STRICT:
            raise PipelineFailure(prefixed)
        # WARN / AUTO: silent accumulation. Callers may emit via warnings.warn
        # explicitly if they want to integrate with that infrastructure.

    def ingest_warnings(
        self,
        warnings_iter: Iterable[str],
        *,
        source: str | None = None,
    ) -> None:
        """Record an iterable of warnings (e.g. ``diagnostics['warnings']``)."""
        for w in warnings_iter:
            self.warn(w, source=source)

    # ── Stage validation ingest ─────────────────────────────────────

    def record_stage_validation(self, result: StageValidationResult) -> None:
        """Record a :class:`StageValidationResult` and act on its status."""
        self.accumulated_validations.append(result)
        if result.status is ValidationStatus.PASS:
            return
        for msg in result.messages:
            self.warn(msg, source=result.stage)
        if result.status is ValidationStatus.FAIL and self.policy is FallbackPolicy.STRICT:
            raise PipelineFailure(
                f"{result.stage}: validation failed — {result.messages!r}"
            )

    # ── Auto-fix gate ──────────────────────────────────────────────

    def auto_enabled(self) -> bool:
        """``True`` iff the policy is ``AUTO`` (caller may apply mitigations silently)."""
        return self.policy is FallbackPolicy.AUTO


__all__ = ["FallbackHandler", "FallbackPolicy", "PipelineFailure"]
