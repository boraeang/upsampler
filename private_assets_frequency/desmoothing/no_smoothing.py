"""
Pass-through smoothing model.

Used for asset classes where the underlying instruments are exchange-traded
and the index needs no desmoothing — managed-futures / CTA strategies are the
prototypical case (Stage 1 spec, Model 4). Also useful as a baseline
comparator: running the rest of the pipeline with this model in place of a
real desmoother answers the question "what would the disaggregation look
like if we ignored smoothing entirely?"

The model implements the :class:`SmoothingModel` protocol but every method
is essentially a no-op. ``smoothing_params`` is an empty array and the
``true_returns`` field of the result is exactly the observed series.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.protocols import DesmoothedResult


class NoSmoothing:
    """Pass-through implementation of :class:`SmoothingModel`.

    The class carries no state and is safe to share across calls.
    """

    def fit(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
        priors: dict[str, Any],
    ) -> DesmoothedResult:
        """Return ``observed_returns`` unchanged, packaged as a :class:`DesmoothedResult`."""
        if not isinstance(observed_returns, pd.Series):
            raise TypeError(
                f"observed_returns must be a Series, got {type(observed_returns).__name__}"
            )
        true_returns = observed_returns.copy()
        diagnostics: dict[str, Any] = {
            "method": "no_smoothing",
            "n_observations": int(len(observed_returns)),
            "warnings": [],
        }
        return DesmoothedResult(
            true_returns=true_returns,
            smoothing_params={},
            posterior_summary={},
            diagnostics=diagnostics,
        )

    def log_likelihood(
        self,
        smoothing_params: np.ndarray,
        observed_returns: np.ndarray,
        factor_returns: np.ndarray,
    ) -> float:
        """No-op: returns 0 (no smoothing parameters → constant likelihood)."""
        return 0.0

    def log_prior(
        self,
        smoothing_params: np.ndarray,
        prior_config: dict[str, Any],
    ) -> float:
        """No-op: returns 0 (no prior over an empty parameter set)."""
        return 0.0

    def desmooth(
        self,
        observed_returns: np.ndarray,
        smoothing_params: np.ndarray,
    ) -> np.ndarray:
        """Identity transform: returns a copy of ``observed_returns``."""
        return np.asarray(observed_returns, dtype=float).copy()


__all__ = ["NoSmoothing"]
