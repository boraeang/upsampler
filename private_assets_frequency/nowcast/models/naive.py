r"""
Model 0 — ``NaiveCarryForward``: the bar every other nowcaster must clear.

The nowcast for every unpublished quarter is the last published quarterly
return:

.. math::
    \hat s_{Q+h} = s_{Q}, \qquad h = 1, 2, \dots

This is not a strawman. Reported private markets returns are heavily
autocorrelated by construction — appraisal smoothing *is* an AR(1) filter with
:math:`\lambda` often in 0.5–0.8 — so carrying the last print forward is a
genuinely strong predictor of the next one, and a factor-based nowcaster that
cannot beat it has demonstrated nothing. The literature on forecast evaluation
is emphatic on this point: report the naive benchmark, and report it honestly.

Intervals
---------
The model has no dynamics to propagate, so its default is
``interval_method='empirical_direct'`` — the predictive distribution at horizon
``h`` is the point nowcast plus a resample of the out-of-sample errors at that
same horizon. For a positively autocorrelated series
:math:`\mathrm{Var}(s_{Q+2} - s_Q) > \mathrm{Var}(s_{Q+1} - s_Q)`, so the ``h=2``
interval is wider than ``h=1`` — not because the model says so, but because the
historical errors do.

``'empirical_recursive'`` is also available and treats the reported series as a
random walk, giving :math:`h\sigma^2` growth. That is the model's own internal
assumption rather than an empirical fact, which is why it is not the default
here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ...core.config import FallbackPolicy
from ..information_set import InformationSet
from ..uncertainty import (
    DEFAULT_MIN_OOS,
    ResidualPool,
    build_residual_pool,
    rolling_origin_backtest,
)
from ._base import RecursiveNowcasterBase

__all__ = ["NaiveCarryForward"]


@dataclass
class NaiveCarryForward(RecursiveNowcasterBase):
    """Carry the last published quarterly return forward.

    Parameters
    ----------
    horizons
        Horizons to build residual pools for during :meth:`fit`. Must cover
        every horizon :meth:`predict` will be asked for under
        ``interval_method='empirical_direct'``.
    interval_method
        ``'empirical_direct'`` (default) or ``'empirical_recursive'`` — see the
        module docstring.
    min_train
        Smallest training window in the internal rolling-origin backtest that
        produces the error pools. For carry-forward this only needs to be large
        enough to be representative, not to estimate anything.
    min_oos
        Out-of-sample errors required before a horizon's pool is used as an
        empirical law.
    fallback_policy
        ``'warn'`` (default), ``'strict'`` or ``'auto'``.

    Attributes
    ----------
    name
        ``'naive_carry_forward'``.

    Examples
    --------
    >>> model = NaiveCarryForward().fit(info)            # doctest: +SKIP
    >>> results = model.predict(info, info.unpublished_quarters(),
    ...                         rng=np.random.default_rng(0))  # doctest: +SKIP
    >>> results[0].point == info.reported.iloc[-1]       # doctest: +SKIP
    True
    """

    horizons: tuple[int, ...] = (1, 2)
    interval_method: str = "empirical_direct"
    min_train: int = 12
    min_oos: int = DEFAULT_MIN_OOS
    fallback_policy: FallbackPolicy | str = FallbackPolicy.WARN
    name: str = field(default="naive_carry_forward", init=False)

    # ── Subclass contract ──────────────────────────────────────────

    def _lag_depth(self) -> int:
        return 1

    def _conditional_mean(
        self, info: InformationSet, quarter: pd.Period, history: np.ndarray
    ) -> np.ndarray:
        """The most recent value, carried forward unchanged."""
        return np.asarray(history, dtype=float)[:, 0]

    def _fit_impl(
        self, info: InformationSet
    ) -> tuple[ResidualPool, dict[str, Any]]:
        """Build the out-of-sample error pools; there is nothing to estimate.

        The "backtest" replays the rule over the published history: at every
        origin, predict the next ``h`` quarters as the last published value and
        record the error. For carry-forward this is closed form, but it runs
        through the same :func:`rolling_origin_backtest` used by the estimated
        models so the pools are constructed identically and are comparable
        across models in the evaluation report.
        """
        y = info.reported.to_numpy(dtype=float)
        if y.size < 2:
            raise ValueError(
                f"{self.name}: needs at least 2 published quarters at "
                f"as_of={info.as_of.date()}, got {y.size}."
            )
        horizons = tuple(sorted(set(int(h) for h in self.horizons)))
        if min(horizons) < 1:
            raise ValueError(f"horizons must all be >= 1, got {self.horizons!r}.")

        def predict_fn(n_train: int, hs: Sequence[int]) -> dict[int, float]:
            last = float(y[n_train - 1])
            return {h: last for h in hs}

        min_train = max(1, min(self.min_train, y.size - 1))
        oos, backtest_diag = rolling_origin_backtest(
            y,
            min_train=min_train,
            horizons=horizons,
            predict_fn=predict_fn,
        )
        # "In-sample residuals" of a carry-forward rule are its one-step
        # differences; there are no fitted parameters, so p = 0 and the
        # sqrt(1 + p/n) inflation is exactly 1.
        in_sample = np.diff(y)
        pool = build_residual_pool(
            oos,
            in_sample,
            n_params=0,
            horizons=horizons,
            min_oos=self.min_oos,
            warn=self._policy is not FallbackPolicy.AUTO,
            label=self.name,
        )
        diagnostics: dict[str, Any] = {
            "n_train": int(y.size),
            "n_params": 0,
            "features": ["s_lag1"],
            "horizons": horizons,
            "min_train": min_train,
            "backtest": backtest_diag,
            "last_published_quarter": str(info.last_published_quarter),
            "carry_forward_value": float(y[-1]),
            "feature_budget": {
                "n_features": 1,
                "n_train": int(y.size),
                "budget": float(y.size) / 5.0,
                "within_budget": True,
            },
        }
        return pool, diagnostics
