r"""
Shared machinery for nowcasters with a conditional-mean form (private).

Not part of the module spec's file list — an implementation detail that exists
so :mod:`naive`, :mod:`smoothing_regression`, :mod:`structural` and
:mod:`signature` do not each reimplement horizon bookkeeping, the recursive /
direct interval branch, the fit–predict consistency guard, and the diagnostics
payload. Every model that can write its forecast as

.. math::
    \mathbb{E}[s_Q \mid \mathcal{I}] = m\!\left(Q,\ (s_{Q-1}, s_{Q-2}, \dots)\right)

inherits :class:`RecursiveNowcasterBase` and supplies three things: how to fit,
how many lags the conditional mean reads, and the conditional mean itself
(vectorised over draws).

The base class owns the invariants listed on
:class:`~private_assets_frequency.nowcast.base.Nowcaster`, so a subclass cannot
forget them:

* ``predict`` refuses an information set whose ``as_of`` differs from the one
  ``fit`` saw — using later data with earlier coefficients is a leak;
* ``fit`` clears all prior state before estimating, so a refit cannot inherit a
  previous information set's parameters;
* ``target='true'`` never recurses, because the unsmoothed return has no
  dependence on the lagged *reported* value.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ...core.config import FallbackPolicy
from ..base import NowcastResult, NowcastTarget, make_nowcast_result
from ..information_set import InformationSet
from ..uncertainty import (
    DEFAULT_MIN_OOS,
    InnovationSampler,
    ResidualPool,
    empirical_interval,
    point_chain,
    recursive_predictive_draws,
)

__all__ = ["RecursiveNowcasterBase", "UnsupportedTargetError"]

DEFAULT_N_DRAWS = 2000


class UnsupportedTargetError(NotImplementedError):
    """Raised when a model cannot produce the requested nowcast target."""


@dataclass
class _FitState:
    """Everything ``fit`` establishes, cleared and rebuilt on every call."""

    as_of: pd.Timestamp
    n_train: int
    pool: ResidualPool
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RecursiveNowcasterBase:
    """Base class for nowcasters expressible as a conditional mean.

    Subclasses set ``name`` and implement :meth:`_fit_impl`, :meth:`_lag_depth`
    and :meth:`_conditional_mean`. They may override
    :meth:`_true_target_centre` and :meth:`_true_target_pool_scale` to support
    ``target='true'``; the default raises :class:`UnsupportedTargetError`.

    Attributes
    ----------
    name
        Model identifier carried on every :class:`NowcastResult`.
    target
        ``'reported'`` or ``'true'``.
    interval_method
        ``'empirical_recursive'`` or ``'empirical_direct'``.
    min_oos
        Out-of-sample errors required before a horizon's pool is used as an
        empirical law; below it, inflated in-sample residuals stand in.
    fallback_policy
        ``'warn'`` (default), ``'strict'`` or ``'auto'``, following the parent
        library's :class:`~private_assets_frequency.core.config.FallbackPolicy`.
    """

    name: str = "base"
    target: NowcastTarget = "reported"
    interval_method: str = "empirical_recursive"
    min_oos: int = DEFAULT_MIN_OOS
    fallback_policy: FallbackPolicy | str = FallbackPolicy.WARN

    _state: _FitState | None = None

    # ── Subclass contract ──────────────────────────────────────────

    def _fit_impl(self, info: InformationSet) -> tuple[ResidualPool, dict[str, Any]]:
        """Estimate the model from ``info``; return its pool and diagnostics."""
        raise NotImplementedError

    def _lag_depth(self) -> int:
        """Number of lagged reported values the conditional mean reads."""
        raise NotImplementedError

    def _conditional_mean(
        self, info: InformationSet, quarter: pd.Period, history: np.ndarray
    ) -> np.ndarray:
        """Conditional mean of the reported return, vectorised over draws.

        Parameters
        ----------
        info
            The information set (same ``as_of`` as the fit).
        quarter
            Target quarter.
        history
            Shape ``(n, n_lags)``; column ``0`` is ``s_{Q-1}``, column ``1`` is
            ``s_{Q-2}``, and so on.

        Returns
        -------
        np.ndarray
            Shape ``(n,)``.
        """
        raise NotImplementedError

    def _innovation_sampler(
        self, info: InformationSet, horizons: list[int]
    ) -> InnovationSampler:
        """Innovation law added at each step of the recursion.

        Defaults to a resample of the **one-step** out-of-sample error pool, so
        horizon widening comes from the model's own dynamics rather than from
        switching pools (see
        :mod:`private_assets_frequency.nowcast.uncertainty`). A subclass with a
        parametric predictive variance overrides this to return a Gaussian
        sampler instead.

        Parameters
        ----------
        info
            The information set.
        horizons
            The contiguous horizon chain being simulated.

        Returns
        -------
        Callable
            ``(step_horizon, size, rng) -> np.ndarray``.
        """
        assert self._state is not None
        return self._state.pool.sampler(1)

    def _true_target_centre(
        self, info: InformationSet, quarter: pd.Period
    ) -> float:
        r"""Systematic component :math:`\alpha + \beta \cdot F_Q` for ``quarter``."""
        raise UnsupportedTargetError(
            f"{self.name} does not support target='true': it has no factor "
            "decomposition to read α and β from."
        )

    def _true_target_scale(self) -> float:
        """Divisor mapping reduced-form innovations to unsmoothed-return ones."""
        raise UnsupportedTargetError(
            f"{self.name} does not support target='true': it has no factor "
            "decomposition to read α and β from."
        )

    # ── Policy helper ──────────────────────────────────────────────

    @property
    def _policy(self) -> FallbackPolicy:
        return (
            self.fallback_policy
            if isinstance(self.fallback_policy, FallbackPolicy)
            else FallbackPolicy(self.fallback_policy)
        )

    def _flag(self, message: str, category: type[Warning] = UserWarning) -> None:
        """Warn, or raise under :attr:`FallbackPolicy.STRICT`."""
        if self._policy is FallbackPolicy.STRICT:
            raise category(message)
        warnings.warn(message, category, stacklevel=3)

    # ── Public API ─────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has run."""
        return self._state is not None

    @property
    def fit_diagnostics(self) -> dict[str, Any]:
        """Diagnostics from the last :meth:`fit`."""
        if self._state is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        return dict(self._state.diagnostics)

    @property
    def residual_pool(self) -> ResidualPool:
        """The fitted residual pool."""
        if self._state is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        return self._state.pool

    def fit(self, info: InformationSet) -> RecursiveNowcasterBase:
        """Estimate everything from ``info`` alone.

        Parameters
        ----------
        info
            The information set at the nowcast date.

        Returns
        -------
        RecursiveNowcasterBase
            ``self``, fitted.

        Raises
        ------
        TypeError
            If ``info`` is not an :class:`InformationSet`. Models are never
            handed raw frames.
        """
        if not isinstance(info, InformationSet):
            raise TypeError(
                f"{self.name}.fit expects an InformationSet (built by "
                "build_information_set, which truncates at as_of), got "
                f"{type(info).__name__}."
            )
        self._state = None  # no state survives a refit
        pool, diagnostics = self._fit_impl(info)
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "model": self.name,
                "target": self.target,
                "as_of": info.as_of,
                "data_mode": info.data_mode,
                "evaluation_target": info.evaluation_target,
                "interval_method": self.interval_method,
                "residual_pool": pool.diagnostics,
            }
        )
        self._state = _FitState(
            as_of=info.as_of,
            n_train=int(diagnostics.get("n_train", info.n_published)),
            pool=pool,
            diagnostics=diagnostics,
        )
        return self

    def predict(
        self,
        info: InformationSet,
        quarters: list[pd.Period],
        n_draws: int = DEFAULT_N_DRAWS,
        rng: np.random.Generator | None = None,
    ) -> list[NowcastResult]:
        """Nowcast each quarter in ``quarters``.

        Horizons are simulated recursively from ``h=1`` up to the largest
        requested horizon even if intermediate quarters were not asked for,
        because horizon ``h`` conditions on ``h-1``. Results come back in the
        order requested.

        Parameters
        ----------
        info
            The information set :meth:`fit` was called with. Its ``as_of`` must
            match.
        quarters
            Target quarters, as ``pd.Period`` or anything ``pd.Period`` accepts.
        n_draws
            Predictive draws per quarter.
        rng
            Generator; ``None`` gives a fresh unseeded one, so pass a seeded
            generator for reproducibility.

        Returns
        -------
        list[NowcastResult]

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not run.
        ValueError
            If ``info.as_of`` differs from the fit's, if ``quarters`` is empty,
            or if a quarter is already published at ``as_of``.
        """
        state = self._state
        if state is None:
            raise RuntimeError(f"{self.name}: call fit() before predict().")
        if not isinstance(info, InformationSet):
            raise TypeError(
                f"{self.name}.predict expects an InformationSet, got "
                f"{type(info).__name__}."
            )
        if info.as_of != state.as_of:
            raise ValueError(
                f"{self.name}: fitted at as_of={state.as_of.date()} but asked to "
                f"predict from as_of={info.as_of.date()}. Using a later "
                "information set with earlier coefficients leaks data; re-fit on "
                "the information set you intend to predict from."
            )
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}.")
        rng = rng if rng is not None else np.random.default_rng()

        targets = [
            q if isinstance(q, pd.Period) else pd.Period(q, freq="Q")
            for q in quarters
        ]
        if not targets:
            raise ValueError(f"{self.name}: `quarters` is empty.")
        horizons = {q: info.horizon_of(q) for q in targets}

        if self.target == "true":
            return self._predict_true(info, targets, horizons, n_draws, rng)
        return self._predict_reported(info, targets, horizons, n_draws, rng)

    # ── Prediction branches ────────────────────────────────────────

    def _initial_history(self, info: InformationSet) -> np.ndarray:
        """``[s_{Q_last}, s_{Q_last-1}, …]`` to the depth the model needs."""
        depth = self._lag_depth()
        if depth == 0:
            return np.empty(0, dtype=float)
        reported = info.reported
        if reported.size < depth:
            raise ValueError(
                f"{self.name}: needs {depth} lagged reported value(s) but only "
                f"{reported.size} are published at as_of={info.as_of.date()}."
            )
        last = pd.PeriodIndex(reported.index, freq="Q")[-1]
        values = []
        for j in range(depth):
            quarter = last - j
            if quarter not in reported.index:
                raise ValueError(
                    f"{self.name}: lagged reported value for {quarter} is "
                    f"missing at as_of={info.as_of.date()} (gap in the reported "
                    "series); a lag-based nowcast cannot be formed."
                )
            values.append(float(reported[quarter]))
        return np.asarray(values, dtype=float)

    def _predict_reported(
        self,
        info: InformationSet,
        targets: list[pd.Period],
        horizons: dict[pd.Period, int],
        n_draws: int,
        rng: np.random.Generator,
    ) -> list[NowcastResult]:
        state = self._state
        assert state is not None
        max_h = max(horizons.values())
        chain = list(range(1, max_h + 1))
        last_pub = info.last_published_quarter
        assert last_pub is not None
        quarter_at = {h: last_pub + h for h in chain}

        def mean_fn(h: int, history: np.ndarray) -> np.ndarray:
            return self._conditional_mean(info, quarter_at[h], history)

        history = self._initial_history(info)
        points = point_chain(
            horizons=chain, conditional_mean=mean_fn, history=history
        )

        if self.interval_method in ("empirical_recursive", "parametric"):
            draws = recursive_predictive_draws(
                horizons=chain,
                conditional_mean=mean_fn,
                history=history,
                innovation_sampler=self._innovation_sampler(info, chain),
                n_draws=n_draws,
                rng=rng,
            )
        elif self.interval_method == "empirical_direct":
            draws = {}
            for h in chain:
                if not state.pool.has(h):
                    raise ValueError(
                        f"{self.name}: interval_method='empirical_direct' needs a "
                        f"residual pool at horizon {h}, which the rolling-origin "
                        "backtest did not produce. Use "
                        "interval_method='empirical_recursive', which needs only "
                        "the one-step pool."
                    )
                draws[h] = points[h] + state.pool.sample(h, n_draws, rng)
        else:
            raise ValueError(
                f"{self.name}: unknown interval_method "
                f"{self.interval_method!r}; expected 'empirical_recursive', "
                "'empirical_direct' or 'parametric'."
            )

        suffix = state.pool.method_suffix(chain)
        results: dict[pd.Period, NowcastResult] = {}
        for h in chain:
            quarter = quarter_at[h]
            if quarter not in horizons:
                continue  # simulated only to reach a later horizon
            diagnostics = dict(state.diagnostics)
            diagnostics.update(
                {
                    "horizon": h,
                    "n_oos_errors": state.pool.n_oos.get(h),
                    "residual_source": state.pool.source.get(h),
                    # The innovation pool is deliberately *not* centred, so the
                    # predictive draws sit `pool_mean` away from `point` — the
                    # model's own historical bias, carried into the interval.
                    "pool_mean": (
                        float(np.mean(state.pool.errors[1]))
                        if state.pool.has(1)
                        else None
                    ),
                    "innovation_horizon": (
                        h if self.interval_method == "empirical_direct" else 1
                    ),
                    "factor_quarter_complete": bool(
                        info.factor_quarter_complete.get(quarter, False)
                    ),
                }
            )
            # Cross-check: a materially wider direct interval signals that the
            # recursion's i.i.d. one-step assumption is missing something.
            if self.interval_method != "empirical_direct" and state.pool.has(h):
                direct = points[h] + state.pool.sample(h, n_draws, rng)
                diagnostics["interval_80_direct"] = empirical_interval(direct, 0.80)
            results[quarter] = make_nowcast_result(
                quarter=quarter,
                horizon=h,
                as_of=info.as_of,
                model=self.name,
                target=self.target,
                point=points[h],
                draws=draws[h],
                interval_method=self.interval_method + suffix,
                diagnostics=diagnostics,
            )
        return [results[q] for q in targets]

    def _predict_true(
        self,
        info: InformationSet,
        targets: list[pd.Period],
        horizons: dict[pd.Period, int],
        n_draws: int,
        rng: np.random.Generator,
    ) -> list[NowcastResult]:
        r"""Nowcast the unsmoothed return — a wrapper over the factor model.

        No recursion: :math:`r_Q = \alpha + \beta F_Q + \varepsilon_Q` does not
        depend on the lagged *reported* value, and :math:`F_Q` is already known
        for a closed quarter. The predictive distribution is therefore the same
        width at every horizon, and that width is the idiosyncratic volatility
        :math:`\sigma_\varepsilon` — irreducible with more public data.
        """
        state = self._state
        assert state is not None
        scale = self._true_target_scale()
        results: list[NowcastResult] = []
        for quarter in targets:
            centre = self._true_target_centre(info, quarter)
            innovation = state.pool.sample(1, n_draws, rng) / scale
            diagnostics = dict(state.diagnostics)
            diagnostics.update(
                {
                    "horizon": horizons[quarter],
                    "true_target_scale": scale,
                    "residual_source": state.pool.source.get(1),
                    "n_oos_errors": state.pool.n_oos.get(1),
                    "interpretation": (
                        "target='true' is not a nowcast in the same sense as "
                        "target='reported': the predictable part is the "
                        "systematic component alpha + beta*F_Q and the "
                        "idiosyncratic part has expectation zero. The interval "
                        "is dominated by idiosyncratic risk and does not narrow "
                        "with more public data, nor with horizon."
                    ),
                }
            )
            results.append(
                make_nowcast_result(
                    quarter=quarter,
                    horizon=horizons[quarter],
                    as_of=info.as_of,
                    model=self.name,
                    target="true",
                    point=centre,
                    draws=centre + innovation,
                    interval_method="empirical_idiosyncratic"
                    + state.pool.method_suffix([1]),
                    diagnostics=diagnostics,
                )
            )
        return results
