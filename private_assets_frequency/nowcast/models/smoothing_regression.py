r"""
Model 1 — ``SmoothingRegressionNowcaster``: the joint smoothing regression.

This is the **benchmark, not a strawman**. It is the reduced form of the
library's own AR(1) and Rudin smoothing models, and it is what a careful
practitioner would actually run:

.. math::
    s_Q = a + \sum_k b_k F_{k,Q} + \sum_{j=1}^{q} c_j\, s_{Q-j} + e_Q

with :math:`s_Q` the reported quarterly return and :math:`F_{k,Q}` the public
factor return over quarter :math:`Q`, compounded from the high-frequency data as
:math:`\prod(1+r) - 1`.

Why it is the reduced form
--------------------------
Substituting the factor model :math:`r_Q = \alpha + \beta F_Q + \varepsilon_Q`
into the AR(1) appraisal filter :math:`s_Q = (1-\lambda) r_Q + \lambda s_{Q-1}`:

.. math::
    s_Q = \underbrace{(1-\lambda)\alpha}_{a}
        + \underbrace{(1-\lambda)\beta}_{b} F_Q
        + \underbrace{\lambda}_{c_1} s_{Q-1}
        + \underbrace{(1-\lambda)\varepsilon_Q}_{e_Q}

so a single OLS on observables recovers :math:`b = (1-\lambda)\beta` and
:math:`c_1 = \lambda` without ever estimating :math:`\lambda` separately. For
:math:`q \ge 2` the same algebra is the Rudin-Mao-Zhang-Fink Eq. 4 form, with
:math:`\theta_0 = 1 - \sum_j c_j` — see
:mod:`private_assets_frequency.desmoothing.rudin_reparam`, whose coefficient
unwinding this module reuses for ``target='true'``.

The reference paper's Theorem 1 says signature regression contains this
regression as a special case. That is a statement about representation, not
about out-of-sample accuracy at ~90 observations — which is why this model is
the thing to beat and :mod:`..models.signature` is the challenger.

Estimation
----------
OLS by default. Optional ridge penalises the **factor** coefficients only —
never the intercept, and never the lagged-return coefficients, because
:math:`c_j` is the smoothing parameter and shrinking it toward zero would
quietly undo the thing being modelled.

``q`` and the ridge strength are selected by **inner rolling-origin
validation** over published quarters only, never in sample. Because an
:class:`~private_assets_frequency.nowcast.information_set.InformationSet`
contains no unpublished quarter, the outer evaluation quarter cannot enter the
selection by construction; :meth:`SmoothingRegressionNowcaster.fit` asserts this
rather than trusting it.

References
----------
.. [1] Geltner (1993) — "Estimating Market Values from Appraised Values without
       Assuming an Efficient Market."
.. [2] Rudin, Mao, Zhang & Fink (2019) — "Fitting Private Equity into the Total
       Portfolio Framework," Eq. 3–4.
.. [3] Cohen, Mantoan, Nesheim, de Paula, Turrell & Yang (2023) —
       "Nowcasting using regression on signatures," arXiv:2305.10256v3,
       Theorem 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import linalg

from ...core.config import FallbackPolicy
from ..base import NowcastTarget
from ..information_set import InformationSet
from ..uncertainty import (
    DEFAULT_MIN_OOS,
    InnovationSampler,
    ResidualPool,
    build_residual_pool,
    rolling_origin_backtest,
)
from ._base import RecursiveNowcasterBase

__all__ = [
    "SmoothingRegressionNowcaster",
    "SmoothingRegressionWarning",
    "unwind_to_structural",
]

_THETA0_FLOOR = 0.10
"""Below this, ``β = b/θ₀`` is numerically unstable — matches ``rudin_reparam``."""


class SmoothingRegressionWarning(UserWarning):
    """Emitted on a suspect smoothing regression fit."""


# ──────────────────────────────────────────────────────────────────
# Design matrix
# ──────────────────────────────────────────────────────────────────


class _Design(NamedTuple):
    """Design matrix split into exogenous and lagged-dependent blocks.

    The split exists so the rolling-origin backtest can substitute *predicted*
    lagged values for unknown ones at horizons ``>= 2`` while leaving the
    observable factor columns alone.
    """

    index: pd.PeriodIndex
    y: np.ndarray  # (n,)
    exog: np.ndarray  # (n, k_exog) — const, factors [, lagged factors]
    lags: np.ndarray  # (n, q) — s_{Q-1} … s_{Q-q}
    exog_names: list[str]
    lag_names: list[str]
    factor_positions: np.ndarray  # indices into the full column vector

    @property
    def n_params(self) -> int:
        return self.exog.shape[1] + self.lags.shape[1]

    @property
    def names(self) -> list[str]:
        return list(self.exog_names) + list(self.lag_names)

    def full(self) -> np.ndarray:
        """``[exog | lags]`` as one matrix."""
        return np.hstack([self.exog, self.lags])


def _build_design(
    info: InformationSet,
    *,
    n_lags: int,
    include_lagged_factors: bool,
) -> _Design:
    r"""Assemble the regression rows from published quarters only.

    Lags are formed on a *contiguous* quarterly grid rather than by positional
    shift of the published index, so an interior gap in the reported series drops
    the adjacent rows instead of silently pairing non-adjacent quarters.

    Parameters
    ----------
    info
        Information set. Only ``info.reported`` (published values) and complete
        quarterly factor aggregates are read.
    n_lags
        ``q`` — number of lagged reported-return terms.
    include_lagged_factors
        Also include :math:`F_{k,Q-1}`.

    Returns
    -------
    _Design

    Raises
    ------
    ValueError
        If fewer rows survive than there are parameters to estimate.
    """
    s = info.reported.astype(float)
    published = pd.PeriodIndex(s.index, freq="Q")
    grid = pd.period_range(published[0], published[-1], freq="Q")

    s_grid = s.reindex(grid)
    cols = list(info.factor_columns)
    complete = info.factor_quarter_complete.reindex(grid).fillna(False).to_numpy()
    # Broadcast the per-quarter completeness flag across every factor column; a
    # bare (n, 1) mask raises against a multi-column frame.
    mask = pd.DataFrame(
        np.repeat(complete[:, None], len(cols), axis=1), index=grid, columns=cols
    )
    f_grid = info.factors_quarterly.reindex(grid)[cols].where(mask)

    frame = pd.DataFrame(index=grid)
    frame["const"] = 1.0
    for c in cols:
        frame[c] = f_grid[c]
    exog_names = ["const"] + list(cols)
    if include_lagged_factors:
        for c in cols:
            name = f"{c}_lag1"
            frame[name] = f_grid[c].shift(1)
            exog_names.append(name)
    lag_names = [f"s_lag{j}" for j in range(1, n_lags + 1)]
    for j in range(1, n_lags + 1):
        frame[f"s_lag{j}"] = s_grid.shift(j)

    frame["__y__"] = s_grid
    keep = frame.notna().all(axis=1) & frame.index.isin(published)
    frame = frame.loc[keep]

    n_params = len(exog_names) + n_lags
    if len(frame) <= n_params:
        raise ValueError(
            f"smoothing regression design has {len(frame)} usable rows for "
            f"{n_params} parameters at as_of={info.as_of.date()}; need more "
            "published history (or fewer lags / factors)."
        )

    factor_positions = np.asarray(
        [exog_names.index(c) for c in exog_names if c != "const"], dtype=int
    )
    return _Design(
        index=pd.PeriodIndex(frame.index, freq="Q"),
        y=frame["__y__"].to_numpy(dtype=float),
        exog=frame[exog_names].to_numpy(dtype=float),
        lags=frame[lag_names].to_numpy(dtype=float).reshape(len(frame), n_lags),
        exog_names=exog_names,
        lag_names=lag_names,
        factor_positions=factor_positions,
    )


# ──────────────────────────────────────────────────────────────────
# Estimation
# ──────────────────────────────────────────────────────────────────


class _Fit(NamedTuple):
    """A single ridge/OLS solution."""

    coef: np.ndarray
    xtx_inv: np.ndarray | None
    sigma2: float
    residuals: np.ndarray
    n_obs: int
    n_params: int
    condition_number: float


def _solve(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge_alpha: float,
    penalised: np.ndarray,
) -> _Fit:
    r"""Solve :math:`(X'X + \Lambda)\hat\beta = X'y` with a selective penalty.

    Parameters
    ----------
    X
        Design matrix, shape ``(n, p)``.
    y
        Response, shape ``(n,)``.
    ridge_alpha
        Ridge strength. ``0`` gives OLS.
    penalised
        Column indices the penalty applies to — the factor columns. The
        intercept and the lagged-return coefficients are left unpenalised: the
        latter *are* the smoothing parameter, and shrinking them toward zero
        would undo the model.

    Returns
    -------
    _Fit

    Notes
    -----
    Solved through the normal equations with a Cholesky factorisation, falling
    back to :func:`numpy.linalg.lstsq` (a pseudo-inverse) when the system is not
    positive definite. The fallback keeps a collinear factor panel from raising,
    which matters because the rolling-origin backtest refits on short windows
    where collinearity is common.
    """
    n, p = X.shape
    gram = X.T @ X
    if ridge_alpha > 0 and penalised.size:
        gram = gram.copy()
        gram[penalised, penalised] += float(ridge_alpha)
    rhs = X.T @ y
    cond = float(np.linalg.cond(gram)) if p else 1.0
    try:
        coef = linalg.solve(gram, rhs, assume_a="pos")
        xtx_inv = linalg.inv(gram)
    except (linalg.LinAlgError, np.linalg.LinAlgError, ValueError):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        xtx_inv = np.linalg.pinv(gram)
    residuals = y - X @ coef
    dof = max(n - p, 1)
    sigma2 = float(residuals @ residuals / dof)
    return _Fit(
        coef=np.asarray(coef, dtype=float),
        xtx_inv=np.asarray(xtx_inv, dtype=float),
        sigma2=sigma2,
        residuals=residuals,
        n_obs=int(n),
        n_params=int(p),
        condition_number=cond,
    )


def _recursive_backtest_predict(
    design: _Design,
    *,
    ridge_alpha: float,
    n_train: int,
    horizons: Sequence[int],
) -> dict[int, float]:
    """Fit on rows ``[0, n_train)`` and nowcast forward recursively.

    At horizon 1 every regressor is observable. At horizon ``h >= 2`` the
    ``s_lag`` columns reach into quarters that were themselves unpublished at the
    origin, so the predicted value is substituted — the same recursion
    :meth:`RecursiveNowcasterBase.predict` performs, which is what makes the
    resulting error pools comparable to live nowcast errors.
    """
    X_train = design.full()[:n_train]
    fit = _solve(
        X_train,
        design.y[:n_train],
        ridge_alpha=ridge_alpha,
        penalised=design.factor_positions,
    )
    k_exog = design.exog.shape[1]
    q = design.lags.shape[1]
    beta_exog = fit.coef[:k_exog]
    beta_lag = fit.coef[k_exog:]

    history = design.lags[n_train].copy() if q else np.empty(0)
    out: dict[int, float] = {}
    for h in sorted(horizons):
        row = n_train + h - 1
        if row >= design.y.size:
            break
        pred = float(design.exog[row] @ beta_exog)
        if q:
            pred += float(history @ beta_lag)
        out[h] = pred
        if q:
            history = np.concatenate([[pred], history[: q - 1]])
    return out


def unwind_to_structural(
    coef: np.ndarray,
    names: list[str],
    factor_columns: list[str],
) -> dict[str, Any]:
    r"""Recover :math:`(\theta_0, \alpha, \beta)` from reduced-form coefficients.

    Applies the Rudin-Mao-Zhang-Fink Eq. 4 unwinding, which generalises the
    AR(1) case:

    .. math::
        \theta_0 = 1 - \sum_{j=1}^{q} c_j, \qquad
        \beta_k = \frac{b_k}{\theta_0}, \qquad
        \alpha = \frac{a}{\theta_0}

    For ``q = 1`` this is exactly the AR(1) mapping, since
    :math:`\theta_0 = 1 - c_1 = 1 - \lambda` and
    :math:`\beta = b/(1-\lambda)`.

    Parameters
    ----------
    coef
        Reduced-form coefficients.
    names
        Coefficient names, aligned with ``coef``; lag coefficients are those
        named ``s_lag*``.
    factor_columns
        Names whose unwound value is reported as a factor beta.

    Returns
    -------
    dict
        ``{'theta_0', 'alpha', 'beta', 'lambda_implied', 'theta', 'stable'}``.
        ``lambda_implied`` is :math:`1 - \theta_0`, which equals :math:`\lambda`
        under AR(1) smoothing and is the total smoothing weight on past reports
        more generally.

    Notes
    -----
    :math:`\beta = b/\theta_0` blows up as :math:`\theta_0 \to 0` — the regime
    where the reported series is almost entirely a function of its own past.
    ``stable`` is ``False`` below ``θ₀ = 0.10``, the same floor
    :mod:`private_assets_frequency.desmoothing.rudin_reparam` applies.
    """
    series = pd.Series(np.asarray(coef, dtype=float), index=list(names))
    lag_names = [n for n in series.index if n.startswith("s_lag")]
    theta_high = series[lag_names].to_numpy() if lag_names else np.empty(0)
    theta_0 = float(1.0 - theta_high.sum())
    stable = abs(theta_0) >= _THETA0_FLOOR
    divisor = theta_0 if stable else np.nan
    beta = {
        c: float(series[c] / divisor) for c in factor_columns if c in series.index
    }
    other = {
        n: float(series[n] / divisor)
        for n in series.index
        if n not in lag_names and n != "const" and n not in factor_columns
    }
    beta.update(other)
    return {
        "theta_0": theta_0,
        "theta": np.concatenate([[theta_0], theta_high]),
        "alpha": float(series.get("const", np.nan) / divisor),
        "beta": beta,
        "lambda_implied": float(1.0 - theta_0),
        "stable": bool(stable),
    }


# ──────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────


@dataclass
class SmoothingRegressionNowcaster(RecursiveNowcasterBase):
    r"""Nowcast the reported return by the joint smoothing regression.

    Parameters
    ----------
    n_lags
        ``q``, the number of lagged reported-return terms. ``None`` (default)
        selects from ``n_lags_candidates`` by inner rolling-origin validation.
        An explicit value skips selection.
    n_lags_candidates
        Candidates for ``q``. Default ``(1, 2)`` per the module spec; ``1`` is
        the AR(1) reduced form.
    ridge_alpha
        Ridge strength on the factor coefficients. ``None`` selects from
        ``ridge_alpha_candidates``; ``0.0`` (the default) is plain OLS.
    ridge_alpha_candidates
        Candidates for the ridge strength when ``ridge_alpha is None``.
    include_lagged_factors
        Include :math:`F_{k,Q-1}` terms. Off by default — they add one parameter
        per factor for a signal the smoothing filter largely already carries.
    target
        ``'reported'`` (default) or ``'true'``; see :mod:`..base`.
    interval_method
        ``'empirical_recursive'`` (default), ``'empirical_direct'`` or
        ``'parametric'``.
    horizons
        Horizons to build residual pools for.
    min_train
        Smallest training window in the internal rolling-origin backtest that
        builds the residual pool. Raised automatically to ``n_params + 5`` if
        that is larger. The default of 40 is the module spec's minimum initial
        training window, reused here deliberately: pooling errors from very
        short windows mixes estimation-error regimes and inflates the pool — at
        ``min_train=24`` on the linear DGP the pooled error sd runs ~20 % above
        the true one-step sd, so the intervals over-cover. Forty quarters puts
        the pool's errors at a training size representative of the live fit.
    inner_validation_quarters
        Number of most-recent published quarters forming the inner selection
        split. The outer evaluation quarter cannot appear here — it is not
        published at ``as_of``, so it is not in the information set at all.
    selection_horizons
        Horizons whose RMSE drives hyperparameter selection. ``(1,)`` by
        default: the one-step error is the most stable criterion at this sample
        size.
    min_oos
        Out-of-sample errors required before a horizon's pool is used directly.
    fallback_policy
        ``'warn'`` (default), ``'strict'`` or ``'auto'``.

    Attributes
    ----------
    name
        ``'smoothing_regression'``.

    Examples
    --------
    >>> model = SmoothingRegressionNowcaster(n_lags=1).fit(info)   # doctest: +SKIP
    >>> model.coefficients                                        # doctest: +SKIP
    const            0.0021
    equity_market    0.4305
    s_lag1           0.6117
    dtype: float64
    """

    n_lags: int | None = None
    n_lags_candidates: tuple[int, ...] = (1, 2)
    ridge_alpha: float | None = 0.0
    ridge_alpha_candidates: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)
    include_lagged_factors: bool = False
    target: NowcastTarget = "reported"
    interval_method: str = "empirical_recursive"
    horizons: tuple[int, ...] = (1, 2)
    min_train: int = 40
    inner_validation_quarters: int = 12
    selection_horizons: tuple[int, ...] = (1,)
    min_oos: int = DEFAULT_MIN_OOS
    fallback_policy: FallbackPolicy | str = FallbackPolicy.WARN
    name: str = field(default="smoothing_regression", init=False)

    _design: _Design | None = field(default=None, init=False, repr=False)
    _fit: _Fit | None = field(default=None, init=False, repr=False)
    _selected_lags: int | None = field(default=None, init=False, repr=False)
    _selected_alpha: float | None = field(default=None, init=False, repr=False)
    _structural: dict[str, Any] | None = field(default=None, init=False, repr=False)

    # ── Fitted-parameter accessors ─────────────────────────────────

    @property
    def coefficients(self) -> pd.Series:
        """Reduced-form coefficients :math:`(a, b_k, c_j)`."""
        if self._fit is None or self._design is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        return pd.Series(self._fit.coef, index=self._design.names, name="coef")

    @property
    def structural_params(self) -> dict[str, Any]:
        r"""Unwound :math:`(\theta_0, \alpha, \beta, \lambda)` — see
        :func:`unwind_to_structural`."""
        if self._structural is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        return dict(self._structural)

    @property
    def sigma(self) -> float:
        """Residual standard deviation of the reduced-form regression."""
        if self._fit is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        return float(np.sqrt(self._fit.sigma2))

    # ── Subclass contract ──────────────────────────────────────────

    def _lag_depth(self) -> int:
        if self._selected_lags is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        return int(self._selected_lags)

    def _exog_row(self, info: InformationSet, quarter: pd.Period) -> np.ndarray:
        """Observable regressors for ``quarter``: constant, factors, lagged factors."""
        assert self._design is not None
        values = [1.0]
        factors = info.factor_row(quarter, require_complete=True)
        cols = list(info.factor_columns)
        values.extend(float(factors[c]) for c in cols)
        if self.include_lagged_factors:
            prev = info.factor_row(quarter - 1, require_complete=True)
            values.extend(float(prev[c]) for c in cols)
        row = np.asarray(values, dtype=float)
        if row.size != self._design.exog.shape[1]:
            raise ValueError(
                f"{self.name}: built {row.size} exogenous regressors for "
                f"{quarter} but the design has {self._design.exog.shape[1]}."
            )
        return row

    def _conditional_mean(
        self, info: InformationSet, quarter: pd.Period, history: np.ndarray
    ) -> np.ndarray:
        assert self._fit is not None and self._design is not None
        k_exog = self._design.exog.shape[1]
        beta_exog = self._fit.coef[:k_exog]
        beta_lag = self._fit.coef[k_exog:]
        hist = np.asarray(history, dtype=float)
        mean = float(self._exog_row(info, quarter) @ beta_exog)
        if beta_lag.size:
            return mean + hist @ beta_lag
        return np.full(hist.shape[0], mean)

    def _true_target_centre(
        self, info: InformationSet, quarter: pd.Period
    ) -> float:
        r"""Systematic part of the unsmoothed return, :math:`\alpha + \beta F_Q`."""
        assert self._structural is not None
        if not self._structural["stable"]:
            raise ValueError(
                f"{self.name}: target='true' requires unwinding β = b/θ₀, but "
                f"θ₀ = {self._structural['theta_0']:.4f} is below the stability "
                f"floor {_THETA0_FLOOR}. The reported series is almost entirely "
                "a function of its own past, so the factor loadings are not "
                "identified; nowcast target='reported' instead."
            )
        centre = float(self._structural["alpha"])
        factors = info.factor_row(quarter, require_complete=True)
        for col, beta in self._structural["beta"].items():
            if col.endswith("_lag1"):
                base = col[: -len("_lag1")]
                centre += beta * float(
                    info.factor_row(quarter - 1, require_complete=True)[base]
                )
            else:
                centre += beta * float(factors[col])
        return centre

    def _true_target_scale(self) -> float:
        r""":math:`\theta_0`, since :math:`e_Q = \theta_0 \varepsilon_Q`."""
        assert self._structural is not None
        if not self._structural["stable"]:
            raise ValueError(
                f"{self.name}: θ₀ = {self._structural['theta_0']:.4f} is below "
                f"the stability floor {_THETA0_FLOOR}; the idiosyncratic scale "
                "e/θ₀ is not meaningful."
            )
        return float(self._structural["theta_0"])

    def _innovation_sampler(
        self, info: InformationSet, horizons: list[int]
    ) -> InnovationSampler:
        """Gaussian prediction variance under ``interval_method='parametric'``."""
        if self.interval_method != "parametric":
            return super()._innovation_sampler(info, horizons)
        assert self._fit is not None and self._design is not None
        last_pub = info.last_published_quarter
        assert last_pub is not None

        # Leverage of the h=1 design row, x'(X'X + Λ)⁻¹x. Reused for every step:
        # the regressor row changes with the horizon only through observable
        # factors, whose leverage is of the same order, and holding it fixed
        # keeps the recursion's variance growth attributable to the dynamics.
        exog = self._exog_row(info, last_pub + 1)
        lags = np.asarray(
            [float(info.reported.iloc[-1 - j]) for j in range(self._lag_depth())],
            dtype=float,
        )
        x = np.concatenate([exog, lags])
        xtx_inv = self._fit.xtx_inv
        leverage = float(x @ xtx_inv @ x) if xtx_inv is not None else 0.0
        sd = float(np.sqrt(self._fit.sigma2 * (1.0 + max(leverage, 0.0))))

        def _sample(
            step_horizon: int, size: int, rng: np.random.Generator
        ) -> np.ndarray:
            return rng.normal(loc=0.0, scale=sd, size=size)

        return _sample

    # ── Fitting ────────────────────────────────────────────────────

    def _candidate_rmse(
        self,
        info: InformationSet,
        *,
        n_lags: int,
        ridge_alpha: float,
    ) -> tuple[float, dict[str, Any]]:
        """Inner rolling-origin RMSE of one ``(q, ridge)`` configuration.

        The inner split uses only the most recent ``inner_validation_quarters``
        origins *within the information set*, which contains published quarters
        only.
        """
        design = _build_design(
            info,
            n_lags=n_lags,
            include_lagged_factors=self.include_lagged_factors,
        )
        min_train = max(self.min_train, design.n_params + 5)
        if design.y.size <= min_train:
            return float("inf"), {
                "reason": "not enough rows for the inner validation split",
                "n_rows": int(design.y.size),
                "min_train": min_train,
            }

        def predict_fn(n_train: int, hs: Sequence[int]) -> dict[int, float]:
            return _recursive_backtest_predict(
                design, ridge_alpha=ridge_alpha, n_train=n_train, horizons=hs
            )

        errors, diag = rolling_origin_backtest(
            design.y,
            min_train=min_train,
            horizons=self.selection_horizons,
            predict_fn=predict_fn,
            max_origins=self.inner_validation_quarters,
        )
        pooled = np.concatenate(
            [errors[h] for h in self.selection_horizons if errors.get(h) is not None]
            or [np.empty(0)]
        )
        if pooled.size == 0:
            return float("inf"), {"reason": "no inner validation errors", **diag}
        rmse = float(np.sqrt(np.mean(pooled**2)))
        return rmse, {
            "rmse": rmse,
            "n_errors": int(pooled.size),
            "selection_index": [
                str(q) for q in design.index[diag["first_origin"] :]
            ],
            **diag,
        }

    def _select(
        self, info: InformationSet
    ) -> tuple[int, float, dict[str, Any]]:
        """Choose ``q`` and the ridge strength by inner rolling-origin validation."""
        lag_grid = (
            (int(self.n_lags),)
            if self.n_lags is not None
            else tuple(int(q) for q in self.n_lags_candidates)
        )
        alpha_grid = (
            (float(self.ridge_alpha),)
            if self.ridge_alpha is not None
            else tuple(float(a) for a in self.ridge_alpha_candidates)
        )
        if not lag_grid or min(lag_grid) < 0:
            raise ValueError(f"invalid n_lags candidates {lag_grid!r}.")
        if min(alpha_grid) < 0:
            raise ValueError(f"ridge_alpha must be >= 0, got {alpha_grid!r}.")

        if len(lag_grid) == 1 and len(alpha_grid) == 1:
            return lag_grid[0], alpha_grid[0], {
                "selected_by": "user",
                "n_lags": lag_grid[0],
                "ridge_alpha": alpha_grid[0],
            }

        scores: dict[tuple[int, float], float] = {}
        detail: dict[str, Any] = {}
        for q in lag_grid:
            for a in alpha_grid:
                rmse, info_dict = self._candidate_rmse(
                    info, n_lags=q, ridge_alpha=a
                )
                scores[(q, a)] = rmse
                detail[f"q={q},alpha={a:g}"] = info_dict
        best = min(scores, key=lambda k: (scores[k], k[0], k[1]))
        if not np.isfinite(scores[best]):
            self._flag(
                f"{self.name}: inner rolling-origin validation produced no "
                f"usable errors for any of {len(scores)} configurations at "
                f"as_of={info.as_of.date()}; falling back to the smallest "
                "candidate (q=min, ridge=min). Check that enough published "
                "history exists.",
                SmoothingRegressionWarning,
            )
            best = (min(lag_grid), min(alpha_grid))
        return best[0], best[1], {
            "selected_by": "inner_rolling_origin",
            "n_lags": best[0],
            "ridge_alpha": best[1],
            "inner_validation_quarters": self.inner_validation_quarters,
            "selection_horizons": tuple(self.selection_horizons),
            "scores": {f"q={q},alpha={a:g}": v for (q, a), v in scores.items()},
            "detail": detail,
        }

    def _fit_impl(
        self, info: InformationSet
    ) -> tuple[ResidualPool, dict[str, Any]]:
        if self.interval_method not in (
            "empirical_recursive",
            "empirical_direct",
            "parametric",
        ):
            raise ValueError(
                f"{self.name}: unknown interval_method {self.interval_method!r}."
            )

        n_lags, ridge_alpha, selection = self._select(info)
        design = _build_design(
            info,
            n_lags=n_lags,
            include_lagged_factors=self.include_lagged_factors,
        )

        # Nested-selection guard: no unpublished quarter may appear in the
        # design at all, so it cannot have entered the inner split either.
        unpublished = set(info.unpublished_quarters(include_open_quarter=True))
        leaked = unpublished.intersection(set(design.index))
        if leaked:  # pragma: no cover - structurally impossible, asserted anyway
            raise AssertionError(
                f"{self.name}: unpublished quarters {sorted(map(str, leaked))} "
                "entered the training design; this is a leak."
            )

        fit = _solve(
            design.full(),
            design.y,
            ridge_alpha=ridge_alpha,
            penalised=design.factor_positions,
        )
        self._design = design
        self._fit = fit
        self._selected_lags = n_lags
        self._selected_alpha = ridge_alpha
        self._structural = unwind_to_structural(
            fit.coef, design.names, list(info.factor_columns)
        )

        horizons = tuple(sorted(set(int(h) for h in self.horizons)))
        min_train = max(self.min_train, design.n_params + 5)

        def predict_fn(n_train: int, hs: Sequence[int]) -> dict[int, float]:
            return _recursive_backtest_predict(
                design, ridge_alpha=ridge_alpha, n_train=n_train, horizons=hs
            )

        oos, backtest_diag = rolling_origin_backtest(
            design.y,
            min_train=min_train,
            horizons=horizons,
            predict_fn=predict_fn,
        )
        pool = build_residual_pool(
            oos,
            fit.residuals,
            n_params=design.n_params,
            horizons=horizons,
            min_oos=self.min_oos,
            warn=self._policy is not FallbackPolicy.AUTO,
            label=self.name,
        )

        self._emit_fit_warnings(info, design, fit)

        diagnostics: dict[str, Any] = {
            "n_train": int(design.y.size),
            "n_params": int(design.n_params),
            "features": design.names,
            "n_lags": int(n_lags),
            "ridge_alpha": float(ridge_alpha),
            "include_lagged_factors": bool(self.include_lagged_factors),
            "factor_columns": tuple(info.factor_columns),
            "coefficients": {
                name: float(fit.coef[i]) for i, name in enumerate(design.names)
            },
            "sigma": float(np.sqrt(fit.sigma2)),
            "r_squared": _r_squared(design.y, fit.residuals),
            "condition_number": fit.condition_number,
            "structural": {
                k: v for k, v in self._structural.items() if k != "theta"
            },
            "selection": selection,
            "horizons": horizons,
            "min_train": min_train,
            "backtest": backtest_diag,
            "training_quarters": (str(design.index[0]), str(design.index[-1])),
            "feature_budget": {
                "n_features": int(design.n_params),
                "n_train": int(design.y.size),
                "budget": float(design.y.size) / 5.0,
                "within_budget": bool(design.n_params <= design.y.size / 5.0),
            },
        }
        return pool, diagnostics

    def _emit_fit_warnings(
        self, info: InformationSet, design: _Design, fit: _Fit
    ) -> None:
        """Surface the ways this fit could be untrustworthy."""
        assert self._structural is not None
        theta_0 = self._structural["theta_0"]
        if not self._structural["stable"]:
            self._flag(
                f"{self.name}: θ₀ = 1 - Σc_j = {theta_0:.4f} is below the "
                f"stability floor {_THETA0_FLOOR}, so the implied smoothing "
                f"weight on past reports is {1 - theta_0:.3f}. Reduced-form "
                "nowcasts remain usable but β = b/θ₀ and target='true' are not.",
                SmoothingRegressionWarning,
            )
        c_sum = float(1.0 - theta_0)
        if c_sum >= 1.0:
            self._flag(
                f"{self.name}: Σc_j = {c_sum:.4f} >= 1, so the fitted reported "
                "series is non-stationary in its own lags and multi-quarter "
                "recursion will diverge. Consider q=1 or a shorter sample.",
                SmoothingRegressionWarning,
            )
        if fit.condition_number > 1e10:
            self._flag(
                f"{self.name}: design condition number "
                f"{fit.condition_number:.3g} indicates near-collinear "
                "regressors; coefficients are not separately interpretable. "
                "Consider a ridge penalty or fewer factors.",
                SmoothingRegressionWarning,
            )
        budget = design.y.size / 5.0
        if design.n_params > budget:
            self._flag(
                f"{self.name}: {design.n_params} parameters against "
                f"{design.y.size} training rows exceeds the n_train/5 = "
                f"{budget:.1f} feature budget.",
                SmoothingRegressionWarning,
            )


def _r_squared(y: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination."""
    centred = y - y.mean()
    tss = float(centred @ centred)
    rss = float(residuals @ residuals)
    return float(1.0 - rss / tss) if tss > 0 else float("nan")
