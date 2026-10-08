r"""
Model 3 — ``SignatureNowcaster``: the challenger, from the reference paper.

Regresses the reported quarterly return on **path-signature features of the
public market path** over a lookback window ending at the target quarter, plus
the last known reported return:

.. math::
    s_Q = a + \sum_{w \in \mathcal{W}} \gamma_w\, \tilde S^{w}(X_Q)
        + c\, s_{Q-1} + e_Q

where :math:`X_Q` is the path built by
:func:`~private_assets_frequency.nowcast.signatures.build_path`,
:math:`\tilde S^w` are standardised signature terms, and :math:`\mathcal{W}` is
the word set selected by ``keep_sigs``.

What this model is for
----------------------
The reference paper's Theorem 1 says signature regression **contains** the
linear/Kalman predictor as a special case. That is a statement about
representation, not about out-of-sample accuracy at ~90 observations. So the
honest framing is: this model can only help if there is genuine *within-quarter
path* information in how the appraiser marks — if the mark responds to the
average level over the quarter, or weights late moves more heavily, rather than
only to the quarter-end return. If there is not, the extra features are noise and
:mod:`.smoothing_regression` should win. ``evaluation.py`` reports that
comparison plainly either way.

Three things are structural, not incidental
-------------------------------------------
**Time-only signature terms are dropped from the regressors.** Time is rescaled
to ``[0, 1]`` over the realised window, so every time-only term is a *constant*:
:math:`S^{(0)} = 1`, :math:`S^{(0,0)} = 1/2`, :math:`S^{(0,0,0)} = 1/6`,
regardless of the data. Keeping them would add exact collinearity with the
intercept. Removing them is also what makes the nesting against
:mod:`.smoothing_regression` exact rather than approximate: at ``level=1`` with
one factor channel the retained feature set is ``[S^(factor)]``, which with
``level_channel='simple'`` equals :math:`F_Q` to the last bit.

**The ridge penalty exempts the lagged return and the intercept.** As in
:mod:`.smoothing_regression`, :math:`c` *is* the smoothing parameter; shrinking
it toward zero would quietly undo the thing being modelled. The default
``l1_ratio=0`` path uses a selective-penalty normal-equation solve that does
this exactly. ``l1_ratio > 0`` hands the whole block to scikit-learn's
``ElasticNet``, which penalises every coefficient — including :math:`c` — and
says so in a warning.

**The feature budget is checked before any fitting.** With ~90 quarterly
observations, ``n_features > n_train / 5`` is not a model, it is an
interpolation. Over-budget configurations are refused under
:attr:`FallbackPolicy.STRICT` and skipped with a warning under ``WARN``/``AUTO``
— the module spec's mandatory guard, implemented from the *word counts* so it
costs nothing and runs before a single signature is computed. At 90 quarters the
budget admits up to three factor channels at ``level=2`` but only one at
``level=3``; :func:`feature_budget` is the authority, since the count excludes
the dropped time-only terms and includes the lagged return and intercept.

Dependencies
------------
scikit-learn is needed only for ``l1_ratio > 0``. The default (ridge) path is
pure numpy, so this module imports and fits without the ``nowcast`` extra; the
import is lazy and the error message names the extra.

References
----------
.. [1] Cohen, Mantoan, Nesheim, de Paula, Turrell & Yang (2023) —
       "Nowcasting using regression on signatures," arXiv:2305.10256v3.
       Theorem 1 (linear signature terms); Eq. 11 (multiplier terms).
.. [2] Zou & Hastie (2005) — "Regularization and variable selection via the
       elastic net."
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy import linalg

from ...core.config import FallbackPolicy
from ..base import NowcastTarget
from ..information_set import InformationSet
from ..signatures import (
    MAX_SIGNATURE_LEVEL,
    FactorPCA,
    KeepSigs,
    PathSpec,
    build_path,
    compute_signature,
    linear_term_mask,
    signature_feature_names,
    signature_words,
)
from ..uncertainty import (
    DEFAULT_MIN_OOS,
    ResidualPool,
    build_residual_pool,
    rolling_origin_backtest,
)
from ._base import RecursiveNowcasterBase

__all__ = [
    "FEATURE_BUDGET_DIVISOR",
    "FeatureBudgetError",
    "MultiplierTerms",
    "SignatureNowcaster",
    "SignatureNowcasterWarning",
    "feature_budget",
]

FEATURE_BUDGET_DIVISOR = 5
"""``n_features`` must not exceed ``n_train / 5`` — the module spec's cap."""

MultiplierTerms = Literal["none", "time", "all"]
_TIME_CHANNEL = 0


class SignatureNowcasterWarning(UserWarning):
    """Emitted on a skipped configuration or a degenerate feature set."""


class FeatureBudgetError(ValueError):
    """Raised when a configuration exceeds the feature budget under ``strict``."""


# ──────────────────────────────────────────────────────────────────
# Feature budget
# ──────────────────────────────────────────────────────────────────


def feature_budget(
    n_train: int,
    *,
    n_channels: int,
    level: int,
    keep_sigs: KeepSigs,
    include_time: bool = True,
    multiplier_terms: MultiplierTerms = "none",
    divisor: int = FEATURE_BUDGET_DIVISOR,
) -> dict[str, Any]:
    r"""Count the features a configuration would use, and compare to the budget.

    Computed from word counts alone — no path is built and no signature is
    computed — so the guard can run before any fitting, which is what the module
    spec requires.

    The count is ``signature terms (excluding time-only) + 1 lagged reported
    return + 1 intercept + multiplier terms``. Time-only terms are excluded
    because they are structurally constant and are dropped from the regressors.

    Parameters
    ----------
    n_train
        Number of training rows the design would have.
    n_channels
        Path dimension ``d``, including the time channel when present.
    level
        Truncation level.
    keep_sigs
        ``'all'`` or ``'linear'``.
    include_time
        Whether channel 0 is time.
    multiplier_terms
        ``'none'``, ``'time'`` or ``'all'`` — see
        :class:`SignatureNowcaster`.
    divisor
        Budget divisor; ``5`` per the spec.

    Returns
    -------
    dict
        ``{'n_features', 'n_signature_terms', 'n_time_only_dropped',
        'n_multiplier_terms', 'n_train', 'budget', 'within_budget'}``.

    Examples
    --------
    >>> b = feature_budget(90, n_channels=2, level=2, keep_sigs="linear")
    >>> b["n_signature_terms"], b["n_time_only_dropped"], b["n_features"]
    (3, 2, 5)
    >>> b["budget"], b["within_budget"]
    (18.0, True)
    """
    time_channel = _TIME_CHANNEL if include_time else None
    words = signature_words(n_channels, level)
    if keep_sigs == "linear":
        mask = linear_term_mask(n_channels, level, time_channel=time_channel)
        words = [words[i] for i in range(len(words)) if mask[i]]
    elif keep_sigs != "all":
        raise ValueError(f"keep_sigs must be 'all' or 'linear', got {keep_sigs!r}.")

    time_only = [
        w for w in words if include_time and all(c == _TIME_CHANNEL for c in w)
    ]
    retained = [w for w in words if w not in set(time_only)]

    if multiplier_terms == "none":
        n_multiplier = 0
    elif multiplier_terms == "time":
        n_multiplier = len(time_only)
    elif multiplier_terms == "all":
        n_multiplier = len(retained)
    else:
        raise ValueError(
            f"multiplier_terms must be 'none', 'time' or 'all', got "
            f"{multiplier_terms!r}."
        )

    n_features = len(retained) + 2 + n_multiplier  # + s_lag1 + intercept
    budget = float(n_train) / float(divisor)
    return {
        "n_features": int(n_features),
        "n_signature_terms": int(len(retained)),
        "n_time_only_dropped": int(len(time_only)),
        "n_multiplier_terms": int(n_multiplier),
        "n_train": int(n_train),
        "budget": budget,
        "budget_divisor": int(divisor),
        "within_budget": bool(n_features <= budget),
        "level": int(level),
        "keep_sigs": keep_sigs,
        "n_channels": int(n_channels),
    }


# ──────────────────────────────────────────────────────────────────
# Estimation
# ──────────────────────────────────────────────────────────────────


class _LinearFit(NamedTuple):
    r"""A fitted signature regression, with its standardisation baked in.

    Attributes
    ----------
    mean, scale
        Column means and standard deviations of the signature block, estimated
        on **training rows only**. ``scale`` is floored at a small positive
        value so a constant column cannot produce a division by zero.
    coef_sig
        Coefficients on the standardised signature block.
    coef_lag
        Coefficient on :math:`s_{Q-1}`.
    coef_mult
        Coefficients on the multiplier (interaction) block.
    intercept
        Fitted intercept.
    residuals
        In-sample residuals.
    estimator
        ``'numpy_ridge'`` or ``'sklearn_elasticnet'``.
    """

    mean: np.ndarray
    scale: np.ndarray
    coef_sig: np.ndarray
    coef_lag: float
    coef_mult: np.ndarray
    intercept: float
    residuals: np.ndarray
    estimator: str

    def standardise(self, sig: np.ndarray) -> np.ndarray:
        """Apply the fitted standardisation to a signature block."""
        return (np.atleast_2d(sig) - self.mean) / self.scale

    def predict(
        self, sig: np.ndarray, lag: np.ndarray, mult_source: np.ndarray | None = None
    ) -> np.ndarray:
        r"""Predict :math:`s_Q` for rows of signature features and lag values.

        Parameters
        ----------
        sig
            Raw (unstandardised) signature features, shape ``(n, n_sig)``.
        lag
            :math:`s_{Q-1}` per row, shape ``(n,)``.
            For ``h >= 2`` these are the predictive draws of the previous
            quarter, which is what makes the recursion propagate uncertainty.
        mult_source
            Raw features the multiplier block interacts with, shape
            ``(n, n_mult)``, or ``None`` when there are none.

        Returns
        -------
        np.ndarray
            Shape ``(n,)``.
        """
        z = self.standardise(sig)
        lag = np.asarray(lag, dtype=float).reshape(-1)
        out = self.intercept + z @ self.coef_sig + self.coef_lag * lag
        if self.coef_mult.size:
            if mult_source is None:
                raise ValueError("this fit has multiplier terms but none were given.")
            source = np.atleast_2d(mult_source)
            out = out + (source * lag[:, None]) @ self.coef_mult
        return out


def _standardisation(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and standard deviations, with a floor on the scale."""
    mean = block.mean(axis=0)
    scale = block.std(axis=0, ddof=0)
    # A structurally constant column would otherwise divide by zero. Flooring at
    # 1.0 leaves such a column at zero after centring, so it contributes nothing
    # rather than exploding — the same outcome as dropping it, without changing
    # the column layout mid-fit.
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def _fit_selective_ridge(
    design: np.ndarray,
    y: np.ndarray,
    *,
    penalised: np.ndarray,
    alpha: float,
) -> np.ndarray:
    r"""Ridge with a penalty applied to selected columns only.

    Solves :math:`(X'X + n\alpha \Lambda)\hat b = X'y` with :math:`\Lambda`
    diagonal, one on penalised columns and zero elsewhere. The :math:`n\alpha`
    scaling matches scikit-learn's ``ElasticNet`` objective
    :math:`\tfrac{1}{2n}\lVert y - Xw\rVert^2 + \tfrac{\alpha}{2}\lVert
    w\rVert^2` at ``l1_ratio=0``, so the ``alpha`` grid means the same thing on
    both code paths.

    Falls back to :func:`numpy.linalg.lstsq` when the system is not positive
    definite, which happens on the short windows the rolling-origin backtest
    refits on.
    """
    n = design.shape[0]
    gram = design.T @ design
    if alpha > 0 and penalised.size:
        gram = gram.copy()
        gram[penalised, penalised] += float(alpha) * n
    rhs = design.T @ y
    try:
        return np.asarray(linalg.solve(gram, rhs, assume_a="pos"), dtype=float)
    except (linalg.LinAlgError, np.linalg.LinAlgError, ValueError):
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        return np.asarray(coef, dtype=float)


def _fit_elasticnet(
    design: np.ndarray, y: np.ndarray, *, alpha: float, l1_ratio: float
) -> tuple[np.ndarray, float]:
    """Fit scikit-learn's ``ElasticNet`` on ``design`` (no intercept column).

    Raises
    ------
    ImportError
        With a message naming the optional extra, since this is the only code
        path in the package that needs scikit-learn.
    """
    try:
        from sklearn.linear_model import ElasticNet  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "SignatureNowcaster needs scikit-learn for l1_ratio > 0. Install it "
            "with `pip install private_assets_frequency[nowcast]`, or use the "
            "default l1_ratio=0, whose ridge path is pure numpy."
        ) from exc

    model = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=20_000,
        tol=1e-8,
        selection="cyclic",
    )
    model.fit(design, y)
    return np.asarray(model.coef_, dtype=float), float(model.intercept_)


def _fit_regression(
    sig: np.ndarray,
    lag: np.ndarray,
    y: np.ndarray,
    mult_source: np.ndarray | None,
    *,
    alpha: float,
    l1_ratio: float,
    standardize: bool,
) -> _LinearFit:
    """Fit the signature regression on one training block.

    The standardisation is estimated here, from these rows only — which is why
    the rolling-origin backtest calls this per origin rather than reusing a
    scaler fitted once.
    """
    sig = np.atleast_2d(sig)
    lag = np.asarray(lag, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, n_sig = sig.shape

    if standardize:
        mean, scale = _standardisation(sig)
    else:
        mean = np.zeros(n_sig)
        scale = np.ones(n_sig)
    z = (sig - mean) / scale

    blocks = [z, lag.reshape(-1, 1)]
    n_mult = 0
    if mult_source is not None and np.atleast_2d(mult_source).shape[1]:
        source = np.atleast_2d(mult_source)
        blocks.append(source * lag[:, None])
        n_mult = source.shape[1]
    stacked = np.hstack(blocks)

    if l1_ratio > 0.0:
        coef, intercept = _fit_elasticnet(
            stacked, y, alpha=alpha, l1_ratio=l1_ratio
        )
        estimator = "sklearn_elasticnet"
    else:
        design = np.hstack([np.ones((n, 1)), stacked])
        # Penalise the signature block and the multipliers, never the intercept
        # (column 0) and never the lagged reported return (column 1 + n_sig).
        penalised = np.concatenate(
            [
                np.arange(1, 1 + n_sig),
                np.arange(2 + n_sig, 2 + n_sig + n_mult),
            ]
        ).astype(int)
        full = _fit_selective_ridge(design, y, penalised=penalised, alpha=alpha)
        intercept = float(full[0])
        coef = full[1:]
        estimator = "numpy_ridge"

    coef_sig = coef[:n_sig]
    coef_lag = float(coef[n_sig])
    coef_mult = coef[n_sig + 1 :]
    multiplier_contribution: np.ndarray | float = 0.0
    if n_mult:
        # n_mult is non-zero only where `mult_source` was supplied above, so the
        # two travel together; spelling it out keeps the invariant checkable.
        assert mult_source is not None
        multiplier_contribution = (
            np.atleast_2d(mult_source) * lag[:, None]
        ) @ coef_mult
    fitted = intercept + z @ coef_sig + coef_lag * lag + multiplier_contribution
    return _LinearFit(
        mean=mean,
        scale=scale,
        coef_sig=coef_sig,
        coef_lag=coef_lag,
        coef_mult=coef_mult,
        intercept=intercept,
        residuals=y - fitted,
        estimator=estimator,
    )


# ──────────────────────────────────────────────────────────────────
# Design
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Config:
    """One point of the hyperparameter grid."""

    level: int
    lookback: int
    keep_sigs: str
    alpha: float
    l1_ratio: float

    def label(self) -> str:
        return (
            f"level={self.level},lookback={self.lookback},"
            f"keep={self.keep_sigs},alpha={self.alpha:g},l1={self.l1_ratio:g}"
        )


@dataclass
class _Design:
    """Signature design matrix for one ``(level, lookback, keep_sigs)`` triple."""

    index: pd.PeriodIndex
    y: np.ndarray
    sig: np.ndarray
    lag: np.ndarray
    sig_names: list[str]
    mult_names: list[str]
    mult_columns: np.ndarray
    channel_names: list[str]
    diagnostics: dict[str, Any]

    @property
    def n_rows(self) -> int:
        return int(self.y.size)

    def mult_source(self, rows: np.ndarray | slice | None = None) -> np.ndarray | None:
        """The raw columns the multiplier block interacts with."""
        if self.mult_columns.size == 0:
            return None
        block = self.sig[:, self.mult_columns]
        return block if rows is None else block[rows]


# ──────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────


@dataclass
class SignatureNowcaster(RecursiveNowcasterBase):
    r"""Nowcast the reported return from path-signature features.

    Parameters
    ----------
    level
        Signature truncation level. Default ``2``, the module spec's default.
        ``None`` selects from ``level_candidates`` by inner rolling-origin
        validation.
    level_candidates
        Candidates when ``level is None``. ``(1, 2, 3)``; level 3 is reachable
        but not a default, because at ~90 observations the extra selection
        dimension costs more than it buys.
    lookback
        Window length in quarters. Default ``1`` (within-quarter only).
        ``None`` selects from ``lookback_candidates``.
    lookback_candidates
        ``(1, 2, 4)`` per the spec.
    keep_sigs
        ``'linear'`` (default) or ``'all'``. ``None`` selects.
    keep_sigs_candidates
        ``('linear', 'all')``.
    alpha
        Regularisation strength. ``None`` (default) selects from
        ``alpha_candidates`` — this is the one hyperparameter selected by
        default, which keeps the grid at four configurations.
    alpha_candidates
        ``(1e-4, 1e-3, 1e-2, 1e-1)``.
    l1_ratio
        ``0.0`` (default) gives a pure ridge solved in numpy with the penalty
        **exempting** the intercept and :math:`s_{Q-1}`. Any positive value
        hands the problem to scikit-learn's ``ElasticNet``, which penalises every
        coefficient including :math:`s_{Q-1}` — a warning says so.
    l1_ratio_candidates
        Candidates when ``l1_ratio is None``.
    path_frequency, level_channel, basepoint, missing, n_components
        Passed through to
        :class:`~private_assets_frequency.nowcast.signatures.PathSpec`.
        ``level_channel='simple'`` makes the ``level=1`` feature exactly
        :math:`F_Q`, which is what the nesting test uses.
    include_time
        Keep a time channel. On by default; required for the level-2 terms that
        carry path shape.
    include_early_signals
        Use ``info.early_signals`` as extra path channels when present.
    multiplier_terms
        Interactions of :math:`s_{Q-1}` with signature terms — the reference
        paper's Eq. 11. ``'none'`` (default); ``'time'`` is the spec's literal
        reading and is **degenerate** (see Notes); ``'all'`` interacts with the
        retained non-time terms and gives a genuinely path-dependent AR
        coefficient.
    standardize
        Standardise the signature block, with the scaler fitted on training rows
        only. On by default.
    target
        ``'reported'`` only — see Notes.
    interval_method
        ``'empirical_recursive'`` (default) or ``'empirical_direct'``.
    horizons, min_train, inner_validation_quarters, selection_horizons,
    min_oos, fallback_policy
        As in
        :class:`~..models.smoothing_regression.SmoothingRegressionNowcaster`.

    Attributes
    ----------
    name
        ``'signature'``.

    Notes
    -----
    **The spec's ``multiplier_terms='time'`` is degenerate, and the module says
    so.** Time is rescaled to ``[0, 1]`` over the realised window, so every
    time-only signature term is a constant — :math:`S^{(0)} = 1`,
    :math:`S^{(0,0)} = 1/2`. Interacting a constant with :math:`s_{Q-1}` gives a
    rescaled copy of :math:`s_{Q-1}`: exactly collinear, contributing nothing but
    rank deficiency. It is implemented for fidelity to the spec and warns when
    used. ``'all'`` is the version that does something: interacting
    :math:`s_{Q-1}` with the *data* signature terms lets the effective smoothing
    coefficient depend on the public path, which is the behaviour Eq. 11 is
    after.

    **``target='true'`` is not available.** This model has no structural
    decomposition to read :math:`\alpha` and :math:`\beta` from — its
    coefficients are on signature terms, not on factor returns, and there is no
    :math:`\theta_0` to unwind. Use
    :class:`~..models.smoothing_regression.SmoothingRegressionNowcaster` or
    :class:`~..models.structural.StructuralSmoothingNowcaster` for that target.

    **PCA, when enabled, is fitted once per information set** rather than per
    inner-validation origin, because refitting it would force every signature to
    be recomputed at every origin. The outer no-leakage guarantee is untouched —
    the rotation never sees data after ``as_of`` — but an inner-validation origin
    does see a rotation estimated on later *published* quarters. Recorded as
    ``diagnostics['pca_fit_scope']``. PCA is off by default.

    Examples
    --------
    >>> model = SignatureNowcaster(level=2, lookback=1).fit(info)   # doctest: +SKIP
    >>> model.fit_diagnostics["feature_budget"]                    # doctest: +SKIP
    {'n_features': 5, 'n_train': 100, 'budget': 20.0, 'within_budget': True, ...}
    """

    level: int | None = 2
    level_candidates: tuple[int, ...] = (1, 2, 3)
    lookback: int | None = 1
    lookback_candidates: tuple[int, ...] = (1, 2, 4)
    keep_sigs: KeepSigs | None = "linear"
    keep_sigs_candidates: tuple[str, ...] = ("linear", "all")
    alpha: float | None = None
    alpha_candidates: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
    l1_ratio: float | None = 0.0
    l1_ratio_candidates: tuple[float, ...] = (0.0, 0.5)

    path_frequency: str = "daily"
    level_channel: str = "log"
    basepoint: bool = True
    missing: str = "ffill"
    n_components: int | None = None
    include_time: bool = True
    include_early_signals: bool = True
    multiplier_terms: MultiplierTerms = "none"
    standardize: bool = True

    target: NowcastTarget = "reported"
    interval_method: str = "empirical_recursive"
    horizons: tuple[int, ...] = (1, 2)
    min_train: int = 40
    inner_validation_quarters: int = 12
    selection_horizons: tuple[int, ...] = (1,)
    min_oos: int = DEFAULT_MIN_OOS
    backend: str = "auto"
    fallback_policy: FallbackPolicy | str = FallbackPolicy.WARN
    name: str = field(default="signature", init=False)

    _design: _Design | None = field(default=None, init=False, repr=False)
    _fit: _LinearFit | None = field(default=None, init=False, repr=False)
    _config: _Config | None = field(default=None, init=False, repr=False)
    _pca: FactorPCA | None = field(default=None, init=False, repr=False)
    _spec: PathSpec | None = field(default=None, init=False, repr=False)

    # ── Accessors ──────────────────────────────────────────────────

    @property
    def config(self) -> _Config:
        """The selected hyperparameter configuration."""
        if self._config is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        return self._config

    @property
    def coefficients(self) -> pd.Series:
        """Fitted coefficients on standardised features."""
        if self._fit is None or self._design is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        names = ["intercept"] + self._design.sig_names + ["s_lag1"]
        values = [self._fit.intercept, *self._fit.coef_sig, self._fit.coef_lag]
        names.extend(self._design.mult_names)
        values.extend(self._fit.coef_mult)
        return pd.Series(values, index=names, name="coef")

    @property
    def sigma(self) -> float:
        """Residual standard deviation of the fitted regression."""
        if self._fit is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        resid = self._fit.residuals
        dof = max(resid.size - self.coefficients.size, 1)
        return float(np.sqrt(resid @ resid / dof))

    # ── Path spec ──────────────────────────────────────────────────

    def _path_spec(self, info: InformationSet, lookback: int) -> PathSpec:
        return PathSpec(
            lookback=lookback,
            path_frequency=self.path_frequency,  # type: ignore[arg-type]
            level_channel=self.level_channel,  # type: ignore[arg-type]
            include_time=self.include_time,
            basepoint=self.basepoint,
            missing=self.missing,  # type: ignore[arg-type]
            n_components=self.n_components,
            early_signal_columns=(
                None if self.include_early_signals else ()
            ),
        )

    def _n_channels(self, info: InformationSet) -> int:
        n_early = 0
        if self.include_early_signals and info.early_signals is not None:
            n_early = int(info.early_signals.shape[1])
        data = (
            self.n_components
            if self.n_components is not None
            else len(info.factor_columns)
        )
        return int(self.include_time) + int(data) + n_early

    # ── Eligible rows (cheap, for the pre-fit budget check) ────────

    def _eligible_quarters(
        self, info: InformationSet, lookback: int
    ) -> pd.PeriodIndex:
        """Published quarters that could carry a design row, without building one.

        A quarter qualifies when its own factor aggregate is complete, every
        quarter in its lookback window has complete factor data, and the previous
        quarter's reported value is published. Counting these is what lets the
        feature-budget guard run before any signature is computed.
        """
        published = pd.PeriodIndex(info.reported.index, freq="Q")
        complete = info.factor_quarter_complete
        eligible: list[pd.Period] = []
        for quarter in published:
            if quarter - 1 not in info.reported.index:
                continue
            window = [quarter - j for j in range(lookback)]
            if all(bool(complete.get(q, False)) for q in window):
                eligible.append(quarter)
        return pd.PeriodIndex(eligible, freq="Q")

    # ── Design construction ────────────────────────────────────────

    def _build_design(
        self, info: InformationSet, level: int, lookback: int, keep_sigs: str
    ) -> _Design:
        """Build the signature design matrix over eligible published quarters."""
        spec = self._path_spec(info, lookback)
        quarters = self._eligible_quarters(info, lookback)
        if quarters.size == 0:
            raise ValueError(
                f"{self.name}: no published quarter at as_of={info.as_of.date()} "
                f"has a complete {lookback}-quarter factor window and a "
                "published previous reported value."
            )

        time_channel = _TIME_CHANNEL if self.include_time else None
        rows: list[np.ndarray] = []
        kept: list[pd.Period] = []
        channel_names: list[str] = []
        names: list[str] = []
        first_diag: dict[str, Any] | None = None

        for quarter in quarters:
            try:
                path, chan, diag = build_path(info, quarter, spec, pca=self._pca)
            except ValueError:
                continue
            features = compute_signature(
                path,
                level,
                backend=self.backend,
                keep_sigs=keep_sigs,  # type: ignore[arg-type]
                time_channel=time_channel,
            )
            if not channel_names:
                channel_names = list(chan)
                names = signature_feature_names(
                    channel_names, level, keep_sigs, time_channel=time_channel  # type: ignore[arg-type]
                )
                first_diag = diag
            elif list(chan) != channel_names:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"{self.name}: path channels changed between quarters "
                    f"({channel_names} vs {list(chan)})."
                )
            rows.append(features)
            kept.append(quarter)

        if not rows:
            raise ValueError(
                f"{self.name}: no usable path could be built for any eligible "
                f"quarter at as_of={info.as_of.date()}."
            )

        raw = np.vstack(rows)
        index = pd.PeriodIndex(kept, freq="Q")

        # Drop the structurally constant time-only terms; keep their positions so
        # multiplier_terms='time' can still reference them.
        words = signature_words(len(channel_names), level)
        if keep_sigs == "linear":
            mask = linear_term_mask(
                len(channel_names), level, time_channel=time_channel
            )
            words = [words[i] for i in range(len(words)) if mask[i]]
        is_time_only = np.array(
            [self.include_time and all(c == _TIME_CHANNEL for c in w) for w in words],
            dtype=bool,
        )
        keep_cols = np.flatnonzero(~is_time_only)
        time_cols = np.flatnonzero(is_time_only)
        sig = raw[:, keep_cols]
        sig_names = [names[i] for i in keep_cols]

        if self.multiplier_terms == "time":
            mult_cols_in_raw = time_cols
        elif self.multiplier_terms == "all":
            mult_cols_in_raw = keep_cols
        else:
            mult_cols_in_raw = np.empty(0, dtype=int)
        # Re-express multiplier source columns as positions within `sig` where
        # possible; time-only columns are not in `sig`, so carry them separately.
        if self.multiplier_terms == "time" and time_cols.size:
            mult_block = raw[:, time_cols]
            mult_names = [f"s_lag1*{names[i]}" for i in time_cols]
            sig = np.hstack([sig, mult_block])
            sig_names = sig_names + [f"_mult_source_{names[i]}" for i in time_cols]
            mult_columns = np.arange(len(keep_cols), len(keep_cols) + time_cols.size)
        elif self.multiplier_terms == "all" and keep_cols.size:
            mult_names = [f"s_lag1*{names[i]}" for i in keep_cols]
            mult_columns = np.arange(len(keep_cols))
        else:
            mult_names = []
            mult_columns = np.empty(0, dtype=int)
        del mult_cols_in_raw

        lag = np.asarray(
            [float(info.reported[q - 1]) for q in index], dtype=float
        )
        y = np.asarray([float(info.reported[q]) for q in index], dtype=float)

        diagnostics = dict(first_diag or {})
        diagnostics.update(
            {
                "n_eligible_quarters": int(quarters.size),
                "n_rows": int(index.size),
                "channel_names": tuple(channel_names),
                "n_time_only_dropped": int(time_cols.size),
                "dropped_time_only_terms": tuple(names[i] for i in time_cols),
            }
        )
        return _Design(
            index=index,
            y=y,
            sig=sig,
            lag=lag,
            sig_names=sig_names,
            mult_names=mult_names,
            mult_columns=mult_columns,
            channel_names=channel_names,
            diagnostics=diagnostics,
        )

    # ── Backtest over a design ─────────────────────────────────────

    def _backtest_predict(
        self,
        design: _Design,
        config: _Config,
        n_train: int,
        horizons: Sequence[int],
    ) -> dict[int, float]:
        """Fit on rows ``[0, n_train)`` and nowcast forward recursively.

        Everything is re-estimated here, standardisation included. The signature
        features themselves are not re-estimated because they are not estimated
        at all — they are a deterministic function of observable public data.
        """
        fit = _fit_regression(
            design.sig[:n_train],
            design.lag[:n_train],
            design.y[:n_train],
            design.mult_source(slice(0, n_train)),
            alpha=config.alpha,
            l1_ratio=config.l1_ratio,
            standardize=self.standardize,
        )
        lag_value = float(design.lag[n_train])
        out: dict[int, float] = {}
        for h in sorted(horizons):
            row = n_train + h - 1
            if row >= design.n_rows:
                break
            source = design.mult_source(np.array([row])) if design.mult_names else None
            pred = float(
                fit.predict(
                    design.sig[row : row + 1], np.array([lag_value]), source
                )[0]
            )
            out[h] = pred
            lag_value = pred
        return out

    def _candidate_rmse(
        self, info: InformationSet, design: _Design, config: _Config
    ) -> tuple[float, dict[str, Any]]:
        """Inner rolling-origin RMSE of one configuration, published rows only."""
        n_params = len(design.sig_names) + 2 + len(design.mult_names)
        min_train = max(self.min_train, n_params + 5)
        if design.n_rows <= min_train:
            return float("inf"), {
                "reason": "not enough rows for the inner validation split",
                "n_rows": design.n_rows,
                "min_train": min_train,
            }

        def predict_fn(n_train: int, hs: Sequence[int]) -> dict[int, float]:
            return self._backtest_predict(design, config, n_train, hs)

        errors, diag = rolling_origin_backtest(
            design.y,
            min_train=min_train,
            horizons=self.selection_horizons,
            predict_fn=predict_fn,
            max_origins=self.inner_validation_quarters,
        )
        pooled = np.concatenate(
            [errors[h] for h in self.selection_horizons if h in errors]
            or [np.empty(0)]
        )
        if pooled.size == 0:
            return float("inf"), {"reason": "no inner validation errors", **diag}
        rmse = float(np.sqrt(np.mean(pooled**2)))
        first = diag["first_origin"]
        return rmse, {
            "rmse": rmse,
            "n_errors": int(pooled.size),
            "selection_index": [str(q) for q in design.index[first:]],
            **diag,
        }

    # ── Selection, with the budget guard ───────────────────────────

    def _grid(self) -> list[_Config]:
        levels = (
            (int(self.level),)
            if self.level is not None
            else tuple(int(v) for v in self.level_candidates)
        )
        lookbacks = (
            (int(self.lookback),)
            if self.lookback is not None
            else tuple(int(v) for v in self.lookback_candidates)
        )
        keeps = (
            (str(self.keep_sigs),)
            if self.keep_sigs is not None
            else tuple(str(v) for v in self.keep_sigs_candidates)
        )
        alphas = (
            (float(self.alpha),)
            if self.alpha is not None
            else tuple(float(v) for v in self.alpha_candidates)
        )
        ratios = (
            (float(self.l1_ratio),)
            if self.l1_ratio is not None
            else tuple(float(v) for v in self.l1_ratio_candidates)
        )
        for value, label in ((levels, "level"), (lookbacks, "lookback")):
            if min(value) < 1:
                raise ValueError(f"{label} candidates must be >= 1, got {value!r}.")
        if max(levels) > MAX_SIGNATURE_LEVEL:
            raise ValueError(
                f"level must be <= {MAX_SIGNATURE_LEVEL}, got {max(levels)}."
            )
        for keep in keeps:
            if keep not in ("linear", "all"):
                raise ValueError(
                    f"keep_sigs must be 'linear' or 'all', got {keep!r}."
                )
        if min(alphas) < 0:
            raise ValueError(f"alpha must be >= 0, got {alphas!r}.")
        if min(ratios) < 0 or max(ratios) > 1:
            raise ValueError(f"l1_ratio must be in [0, 1], got {ratios!r}.")
        return [
            _Config(level=lv, lookback=lb, keep_sigs=ks, alpha=a, l1_ratio=r)
            for lv in levels
            for lb in lookbacks
            for ks in keeps
            for a in alphas
            for r in ratios
        ]

    def _budget_for(
        self, info: InformationSet, config: _Config
    ) -> dict[str, Any]:
        """Feature budget for one configuration, computed before any fitting."""
        n_train = int(self._eligible_quarters(info, config.lookback).size)
        return feature_budget(
            n_train,
            n_channels=self._n_channels(info),
            level=config.level,
            keep_sigs=config.keep_sigs,  # type: ignore[arg-type]
            include_time=self.include_time,
            multiplier_terms=self.multiplier_terms,
        )

    def _select(
        self, info: InformationSet, grid: list[_Config] | None = None
    ) -> tuple[_Config, _Design, dict[str, Any]]:
        """Choose a configuration, enforcing the feature budget first.

        Parameters
        ----------
        info
            The information set.
        grid
            A pre-validated grid from :meth:`_grid`; built here when omitted.
        """
        grid = grid if grid is not None else self._grid()
        budgets = {c.label(): self._budget_for(info, c) for c in grid}
        admissible = [c for c in grid if budgets[c.label()]["within_budget"]]
        skipped = [c for c in grid if not budgets[c.label()]["within_budget"]]

        if skipped:
            worst = max(
                (budgets[c.label()] for c in skipped),
                key=lambda b: b["n_features"],
            )
            message = (
                f"{self.name}: {len(skipped)} of {len(grid)} configuration(s) "
                f"exceed the feature budget n_train/{FEATURE_BUDGET_DIVISOR}. "
                f"The largest needs {worst['n_features']} features against a "
                f"budget of {worst['budget']:.1f} "
                f"(n_train={worst['n_train']}, level={worst['level']}, "
                f"keep_sigs={worst['keep_sigs']!r}, "
                f"channels={worst['n_channels']}). "
                + (
                    "Skipping them."
                    if admissible
                    else "No configuration is admissible."
                )
            )
            if self._policy is FallbackPolicy.STRICT:
                raise FeatureBudgetError(message)
            self._flag(message, SignatureNowcasterWarning)

        if not admissible:
            raise FeatureBudgetError(
                f"{self.name}: every configuration in the grid exceeds the "
                f"feature budget n_train/{FEATURE_BUDGET_DIVISOR} at "
                f"as_of={info.as_of.date()}. Reduce `level`, set "
                "`keep_sigs='linear'`, use fewer factor channels, or set "
                "`n_components` to compress them. Budgets: "
                f"{ {k: (v['n_features'], v['budget']) for k, v in budgets.items()} }"
            )

        designs: dict[tuple[int, int, str], _Design] = {}

        def design_for(config: _Config) -> _Design:
            key = (config.level, config.lookback, config.keep_sigs)
            if key not in designs:
                designs[key] = self._build_design(
                    info, config.level, config.lookback, config.keep_sigs
                )
            return designs[key]

        if len(admissible) == 1:
            chosen = admissible[0]
            return chosen, design_for(chosen), {
                "selected_by": "user",
                "config": chosen.label(),
                "n_configurations": 1,
                "n_skipped_over_budget": len(skipped),
                "budgets": budgets,
            }

        scores: dict[str, float] = {}
        detail: dict[str, Any] = {}
        for config in admissible:
            try:
                rmse, info_dict = self._candidate_rmse(
                    info, design_for(config), config
                )
            except ValueError as exc:
                scores[config.label()] = float("inf")
                detail[config.label()] = {"reason": str(exc)}
                continue
            scores[config.label()] = rmse
            detail[config.label()] = info_dict

        best = min(
            admissible,
            key=lambda c: (
                scores[c.label()],
                c.level,
                c.lookback,
                c.keep_sigs,
                c.alpha,
                c.l1_ratio,
            ),
        )
        if not np.isfinite(scores[best.label()]):
            self._flag(
                f"{self.name}: inner rolling-origin validation produced no usable "
                f"errors for any of {len(admissible)} admissible configurations "
                f"at as_of={info.as_of.date()}; falling back to the smallest. "
                "Check that enough published history exists.",
                SignatureNowcasterWarning,
            )
        return best, design_for(best), {
            "selected_by": "inner_rolling_origin",
            "config": best.label(),
            "n_configurations": len(admissible),
            "n_skipped_over_budget": len(skipped),
            "inner_validation_quarters": self.inner_validation_quarters,
            "selection_horizons": tuple(self.selection_horizons),
            "scores": scores,
            "detail": detail,
            "budgets": budgets,
        }

    # ── Subclass contract ──────────────────────────────────────────

    def _lag_depth(self) -> int:
        return 1

    def _conditional_mean(
        self, info: InformationSet, quarter: pd.Period, history: np.ndarray
    ) -> np.ndarray:
        assert self._fit is not None and self._design is not None
        assert self._config is not None and self._spec is not None
        hist = np.asarray(history, dtype=float)
        features = self._features_for(info, quarter)
        n = hist.shape[0]
        sig = np.repeat(features.reshape(1, -1), n, axis=0)
        source = (
            sig[:, self._design.mult_columns]
            if self._design.mult_names
            else None
        )
        return self._fit.predict(sig, hist[:, 0], source)

    def _features_for(
        self, info: InformationSet, quarter: pd.Period
    ) -> np.ndarray:
        """Signature features for one target quarter, in the design's layout."""
        assert self._config is not None and self._spec is not None
        assert self._design is not None
        time_channel = _TIME_CHANNEL if self.include_time else None
        path, channels, _ = build_path(info, quarter, self._spec, pca=self._pca)
        if list(channels) != self._design.channel_names:
            raise ValueError(
                f"{self.name}: path channels for {quarter} are {list(channels)} "
                f"but the fit used {self._design.channel_names}."
            )
        raw = compute_signature(
            path,
            self._config.level,
            backend=self.backend,
            keep_sigs=self._config.keep_sigs,  # type: ignore[arg-type]
            time_channel=time_channel,
        )
        words = signature_words(len(channels), self._config.level)
        if self._config.keep_sigs == "linear":
            mask = linear_term_mask(
                len(channels), self._config.level, time_channel=time_channel
            )
            words = [words[i] for i in range(len(words)) if mask[i]]
        is_time_only = np.array(
            [self.include_time and all(c == _TIME_CHANNEL for c in w) for w in words],
            dtype=bool,
        )
        keep = raw[~is_time_only]
        if self.multiplier_terms == "time":
            keep = np.concatenate([keep, raw[is_time_only]])
        return keep

    # ── Fitting ────────────────────────────────────────────────────

    def _fit_impl(
        self, info: InformationSet
    ) -> tuple[ResidualPool, dict[str, Any]]:
        if self.target != "reported":
            raise ValueError(
                f"{self.name} supports target='reported' only. It has no "
                "structural factor decomposition to read alpha and beta from — "
                "its coefficients sit on signature terms, not factor returns. "
                "Use SmoothingRegressionNowcaster or "
                "StructuralSmoothingNowcaster for target='true'."
            )
        if self.interval_method not in ("empirical_recursive", "empirical_direct"):
            raise ValueError(
                f"{self.name}: unknown interval_method {self.interval_method!r}; "
                "expected 'empirical_recursive' or 'empirical_direct'."
            )
        # Validate the hyperparameter grid *before* emitting any advisory
        # warnings: under a strict warning filter a warning becomes an exception,
        # and an invalid argument should surface as its own ValueError rather than
        # being masked by advice about a different setting.
        grid = self._grid()
        if self.multiplier_terms == "time":
            self._flag(
                f"{self.name}: multiplier_terms='time' interacts s_(Q-1) with the "
                "time-only signature terms, which are structurally constant "
                "(time is rescaled to [0,1] over every window, so S(0)=1, "
                "S(0,0)=1/2). Those interactions are exactly collinear with "
                "s_(Q-1) and add rank deficiency, not information. Implemented "
                "for fidelity to the spec; use multiplier_terms='all' for a "
                "path-dependent AR coefficient that is not degenerate.",
                SignatureNowcasterWarning,
            )
        if self.l1_ratio is not None and self.l1_ratio > 0:
            self._flag(
                f"{self.name}: l1_ratio={self.l1_ratio} routes the fit through "
                "scikit-learn's ElasticNet, which penalises every coefficient "
                "including the lagged reported return s_(Q-1). That coefficient "
                "is the smoothing parameter, so shrinking it works against the "
                "model. The default l1_ratio=0 uses a selective-penalty ridge "
                "that exempts it.",
                SignatureNowcasterWarning,
            )

        self._pca = None
        if self.n_components is not None:
            # Fitted once per information set — see the class Notes on why, and
            # on exactly which guarantee that does and does not weaken.
            self._pca = FactorPCA(n_components=self.n_components).fit(
                info.public_factors
            )

        config, design, selection = self._select(info, grid)
        self._config = config
        self._design = design
        self._spec = self._path_spec(info, config.lookback)

        unpublished = set(info.unpublished_quarters(include_open_quarter=True))
        leaked = unpublished.intersection(set(design.index))
        if leaked:  # pragma: no cover - structurally impossible, asserted anyway
            raise AssertionError(
                f"{self.name}: unpublished quarters {sorted(map(str, leaked))} "
                "entered the training design; this is a leak."
            )

        fit = _fit_regression(
            design.sig,
            design.lag,
            design.y,
            design.mult_source(),
            alpha=config.alpha,
            l1_ratio=config.l1_ratio,
            standardize=self.standardize,
        )
        self._fit = fit

        n_params = len(design.sig_names) + 2 + len(design.mult_names)
        horizons = tuple(sorted(set(int(h) for h in self.horizons)))
        min_train = max(self.min_train, n_params + 5)

        def predict_fn(n_train: int, hs: Sequence[int]) -> dict[int, float]:
            return self._backtest_predict(design, config, n_train, hs)

        oos, backtest_diag = rolling_origin_backtest(
            design.y,
            min_train=min_train,
            horizons=horizons,
            predict_fn=predict_fn,
        )
        pool = build_residual_pool(
            oos,
            fit.residuals,
            n_params=n_params,
            horizons=horizons,
            min_oos=self.min_oos,
            warn=self._policy is not FallbackPolicy.AUTO,
            label=self.name,
        )

        budget = self._budget_for(info, config)
        diagnostics: dict[str, Any] = {
            "n_train": int(design.n_rows),
            "n_params": int(n_params),
            "features": ["intercept"]
            + list(design.sig_names)
            + ["s_lag1"]
            + list(design.mult_names),
            "level": config.level,
            "lookback": config.lookback,
            "keep_sigs": config.keep_sigs,
            "alpha": config.alpha,
            "l1_ratio": config.l1_ratio,
            "estimator": fit.estimator,
            "channel_names": tuple(design.channel_names),
            "factor_columns": tuple(info.factor_columns),
            "path_frequency": self.path_frequency,
            "level_channel": self.level_channel,
            "basepoint": self.basepoint,
            "missing": self.missing,
            "multiplier_terms": self.multiplier_terms,
            "n_components": self.n_components,
            "pca_fit_scope": (
                "information_set" if self.n_components is not None else None
            ),
            "backend": self.backend,
            "coefficients": self.coefficients.to_dict(),
            "sigma": self.sigma,
            "r_squared": _r_squared(design.y, fit.residuals),
            "feature_budget": budget,
            "selection": selection,
            "horizons": horizons,
            "min_train": min_train,
            "backtest": backtest_diag,
            "training_quarters": (
                str(design.index[0]),
                str(design.index[-1]),
            ),
            "path": design.diagnostics,
        }
        return pool, diagnostics


def _r_squared(y: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination."""
    centred = y - y.mean()
    tss = float(centred @ centred)
    rss = float(residuals @ residuals)
    return float(1.0 - rss / tss) if tss > 0 else float("nan")
