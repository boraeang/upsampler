r"""
Model 2 — ``StructuralSmoothingNowcaster``: reuse a fitted desmoothing model.

Where :mod:`.smoothing_regression` estimates the reduced form from scratch, this
model takes an already-fitted
:class:`~private_assets_frequency.core.protocols.DesmoothedResult` and evaluates
the *same* forecast using that model's structural parameters. Nothing is
re-estimated. The point is interpretability and discipline: if Stage 1 of the
pipeline has already committed to a λ and a β for this strategy, the nowcast
should be consistent with them rather than quietly implying different ones.

Supported smoothing models
--------------------------
``ar1_bayesian``
    :math:`s_Q = (1-\lambda)\,(\alpha + \beta \cdot F_Q) + \lambda\, s_{Q-1}`.
    The fit regresses :math:`y(\lambda) = (s_t - \lambda s_{t-1})/(1-\lambda)` on
    :math:`[F_t, 1]`, so its reported ``alpha``, ``beta`` and ``sigma_eps`` are
    already on the **unsmoothed-return** scale and enter the formula directly.

``rudin_reparam``
    Its Eq. 4 form, exactly as fitted:
    :math:`s_Q = \alpha + \theta_0 \sum_i \beta_i F_{i,Q}
    + \sum_{j=1}^{Q} \theta_j s_{Q-j}`.
    Because ``rudin_reparam`` reports ``beta`` already unwound
    (:math:`\beta_i = c_i/\theta_0`), multiplying back by :math:`\theta_0`
    reproduces the fitted composite coefficient :math:`c_i` exactly, so the
    nowcast is numerically identical to that regression's own prediction.

``okunev_white``
    Not supported — it is a factor-free desmoother, so there is no
    :math:`\alpha + \beta F_Q` to form a nowcast from. Raises
    :class:`UnsupportedSmoothingModelError` with that explanation rather than
    failing obscurely later.

The λ convention
----------------
λ is **not a dimensionless constant** — it is the per-period decay of the
appraisal filter, so its value depends on the frequency the model was fitted at.
A nowcast of a *quarter* needs a *quarterly* λ.

As built, ``ar1_bayesian`` fits the raw observed series at whatever frequency
the caller passes, with no rolling-annual transform (``use_rolling_annual`` is
described in its own docstring as planned, not implemented), and the pipeline
runner passes the native quarterly series. So in the shipped configuration **no
conversion is needed**, and applying one would corrupt λ. This module therefore
*infers* the convention from the fitted result's own index rather than assuming
one, and converts only when it must. See
:func:`convert_lambda_to_quarterly` for the two conversions and their
derivations — including why the obvious :math:`\lambda_q = \lambda_a^{1/4}` is
badly wrong for *rolling* annual returns.

Where the uncertainty comes from
--------------------------------
The residual pool replays the fixed-parameter nowcast over the published history
inside the information set. Those errors contain the model's specification error
but **not** parameter-estimation error, since the parameters were fitted
elsewhere and are held fixed. Two consequences, both surfaced rather than
hidden:

* ``diagnostics['structural_fit_span']`` records the index span of the
  ``DesmoothedResult``. If it extends past the information set's last published
  quarter, the parameters saw the future and the backtest is leaky — the model
  warns, and escalates under :attr:`FallbackPolicy.STRICT`. The evaluation
  harness re-fits the desmoothing model inside each information set to avoid
  this; a full-sample result is only legitimate for a production nowcast at
  today's ``as_of``, where there is no later data to leak.
* ``integrate_lambda_posterior=True`` draws λ from the ``ar1_bayesian`` grid
  posterior so parameter uncertainty enters the predictive distribution.

References
----------
.. [1] Geltner (1993) — AR(1) appraisal smoothing.
.. [2] Rudin, Mao, Zhang & Fink (2019) — "Fitting Private Equity into the Total
       Portfolio Framework," Eq. 3-4.
.. [3] Working (1960) — "Note on the correlation of first differences of
       averages in a random chain." The classic result that overlapping
       averages are strongly autocorrelated even with no underlying
       persistence, which is why the rolling-annual conversion below is not a
       power law.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import integrate, optimize

from ...core.config import FallbackPolicy
from ...core.protocols import DesmoothedResult
from ..base import NowcastTarget
from ..information_set import InformationSet
from ..uncertainty import (
    DEFAULT_MIN_OOS,
    ResidualPool,
    build_residual_pool,
    rolling_origin_backtest,
)
from ._base import RecursiveNowcasterBase

__all__ = [
    "LambdaConvention",
    "StructuralSmoothingNowcaster",
    "StructuralSmoothingWarning",
    "UnsupportedSmoothingModelError",
    "convert_lambda_to_quarterly",
    "infer_fitted_convention",
    "rolling_annual_ar1_autocorrelation",
]

QUARTERS_PER_YEAR = 4
_THETA0_FLOOR = 0.10
_SUPPORTED_METHODS = ("ar1_bayesian", "geltner_classic", "rudin_reparam")

LambdaConvention = Literal[
    "auto", "quarterly", "annual_non_overlapping", "annual_rolling"
]


class StructuralSmoothingWarning(UserWarning):
    """Emitted when borrowed structural parameters are suspect or leaky."""


class UnsupportedSmoothingModelError(NotImplementedError):
    """Raised when the supplied ``DesmoothedResult`` cannot drive a nowcast."""


# ──────────────────────────────────────────────────────────────────
# λ conventions
# ──────────────────────────────────────────────────────────────────


def rolling_annual_ar1_autocorrelation(
    phi: float | np.ndarray, window: int = QUARTERS_PER_YEAR
) -> float | np.ndarray:
    r"""First-order autocorrelation of a rolling sum of an AR(1) process.

    For an AR(1) with parameter :math:`\varphi` and autocovariance
    :math:`\gamma_k = \sigma_x^2 \varphi^{|k|}`, the ``window``-period
    overlapping sum :math:`S_t = \sum_{i=0}^{k-1} x_{t-i}` has

    .. math::
        \rho_1(S) = \frac{\sum_{m=-(k-1)}^{k-1} (k - |m|)\,
                          \varphi^{|m+1|}}
                         {\sum_{m=-(k-1)}^{k-1} (k - |m|)\, \varphi^{|m|}} .

    Parameters
    ----------
    phi
        AR(1) parameter in ``[0, 1)``. Accepts an array.
    window
        Overlap length; ``4`` for rolling four-quarter returns.

    Returns
    -------
    float or np.ndarray
        :math:`\rho_1(S)`.

    Notes
    -----
    At :math:`\varphi = 0` this is :math:`(k-1)/k` — ``0.75`` for ``k=4``. A
    rolling four-quarter return series is 75 % autocorrelated *even when the
    underlying quarterly series is white noise*, purely from the shared
    observations (Working, 1960). That floor is why the mapping from a
    rolling-annual AR(1) coefficient back to the quarterly one is not a power
    law and is poorly conditioned: every quarterly λ in ``[0, 1)`` maps into the
    narrow range ``[0.75, 1)``.

    Verified against simulation to four decimal places for
    :math:`\varphi \in \{0, 0.3, 0.6, 0.8, 0.9\}`.
    """
    phi_arr = np.asarray(phi, dtype=float)
    scalar = phi_arr.ndim == 0
    phi_arr = np.atleast_1d(phi_arr)
    offsets = np.arange(-(window - 1), window)
    weights = (window - np.abs(offsets)).astype(float)
    numerator = (weights * phi_arr[:, None] ** np.abs(offsets + 1)).sum(axis=1)
    denominator = (weights * phi_arr[:, None] ** np.abs(offsets)).sum(axis=1)
    out = numerator / denominator
    return float(out[0]) if scalar else out


def convert_lambda_to_quarterly(
    lam: float,
    convention: str,
    *,
    window: int = QUARTERS_PER_YEAR,
) -> tuple[float, dict[str, Any]]:
    r"""Convert a fitted smoothing parameter to the quarterly convention.

    .. rubric:: ``'quarterly'`` — the identity

    Nothing to do. This is the shipped configuration: ``ar1_bayesian`` fits the
    raw observed series, and the pipeline runner hands it native quarterly
    returns.

    .. rubric:: ``'annual_non_overlapping'`` — :math:`\lambda_q = \lambda_a^{1/4}`

    The quarterly filter :math:`s_t = (1-\lambda_q) r_t + \lambda_q s_{t-1}` has
    the exponentially-weighted representation

    .. math::
        s_t = (1-\lambda_q) \sum_{j \ge 0} \lambda_q^{\,j} r_{t-j} .

    Sampling it every fourth quarter, :math:`S_T = s_{4T}`, and unrolling the
    recursion four steps:

    .. math::
        s_{4T} = \lambda_q^4 s_{4T-4}
               + (1-\lambda_q)\sum_{j=0}^{3} \lambda_q^{\,j} r_{4T-j} .

    The coefficient on the lagged *observation* is therefore
    :math:`\lambda_a = \lambda_q^4`, and the conversion inverts it. The
    decomposition is exact in the λ sense — the remaining weights satisfy
    :math:`(1-\lambda_q)\sum_{j=0}^{3}\lambda_q^{\,j} = 1 - \lambda_q^4 =
    1 - \lambda_a`, as the annual filter requires. What it is *not* exact in is
    the definition of the annual true return: the implied
    :math:`r^{\text{ann}}_T` is the λ-weighted average of the four quarterly
    true returns, not their equal-weighted compound. That is precisely the
    approximation ``ar1_bayesian``'s docstring warns about when it says the
    quarterly AR(1) "does not aggregate exactly to an annual AR(1) with the same
    λ".

    Verified by simulation: true :math:`\lambda_q \in \{0.3, 0.6, 0.85\}`
    recovers :math:`\{0.282, 0.595, 0.849\}`.

    .. rubric:: ``'annual_rolling'`` — numerical inversion, **not** a power law

    If the model was fitted to *rolling* (overlapping) four-quarter returns —
    which is what the parent library's spec recommends — then
    :math:`\lambda_q = \lambda_a^{1/4}` **is badly wrong**, and wrong in the
    dangerous direction. An overlapping sum of an AR(1) is an ARMA(1,3), not an
    AR(1); fitting an AR(1) to it recovers approximately
    :math:`\rho_1` of the overlapping sum (see
    :func:`rolling_annual_ar1_autocorrelation`), which this function inverts
    numerically.

    The magnitude of the error if one used the power law instead:

    .. code-block:: text

        true λ_q = 0.60  →  AR(1) on rolling-annual recovers λ_a ≈ 0.908
                            λ_a^(1/4)             = 0.976   ✗
                            numerical inversion   = 0.600   ✓

    Since desmoothing divides by :math:`(1-\lambda)`, mistaking 0.60 for 0.976
    inflates the recovered volatility by a factor of
    :math:`(1-0.60)/(1-0.976) \approx 17`. This is the single most costly
    arithmetic error available in this module, which is why the power law is not
    offered for the rolling case.

    Parameters
    ----------
    lam
        The fitted smoothing parameter, in ``[0, 1)``.
    convention
        ``'quarterly'``, ``'annual_non_overlapping'`` or ``'annual_rolling'``.
    window
        Quarters per aggregation period; ``4`` for annual.

    Returns
    -------
    lam_quarterly : float
        λ on the quarterly convention.
    details : dict
        ``{'convention', 'lambda_input', 'lambda_quarterly', 'formula',
        'power_law_would_give'}`` — the last entry present for
        ``'annual_rolling'``, so a reviewer can see what was avoided.

    Raises
    ------
    ValueError
        If ``lam`` is outside ``[0, 1)``, if ``convention`` is unknown, or if
        ``'annual_rolling'`` is given a λ below the overlap floor
        :math:`(k-1)/k`, where no quarterly λ can produce it.
    """
    lam = float(lam)
    if not 0.0 <= lam < 1.0:
        raise ValueError(f"lambda must be in [0, 1), got {lam!r}.")

    if convention == "quarterly":
        return lam, {
            "convention": "quarterly",
            "lambda_input": lam,
            "lambda_quarterly": lam,
            "formula": "identity (model fitted at quarterly frequency)",
        }

    if convention == "annual_non_overlapping":
        lam_q = float(lam ** (1.0 / window))
        return lam_q, {
            "convention": "annual_non_overlapping",
            "lambda_input": lam,
            "lambda_quarterly": lam_q,
            "formula": f"lambda_q = lambda_a ** (1/{window})",
        }

    if convention == "annual_rolling":
        floor = (window - 1) / window
        if lam <= floor:
            raise ValueError(
                f"lambda={lam:.4f} is at or below the overlap floor "
                f"{floor:.4f} for a rolling {window}-period sum. A rolling "
                f"{window}-quarter return is {floor:.0%} autocorrelated even "
                "when the underlying quarterly series is white noise "
                "(Working, 1960), so no quarterly lambda can produce this "
                "value. Either the fit was not on rolling-annual data — check "
                "`fitted_convention` — or the AR(1) coefficient is not "
                "identified."
            )

        def residual(phi: float) -> float:
            return float(rolling_annual_ar1_autocorrelation(phi, window)) - lam

        try:
            lam_q = float(
                optimize.brentq(residual, 0.0, 1.0 - 1e-12, xtol=1e-12, rtol=1e-12)
            )
        except (ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"could not invert the rolling-annual autocorrelation for "
                f"lambda={lam:.6f}: {exc}"
            ) from exc
        return lam_q, {
            "convention": "annual_rolling",
            "lambda_input": lam,
            "lambda_quarterly": lam_q,
            "formula": (
                "numerical inversion of rho_1(overlapping sum of AR(1)); "
                "NOT a power law"
            ),
            "power_law_would_give": float(lam ** (1.0 / window)),
            "overlap_floor": floor,
        }

    raise ValueError(
        f"unknown lambda convention {convention!r}; expected 'quarterly', "
        "'annual_non_overlapping' or 'annual_rolling'."
    )


def infer_fitted_convention(result: DesmoothedResult) -> tuple[str, dict[str, Any]]:
    """Infer the frequency a ``DesmoothedResult`` was fitted at, from its index.

    ``DesmoothedResult.true_returns`` is indexed at the native frequency of the
    series passed to ``fit``, which is the only self-describing record of the
    convention.

    Parameters
    ----------
    result
        The fitted desmoothing result.

    Returns
    -------
    convention : str
        ``'quarterly'`` or ``'annual_non_overlapping'``.
    details : dict
        Median spacing in days and the number of observations.

    Raises
    ------
    ValueError
        If the spacing is neither quarterly nor annual. Monthly or daily
        smoothing parameters have no defined mapping to a quarterly appraisal
        filter, and guessing one would be worse than refusing.

    Notes
    -----
    **Rolling-annual fits cannot be detected here.** A rolling four-quarter
    series is indexed *quarterly* — only its values are annual — so it is
    indistinguishable from a quarterly fit by index alone and would be inferred
    as ``'quarterly'``, leaving λ unconverted and far too high. If the
    desmoothing model was fitted on rolling-annual returns (e.g. via
    :func:`private_assets_frequency.utils.time_series.rolling_four_quarter_returns`),
    you **must** pass ``fitted_convention='annual_rolling'`` explicitly. There
    is no way for this module to work it out for you.
    """
    index = result.true_returns.index
    n = len(index)
    if n < 2:
        raise ValueError(
            "cannot infer the fitting convention from a result with "
            f"{n} observation(s)."
        )
    if isinstance(index, pd.PeriodIndex):
        freq = index.freqstr
        if freq.startswith("Q"):
            return "quarterly", {"source": "PeriodIndex", "freqstr": freq, "n": n}
        if freq.startswith(("A", "Y")):
            return "annual_non_overlapping", {
                "source": "PeriodIndex",
                "freqstr": freq,
                "n": n,
            }
        raise ValueError(
            f"fitted result has PeriodIndex freq={freq!r}; only quarterly and "
            "annual fits map onto a quarterly appraisal filter. Pass "
            "`fitted_convention` explicitly if you know the intended meaning."
        )

    spacing = float(np.median(np.diff(np.asarray(index, dtype="datetime64[D]")).astype(float)))
    details = {"source": "DatetimeIndex", "median_spacing_days": spacing, "n": n}
    if 80.0 <= spacing <= 100.0:
        return "quarterly", details
    if 350.0 <= spacing <= 380.0:
        return "annual_non_overlapping", details
    raise ValueError(
        f"median index spacing of {spacing:.0f} days is neither quarterly "
        "(80-100) nor annual (350-380); a smoothing parameter at that "
        "frequency has no defined mapping to a quarterly appraisal filter. "
        "Pass `fitted_convention` explicitly."
    )


# ──────────────────────────────────────────────────────────────────
# Parameter extraction
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _StructuralParams:
    r"""Structural parameters put on a common footing.

    The two supported models are expressed in one form,

    .. math::
        \mathbb{E}[s_Q] = a + \theta_0 \sum_i \beta_i F_{i,Q}
                          + \sum_{j=1}^{q} \theta_j s_{Q-j},

    which for ``ar1_bayesian`` means :math:`a = \theta_0 \alpha`,
    :math:`\theta_0 = 1-\lambda` and :math:`\theta_1 = \lambda`.

    Attributes
    ----------
    method
        Source model's ``diagnostics['method']``.
    theta_0
        Weight on the contemporaneous true return, :math:`1 - \sum_j \theta_j`.
    theta_high
        :math:`(\theta_1, \dots, \theta_q)`, the weights on lagged *reported*
        returns.
    beta
        Factor loadings on the **unsmoothed** return.
    alpha_true
        Intercept of the unsmoothed return, :math:`\alpha`.
    intercept_reported
        Intercept of the reported-return equation, :math:`\theta_0 \alpha`.
    sigma_eps
        Idiosyncratic volatility of the unsmoothed return, where the source
        model reports one.
    lambda_quarterly
        For AR(1)-form models, :math:`\lambda = 1 - \theta_0` after conversion.
    conversion
        Output of :func:`convert_lambda_to_quarterly`, or ``None``.
    """

    method: str
    theta_0: float
    theta_high: np.ndarray
    beta: dict[str, float]
    alpha_true: float
    intercept_reported: float
    sigma_eps: float | None
    lambda_quarterly: float | None
    conversion: dict[str, Any] | None

    @property
    def n_lags(self) -> int:
        return int(self.theta_high.size)

    @property
    def stable(self) -> bool:
        return abs(self.theta_0) >= _THETA0_FLOOR


def _extract_params(
    result: DesmoothedResult,
    *,
    fitted_convention: str,
) -> _StructuralParams:
    """Read structural parameters out of a ``DesmoothedResult``.

    Raises
    ------
    UnsupportedSmoothingModelError
        For factor-free desmoothers, or any method this module does not know
        how to evaluate.
    """
    if not isinstance(result, DesmoothedResult):
        raise TypeError(
            "desmoothed must be a DesmoothedResult (the NamedTuple returned by "
            f"SmoothingModel.fit), got {type(result).__name__}."
        )
    method = str(result.diagnostics.get("method", "unknown"))
    params = result.smoothing_params

    if method == "okunev_white":
        raise UnsupportedSmoothingModelError(
            "StructuralSmoothingNowcaster cannot use an okunev_white result: it "
            "is a factor-free desmoother, so there is no alpha + beta*F_Q "
            "systematic component to build a nowcast from. Use "
            "SmoothingRegressionNowcaster, which estimates the reduced form "
            "directly from the reported series and public factors."
        )
    if method not in _SUPPORTED_METHODS:
        raise UnsupportedSmoothingModelError(
            f"StructuralSmoothingNowcaster does not support "
            f"diagnostics['method']={method!r}; supported: "
            f"{list(_SUPPORTED_METHODS)}. Use SmoothingRegressionNowcaster for "
            "the model-free reduced form."
        )

    beta = params.get("beta")
    if not isinstance(beta, dict) or not beta:
        raise UnsupportedSmoothingModelError(
            f"{method} result has no factor loadings in "
            "smoothing_params['beta']; a structural nowcast needs them."
        )
    beta = {str(k): float(v) for k, v in beta.items()}
    conversion: dict[str, Any] | None = None
    lam_q: float | None = None

    if method in ("ar1_bayesian", "geltner_classic"):
        if "lambda" not in params:
            raise UnsupportedSmoothingModelError(
                f"{method} result has no smoothing_params['lambda']."
            )
        lam_fitted = float(params["lambda"])
        lam_q, conversion = convert_lambda_to_quarterly(
            lam_fitted, fitted_convention
        )
        alpha_true = float(params.get("alpha", 0.0))
        theta_0 = 1.0 - lam_q
        return _StructuralParams(
            method=method,
            theta_0=theta_0,
            theta_high=np.array([lam_q], dtype=float),
            beta=beta,
            alpha_true=alpha_true,
            # ar1_bayesian regresses (s_t - λ s_{t-1})/(1-λ) on [F_t, 1], so its
            # alpha is already on the unsmoothed scale; the reported-equation
            # intercept is therefore θ₀·α.
            intercept_reported=theta_0 * alpha_true,
            sigma_eps=(
                float(params["sigma_eps"]) if "sigma_eps" in params else None
            ),
            lambda_quarterly=lam_q,
            conversion=conversion,
        )

    # rudin_reparam
    theta = np.asarray(params["theta"], dtype=float).reshape(-1)
    if theta.size < 1:
        raise UnsupportedSmoothingModelError(
            "rudin_reparam result has an empty theta vector."
        )
    theta_0 = float(theta[0])
    theta_high = theta[1:].copy()
    if theta_high.size == 1:
        # Q=1 is the AR(1) special case, so the λ conversion applies verbatim.
        lam_q, conversion = convert_lambda_to_quarterly(
            float(theta_high[0]), fitted_convention
        )
        theta_high = np.array([lam_q], dtype=float)
        theta_0 = 1.0 - lam_q
    elif fitted_convention != "quarterly":
        raise ValueError(
            f"rudin_reparam was fitted with {theta_high.size} lags on the "
            f"{fitted_convention!r} convention. The quarterly conversion is "
            "derived for the AR(1) form (one lag); for q >= 2 the mapping from "
            "an annual theta vector to a quarterly one is not identified by the "
            "lag-1 coefficient alone, and this module will not invent one. "
            "Re-fit rudin_reparam on quarterly returns."
        )
    # rudin_reparam reports params['alpha'] as the *raw OLS intercept* of its
    # Eq. 4 — the reported-return equation — while reporting beta already
    # unwound as c_i/θ₀. The two are therefore on different scales in that
    # result object. Here the reported-equation intercept is taken as given and
    # the unsmoothed-scale intercept is derived as α = a/θ₀, consistent with its
    # own beta unwinding.
    intercept_reported = float(params.get("alpha", 0.0))
    alpha_true = (
        intercept_reported / theta_0 if abs(theta_0) >= _THETA0_FLOOR else float("nan")
    )
    return _StructuralParams(
        method=method,
        theta_0=theta_0,
        theta_high=theta_high,
        beta=beta,
        alpha_true=alpha_true,
        intercept_reported=intercept_reported,
        sigma_eps=None,
        lambda_quarterly=lam_q,
        conversion=conversion,
    )


# ──────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────


@dataclass
class StructuralSmoothingNowcaster(RecursiveNowcasterBase):
    r"""Nowcast using an already-fitted desmoothing model's parameters.

    Parameters
    ----------
    desmoothed
        The fitted :class:`~private_assets_frequency.core.protocols.DesmoothedResult`.
        Its ``diagnostics['method']`` selects the formula.
    fitted_convention
        ``'auto'`` (default) infers the fitting frequency from
        ``desmoothed.true_returns.index`` via :func:`infer_fitted_convention`.
        Pass ``'annual_rolling'`` explicitly if the model was fitted on rolling
        four-quarter returns — that case is *not* detectable from the index, and
        getting it wrong is the costliest error available here. See
        :func:`convert_lambda_to_quarterly`.
    integrate_lambda_posterior
        For ``ar1_bayesian``, draw λ from its grid posterior for each predictive
        draw instead of using the posterior mean, so parameter uncertainty
        enters the interval. Requires
        ``posterior_summary['lambda']['grid']`` and ``['density']``.
    smoother
        Optional fitted ``AR1BayesianSmoother`` instance. When given alongside
        ``integrate_lambda_posterior=True``, joint (λ, β, α) posterior draws are
        taken via its ``sample_posterior``, which is materially better than
        λ-only draws — see Notes.
    target
        ``'reported'`` (default) or ``'true'``.
    interval_method
        ``'empirical_recursive'`` (default) or ``'empirical_direct'``.
    horizons
        Horizons to build residual pools for.
    min_train
        Smallest training window in the internal backtest. Lower than the
        smoothing regression's 40 because no parameters are estimated at each
        origin, so there is no estimation-error regime to let settle.
    min_oos
        Out-of-sample errors required before a horizon's pool is used directly.
    fallback_policy
        ``'warn'`` (default), ``'strict'`` or ``'auto'``.

    Attributes
    ----------
    name
        ``'structural_smoothing'``.

    Notes
    -----
    **λ-only posterior draws understate parameter uncertainty, and do so in a
    biased way.** The parent library's own diagnostics flag the λ-β confound as
    the central identification weakness of the Bayesian AR(1) fit: β moves
    systematically as λ moves across the grid. Drawing λ while holding β at its
    marginal posterior mean therefore moves λ *without* the compensating β move,
    which does not sample the posterior ridge — it cuts across it. Pass
    ``smoother`` to get joint draws, or treat the widened interval as
    indicative only.

    Examples
    --------
    >>> from private_assets_frequency.desmoothing.ar1_bayesian import (  # doctest: +SKIP
    ...     AR1BayesianSmoother,
    ... )
    >>> smoother = AR1BayesianSmoother()                              # doctest: +SKIP
    >>> result = smoother.fit(quarterly_returns, quarterly_factors, priors)
    ...                                                               # doctest: +SKIP
    >>> model = StructuralSmoothingNowcaster(desmoothed=result).fit(info)
    ...                                                               # doctest: +SKIP
    """

    desmoothed: DesmoothedResult | None = None
    fitted_convention: str = "auto"
    integrate_lambda_posterior: bool = False
    smoother: Any = None
    target: NowcastTarget = "reported"
    interval_method: str = "empirical_recursive"
    horizons: tuple[int, ...] = (1, 2)
    min_train: int = 12
    min_oos: int = DEFAULT_MIN_OOS
    fallback_policy: FallbackPolicy | str = FallbackPolicy.WARN
    name: str = field(default="structural_smoothing", init=False)

    _params: _StructuralParams | None = field(default=None, init=False, repr=False)
    _lambda_draws: np.ndarray | None = field(default=None, init=False, repr=False)
    _beta_draws: np.ndarray | None = field(default=None, init=False, repr=False)
    _alpha_draws: np.ndarray | None = field(default=None, init=False, repr=False)
    _draw_factor_names: tuple[str, ...] = field(
        default=(), init=False, repr=False
    )

    # ── Accessors ──────────────────────────────────────────────────

    @property
    def structural_params(self) -> _StructuralParams:
        """The extracted structural parameters, on the quarterly convention."""
        if self._params is None:
            raise RuntimeError(f"{self.name}: call fit() first.")
        return self._params

    @property
    def lambda_quarterly(self) -> float | None:
        """λ on the quarterly convention, or ``None`` for a multi-lag Rudin fit."""
        return self.structural_params.lambda_quarterly

    # ── Subclass contract ──────────────────────────────────────────

    def _lag_depth(self) -> int:
        return self.structural_params.n_lags

    def _systematic(self, info: InformationSet, quarter: pd.Period) -> float:
        r""":math:`\sum_i \beta_i F_{i,Q}`, using only factors the fit knows."""
        params = self.structural_params
        factors = info.factor_row(quarter, require_complete=True)
        total = 0.0
        for col, beta in params.beta.items():
            if col in factors.index:
                total += beta * float(factors[col])
        return total

    def _conditional_mean(
        self, info: InformationSet, quarter: pd.Period, history: np.ndarray
    ) -> np.ndarray:
        params = self.structural_params
        hist = np.asarray(history, dtype=float)
        n = hist.shape[0]

        if self._lambda_draws is not None:
            # Parameter uncertainty: evaluate the formula at each drawn (λ, β, α).
            return self._conditional_mean_with_draws(info, quarter, hist)

        systematic = self._systematic(info, quarter)
        mean = params.intercept_reported + params.theta_0 * systematic
        if params.n_lags:
            return mean + hist @ params.theta_high
        return np.full(n, mean)

    def _conditional_mean_with_draws(
        self, info: InformationSet, quarter: pd.Period, hist: np.ndarray
    ) -> np.ndarray:
        r"""Conditional mean evaluated per draw of :math:`(\lambda, \beta, \alpha)`."""
        n = hist.shape[0]
        # Only reached from `_conditional_mean` after it checks the λ draws exist,
        # and `_prepare_posterior_draws` sets β and α together or not at all.
        # Asserted rather than left implicit: the invariant spans two methods.
        assert self._lambda_draws is not None
        lam = self._resize_draws(self._lambda_draws, n)
        theta_0 = 1.0 - lam
        factors = info.factor_row(quarter, require_complete=True)

        if self._beta_draws is not None:
            assert self._alpha_draws is not None
            cols = self._draw_factor_names
            f_vec = np.array(
                [float(factors[c]) if c in factors.index else 0.0 for c in cols],
                dtype=float,
            )
            beta = self._resize_draws(self._beta_draws, n)
            systematic = beta @ f_vec
            alpha = self._resize_draws(self._alpha_draws, n)
        else:
            systematic = np.full(n, self._systematic(info, quarter))
            alpha = np.full(n, self.structural_params.alpha_true)

        mean = theta_0 * (alpha + systematic)
        return mean + lam * hist[:, 0]

    @staticmethod
    def _resize_draws(draws: np.ndarray, n: int) -> np.ndarray:
        """Tile or truncate a draw array to ``n`` rows, deterministically."""
        arr = np.asarray(draws)
        if arr.shape[0] == n:
            return arr
        reps = int(np.ceil(n / arr.shape[0]))
        tiled = np.concatenate([arr] * reps, axis=0)
        return tiled[:n]

    def _true_target_centre(
        self, info: InformationSet, quarter: pd.Period
    ) -> float:
        params = self.structural_params
        if not params.stable:
            raise ValueError(
                f"{self.name}: target='true' needs theta_0 to rescale the "
                f"reduced form, but theta_0 = {params.theta_0:.4f} is below the "
                f"stability floor {_THETA0_FLOOR}."
            )
        return float(params.alpha_true + self._systematic(info, quarter))

    def _true_target_scale(self) -> float:
        params = self.structural_params
        if not params.stable:
            raise ValueError(
                f"{self.name}: theta_0 = {params.theta_0:.4f} is below the "
                f"stability floor {_THETA0_FLOOR}; the idiosyncratic scale "
                "e/theta_0 is not meaningful."
            )
        return float(params.theta_0)

    # ── Fitting ────────────────────────────────────────────────────

    def _fit_impl(
        self, info: InformationSet
    ) -> tuple[ResidualPool, dict[str, Any]]:
        if self.desmoothed is None:
            raise ValueError(
                f"{self.name} requires `desmoothed` — a DesmoothedResult from a "
                "fitted SmoothingModel. It reuses that model's parameters "
                "rather than estimating its own."
            )
        if not isinstance(self.desmoothed, DesmoothedResult):
            # Checked here, before convention inference reads `.true_returns`,
            # so a wrong type gives a useful message rather than an AttributeError.
            raise TypeError(
                "desmoothed must be a DesmoothedResult (the NamedTuple returned "
                f"by SmoothingModel.fit), got {type(self.desmoothed).__name__}."
            )
        if self.interval_method not in ("empirical_recursive", "empirical_direct"):
            raise ValueError(
                f"{self.name}: unknown interval_method {self.interval_method!r}; "
                "expected 'empirical_recursive' or 'empirical_direct'. "
                "Parametric intervals need a design matrix this model does not "
                "estimate; use integrate_lambda_posterior for parameter "
                "uncertainty instead."
            )

        convention = self.fitted_convention
        inferred: dict[str, Any] | None = None
        if convention == "auto":
            convention, inferred = infer_fitted_convention(self.desmoothed)
        params = _extract_params(self.desmoothed, fitted_convention=convention)
        self._params = params
        self._lambda_draws = None
        self._beta_draws = None
        self._alpha_draws = None
        self._draw_factor_names = ()

        self._check_fit_span(info)
        self._check_factor_coverage(info)
        if not params.stable:
            self._flag(
                f"{self.name}: theta_0 = {params.theta_0:.4f} is below the "
                f"stability floor {_THETA0_FLOOR} (implied smoothing weight on "
                f"past reports {1 - params.theta_0:.3f}). Reported-return "
                "nowcasts remain usable; target='true' is not.",
                StructuralSmoothingWarning,
            )
        if self.integrate_lambda_posterior:
            self._prepare_posterior_draws()

        design = self._build_design(info)
        oos, backtest_diag = self._backtest(design)
        pool = build_residual_pool(
            oos,
            design["residuals"],
            # Zero fitted parameters here: everything was estimated elsewhere,
            # so the sqrt(1 + p/n) in-sample inflation has nothing to correct.
            n_params=0,
            horizons=tuple(sorted(set(int(h) for h in self.horizons))),
            min_oos=self.min_oos,
            warn=self._policy is not FallbackPolicy.AUTO,
            label=self.name,
        )

        diagnostics: dict[str, Any] = {
            "n_train": int(design["y"].size),
            "n_params": 0,
            "features": ["const", "theta_0*beta.F"]
            + [f"s_lag{j}" for j in range(1, params.n_lags + 1)],
            "source_method": params.method,
            "fitted_convention": convention,
            "convention_inferred": inferred,
            "lambda_conversion": params.conversion,
            "lambda_quarterly": params.lambda_quarterly,
            "theta_0": params.theta_0,
            "theta_high": params.theta_high.tolist(),
            "beta": dict(params.beta),
            "alpha_true": params.alpha_true,
            "intercept_reported": params.intercept_reported,
            "sigma_eps": params.sigma_eps,
            "stable": params.stable,
            "structural_fit_span": design["fit_span"],
            "integrate_lambda_posterior": bool(self.integrate_lambda_posterior),
            "posterior_draw_mode": (
                "joint"
                if self._beta_draws is not None
                else ("lambda_marginal" if self._lambda_draws is not None else "none")
            ),
            "in_sample_rmse": float(
                np.sqrt(np.mean(np.square(design["residuals"])))
            ),
            "horizons": tuple(sorted(set(int(h) for h in self.horizons))),
            "min_train": int(design["min_train"]),
            "backtest": backtest_diag,
            "training_quarters": (
                str(design["index"][0]),
                str(design["index"][-1]),
            ),
            "feature_budget": {
                "n_features": 0,
                "n_train": int(design["y"].size),
                "budget": float(design["y"].size) / 5.0,
                "within_budget": True,
            },
        }
        return pool, diagnostics

    # ── Fit-time checks ────────────────────────────────────────────

    def _fit_quarters(self) -> pd.PeriodIndex | None:
        """The borrowed fit's index expressed as quarters, or ``None`` if it cannot be."""
        assert self.desmoothed is not None
        index = self.desmoothed.true_returns.index
        try:
            return pd.PeriodIndex(
                index if isinstance(index, pd.PeriodIndex) else pd.DatetimeIndex(index),
                freq="Q",
            )
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def _fit_span_quarters(self) -> tuple[str, str]:
        """``(first, last)`` quarter the borrowed parameters were fitted over.

        Reported as quarter labels rather than raw index values so it is directly
        comparable to ``info.last_published_quarter`` — which is the comparison a
        reader of the diagnostics will want to make.
        """
        span = self._fit_quarters()
        if span is None:  # pragma: no cover - defensive
            index = self.desmoothed.true_returns.index  # type: ignore[union-attr]
            return (str(index[0]), str(index[-1]))
        return (str(span.min()), str(span.max()))

    def _check_fit_span(self, info: InformationSet) -> None:
        """Warn if the borrowed parameters saw data the information set has not."""
        span = self._fit_quarters()
        if span is None:
            return
        last_fit = span.max()
        last_published = info.last_published_quarter
        if last_published is not None and last_fit > last_published:
            self._flag(
                f"{self.name}: the supplied DesmoothedResult was fitted through "
                f"{last_fit}, but only {last_published} is published at "
                f"as_of={info.as_of.date()}. The borrowed parameters therefore "
                "saw data outside this information set, so any backtest built "
                "on them is leaky and its errors are optimistic. This is "
                "legitimate only for a production nowcast at today's as_of; for "
                "evaluation, re-fit the desmoothing model inside each "
                "information set. The span is recorded in "
                "diagnostics['structural_fit_span'].",
                StructuralSmoothingWarning,
            )

    def _check_factor_coverage(self, info: InformationSet) -> None:
        """Warn when the fit's factors and the information set's do not line up."""
        params = self.structural_params
        available = set(info.factor_columns)
        fitted = set(params.beta)
        missing = sorted(fitted - available)
        extra = sorted(available - fitted)
        if missing:
            self._flag(
                f"{self.name}: the fitted model has loadings on {missing} which "
                "are not in this information set's factor panel; those terms "
                "are dropped, so the systematic component is incomplete and the "
                "nowcast is biased toward the intercept.",
                StructuralSmoothingWarning,
            )
        if extra:
            self._flag(
                f"{self.name}: the information set carries factors {extra} that "
                "the fitted model has no loading for; they are ignored. This is "
                "expected when the panel is wider than the strategy's factor "
                "set, but check it is not a naming mismatch.",
                StructuralSmoothingWarning,
            )
        if not fitted & available:
            raise ValueError(
                f"{self.name}: none of the fitted factor loadings "
                f"{sorted(fitted)} appear in the information set's factor "
                f"columns {sorted(available)}. The systematic component would be "
                "identically zero."
            )

    def _prepare_posterior_draws(self) -> None:
        """Set up λ (and optionally β, α) draws for the predictive distribution."""
        assert self.desmoothed is not None
        params = self.structural_params
        if params.method != "ar1_bayesian":
            self._flag(
                f"{self.name}: integrate_lambda_posterior=True is only "
                f"meaningful for ar1_bayesian, which reports a lambda grid "
                f"posterior; the supplied result is {params.method!r}. "
                "Continuing with point parameters.",
                StructuralSmoothingWarning,
            )
            return

        if self.smoother is not None and hasattr(self.smoother, "sample_posterior"):
            draws = self.smoother.sample_posterior(
                2000, rng=np.random.default_rng(0)
            )
            lam = np.asarray(draws["lambda"], dtype=float)
            if params.conversion is not None and (
                params.conversion["convention"] != "quarterly"
            ):
                lam = np.array(
                    [
                        convert_lambda_to_quarterly(
                            float(v), params.conversion["convention"]
                        )[0]
                        for v in lam
                    ]
                )
            self._lambda_draws = lam
            self._beta_draws = np.asarray(draws["beta"], dtype=float)
            self._alpha_draws = np.asarray(draws["alpha"], dtype=float)
            self._draw_factor_names = tuple(
                str(x) for x in np.asarray(draws["factor_names"]).reshape(-1)
            )
            return

        lam_summary = self.desmoothed.posterior_summary.get("lambda", {})
        grid = lam_summary.get("grid")
        density = lam_summary.get("density")
        if grid is None or density is None:
            self._flag(
                f"{self.name}: integrate_lambda_posterior=True but the result's "
                "posterior_summary['lambda'] has no 'grid'/'density'. "
                "Continuing with the posterior mean.",
                StructuralSmoothingWarning,
            )
            return

        grid = np.asarray(grid, dtype=float)
        density = np.asarray(density, dtype=float)
        cdf = integrate.cumulative_trapezoid(density, grid, initial=0.0)
        if cdf[-1] <= 0:  # pragma: no cover - defensive
            self._flag(
                f"{self.name}: the lambda posterior density integrates to "
                f"{cdf[-1]:.3g}; cannot sample it. Continuing with the "
                "posterior mean.",
                StructuralSmoothingWarning,
            )
            return
        cdf = cdf / cdf[-1]
        u = np.linspace(0.5 / 2000.0, 1.0 - 0.5 / 2000.0, 2000)
        lam = np.interp(u, cdf, grid)
        if params.conversion is not None and (
            params.conversion["convention"] != "quarterly"
        ):
            lam = np.array(
                [
                    convert_lambda_to_quarterly(
                        float(v), params.conversion["convention"]
                    )[0]
                    for v in lam
                ]
            )
        self._lambda_draws = lam
        self._flag(
            f"{self.name}: integrating over the lambda grid posterior with beta "
            "and alpha held at their marginal posterior means. The parent "
            "library flags the lambda-beta confound as this model's central "
            "identification weakness, so moving lambda without the compensating "
            "beta move cuts across the posterior ridge rather than along it — "
            "the widened interval is indicative, not calibrated. Pass "
            "`smoother=<fitted AR1BayesianSmoother>` for joint draws.",
            StructuralSmoothingWarning,
        )

    # ── Design and backtest ────────────────────────────────────────

    def _build_design(self, info: InformationSet) -> dict[str, Any]:
        """Replay rows for the backtest: deterministic part plus lag history."""
        params = self.structural_params
        q = params.n_lags
        s = info.reported.astype(float)
        published = pd.PeriodIndex(s.index, freq="Q")
        grid = pd.period_range(published[0], published[-1], freq="Q")
        s_grid = s.reindex(grid)

        cols = [c for c in params.beta if c in info.factor_columns]
        complete = info.factor_quarter_complete.reindex(grid).fillna(False).to_numpy()
        f_grid = info.factors_quarterly.reindex(grid)[cols]
        mask = pd.DataFrame(
            np.repeat(complete[:, None], len(cols), axis=1), index=grid, columns=cols
        )
        f_grid = f_grid.where(mask)
        systematic = sum(
            params.beta[c] * f_grid[c] for c in cols
        )
        exog = params.intercept_reported + params.theta_0 * systematic

        frame = pd.DataFrame({"y": s_grid, "exog": exog})
        for j in range(1, q + 1):
            frame[f"s_lag{j}"] = s_grid.shift(j)
        keep = frame.notna().all(axis=1) & frame.index.isin(published)
        frame = frame.loc[keep]
        if len(frame) < 3:
            raise ValueError(
                f"{self.name}: only {len(frame)} usable rows at "
                f"as_of={info.as_of.date()}; need at least 3 to replay the "
                "structural nowcast."
            )

        lag_names = [f"s_lag{j}" for j in range(1, q + 1)]
        lags = frame[lag_names].to_numpy(dtype=float).reshape(len(frame), q)
        y = frame["y"].to_numpy(dtype=float)
        exog_arr = frame["exog"].to_numpy(dtype=float)
        fitted = exog_arr + (lags @ params.theta_high if q else 0.0)

        return {
            "index": pd.PeriodIndex(frame.index, freq="Q"),
            "y": y,
            "exog": exog_arr,
            "lags": lags,
            "residuals": y - fitted,
            "min_train": max(1, min(self.min_train, len(frame) - 1)),
            "fit_span": self._fit_span_quarters(),
            "factor_columns_used": tuple(cols),
        }

    def _backtest(
        self, design: dict[str, Any]
    ) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        """Replay the fixed-parameter nowcast recursively at every origin.

        Unlike the smoothing regression's backtest, nothing is re-estimated —
        the parameters came from elsewhere and are held fixed. These errors
        therefore measure specification error only; see the module docstring on
        ``structural_fit_span`` for why that is recorded rather than glossed.
        """
        params = self.structural_params
        q = params.n_lags
        exog = design["exog"]
        lags = design["lags"]
        n = design["y"].size

        def predict_fn(n_train: int, hs: Sequence[int]) -> dict[int, float]:
            history = lags[n_train].copy() if q else np.empty(0)
            out: dict[int, float] = {}
            for h in sorted(hs):
                row = n_train + h - 1
                if row >= n:
                    break
                pred = float(exog[row])
                if q:
                    pred += float(history @ params.theta_high)
                out[h] = pred
                if q:
                    history = np.concatenate([[pred], history[: q - 1]])
            return out

        return rolling_origin_backtest(
            design["y"],
            min_train=design["min_train"],
            horizons=tuple(sorted(set(int(h) for h in self.horizons))),
            predict_fn=predict_fn,
        )
