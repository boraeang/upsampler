r"""
Rolling-origin evaluation harness: nested selection, Diebold-Mariano, subperiods.

This module answers one question — *what would a user actually have obtained in
real time?* — and it is built to resist the three ways that question gets
answered too generously (``DESIGN_NOTES.md``). Concretely:

**The harness owns the data; models own an information set.** The full
``reported`` series, the full vintage table and the full factor panel live here
and are never handed to a model. At each evaluation date the harness builds an
:class:`~private_assets_frequency.nowcast.information_set.InformationSet` and
passes only that.

**Selection is nested, and the harness verifies it rather than trusting it.**
Hyperparameters are chosen inside each model's ``fit(info)``, on an inner
rolling-origin split over *published* quarters. Because an information set
contains no unpublished quarter, the outer evaluation quarter cannot enter that
split — and :func:`_assert_nested_selection` checks the model's own recorded
selection index to confirm it, raising :class:`NestedSelectionError` if a
future edit to a model ever breaks the property.

**Overlapping evaluation rows are not independent observations.** The optional
grid of several ``as_of`` dates per quarter produces several nowcasts of the
*same* target. Those are a reporting device for showing how a nowcast evolves as
data arrives, never an observation-count multiplier: headline metrics are
computed on a single ``primary_offset_days`` series, and the Diebold-Mariano test
uses HAC (Newey-West) variance with the Harvey-Leybourne-Newbold small-sample
correction.

What gets reported
------------------
Per model × horizon × subperiod: ``n``, RMSE, MAE, mean error (bias), two
directional hit rates, 80 % and 95 % interval coverage, mean interval width, and
a Diebold-Mariano test against the benchmark. Subperiods are the full sample plus
the GFC, COVID and 2022 windows. If the challenger does not beat the benchmark,
the report says so in :meth:`EvaluationResult.summary` — that is a valid and
useful result, and the spec is explicit that it should be stated plainly.

References
----------
.. [1] Diebold & Mariano (1995) — "Comparing predictive accuracy,"
       *Journal of Business & Economic Statistics*.
.. [2] Harvey, Leybourne & Newbold (1997) — "Testing the equality of prediction
       mean squared errors," *International Journal of Forecasting*. The
       small-sample correction, which matters at ~70 evaluation points.
.. [3] Newey & West (1987) — "A simple, positive semi-definite,
       heteroskedasticity and autocorrelation consistent covariance matrix."
.. [4] Tashman (2000) — "Out-of-sample tests of forecasting accuracy."
.. [5] Cohen, Mantoan, Nesheim, de Paula, Turrell & Yang (2023) —
       "Nowcasting using regression on signatures," arXiv:2305.10256v3,
       Figures 8-9 (nowcast evolution within a quarter).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.config import AssetClassConfig, FallbackPolicy
from .information_set import (
    EvaluationTarget,
    InformationSet,
    build_information_set,
    first_print_values,
    latest_values,
)

__all__ = [
    "DEFAULT_SUBPERIODS",
    "EvaluationData",
    "EvaluationResult",
    "EvaluationWarning",
    "NestedSelectionError",
    "diebold_mariano",
    "newey_west_variance",
    "plot_error_by_horizon",
    "plot_nowcast_evolution",
    "plot_nowcast_vs_realised",
    "rolling_origin_evaluation",
]


DEFAULT_SUBPERIODS: dict[str, tuple[str, str]] = {
    "gfc": ("2008Q3", "2009Q2"),
    "covid": ("2020Q1", "2020Q4"),
    "2022": ("2022Q1", "2022Q4"),
}
"""Stress windows reported alongside the full sample, per the module spec."""

DEFAULT_MIN_TRAIN_QUARTERS = 40
"""Minimum published history before an evaluation date is used."""

_BENCHMARK = "smoothing_regression"


class NestedSelectionError(AssertionError):
    """Raised when a model's hyperparameter selection saw an evaluation quarter."""


class EvaluationWarning(UserWarning):
    """Emitted when a model contributes nothing to the report."""


# ──────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationData:
    r"""The full dataset the harness owns and models never see.

    Parameters
    ----------
    reported
        Final revised reported quarterly returns. Used to build information sets
        in lag-rule mode, to backfill pre-vintage history, and — when
        ``vintages`` is absent — as the realisation to score against.
    public_factors
        Daily or monthly public factor returns.
    vintages
        Long frame ``[quarter, release_date, value]``. Supplying it selects
        vintage mode and makes ``evaluation_target='first_print'`` available.
    early_signals
        Optional irregular proxy series.
    publication_lag
        Days after quarter end at which a quarter is assumed published
        (lag-rule mode, and vintage backfill).
    evaluation_target
        ``'first_print'`` (default in vintage mode) or ``'latest'``. Which
        realisation nowcasts are scored against — see
        :mod:`..information_set` on why this is not a cosmetic choice.
    factor_columns, config
        Factor selection, passed to :func:`build_information_set`.
    """

    reported: pd.Series
    public_factors: pd.DataFrame
    vintages: pd.DataFrame | None = None
    early_signals: pd.DataFrame | None = None
    publication_lag: int | float | pd.Timedelta | None = None
    evaluation_target: EvaluationTarget | None = None
    factor_columns: Sequence[str] | None = None
    config: AssetClassConfig | None = None

    @property
    def quarters(self) -> pd.PeriodIndex:
        """Quarters the reported series covers."""
        index = self.reported.index
        if isinstance(index, pd.PeriodIndex):
            return index
        return pd.PeriodIndex(index, freq="Q")

    def realisations(self) -> pd.Series:
        r"""The series nowcasts are scored against.

        Returns
        -------
        pd.Series
            First prints in vintage mode with
            ``evaluation_target='first_print'``, latest revisions otherwise.

        Notes
        -----
        This reads the **untruncated** vintage table. It is the realisation, not
        an input — it must never reach a model, and it does not: the harness
        calls this after prediction, for scoring only.
        """
        if self.vintages is not None:
            target = self.evaluation_target or "first_print"
            if target == "first_print":
                return first_print_values(self.vintages)
            return latest_values(self.vintages)
        series = self.reported.copy()
        series.index = self.quarters
        return series.rename("latest")

    def information_set(
        self, as_of: pd.Timestamp, **overrides: Any
    ) -> InformationSet:
        """Build the information set at ``as_of`` from this dataset."""
        params: dict[str, Any] = dict(
            reported=self.reported,
            public_factors=self.public_factors,
            as_of=as_of,
            vintages=self.vintages,
            publication_lag=self.publication_lag,
            early_signals=self.early_signals,
            factor_columns=self.factor_columns,
            config=self.config,
            evaluation_target=self.evaluation_target,
        )
        params.update(overrides)
        return build_information_set(**params)


# ──────────────────────────────────────────────────────────────────
# Diebold-Mariano
# ──────────────────────────────────────────────────────────────────


def newey_west_variance(x: np.ndarray, lags: int) -> float:
    r"""Newey-West HAC variance of the sample mean of ``x``.

    .. math::
        \widehat{\mathrm{Var}}(\bar x) = \frac{1}{n}\left(
            \hat\gamma_0 + 2\sum_{j=1}^{L}
            \left(1 - \frac{j}{L+1}\right)\hat\gamma_j \right)

    with Bartlett weights, which guarantees a non-negative estimate.

    Parameters
    ----------
    x
        The series whose mean's variance is wanted.
    lags
        Truncation lag ``L``. ``0`` gives the i.i.d. variance.

    Returns
    -------
    float
        Variance of the sample mean. Zero when ``x`` has fewer than two
        observations.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    n = arr.size
    if n < 2:
        return 0.0
    if lags < 0:
        raise ValueError(f"lags must be >= 0, got {lags}.")
    centred = arr - arr.mean()
    total = float(centred @ centred) / n
    for j in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - j / (lags + 1.0)
        gamma = float(centred[j:] @ centred[:-j]) / n
        total += 2.0 * weight * gamma
    return float(max(total, 0.0) / n)


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    horizon: int = 1,
    loss: str = "squared",
    lags: int | None = None,
    small_sample_correction: bool = True,
) -> dict[str, Any]:
    r"""Diebold-Mariano test of equal predictive accuracy, with HAC variance.

    Tests :math:`H_0: \mathbb{E}[d_t] = 0` for the loss differential
    :math:`d_t = L(e^a_t) - L(e^b_t)`. A **negative** statistic favours model
    ``a`` (lower loss).

    Parameters
    ----------
    errors_a, errors_b
        Paired forecast errors, same length and same evaluation points.
    horizon
        Forecast horizon ``h``. Drives the default truncation lag and the
        small-sample correction.
    loss
        ``'squared'`` (default) or ``'absolute'``.
    lags
        HAC truncation lag. ``None`` uses
        ``max(h - 1, floor(4 (n/100)^(2/9)))`` — the Diebold-Mariano
        prescription for ``h``-step forecasts, floored at the Newey-West
        automatic rule so that revision-induced autocorrelation at ``h=1`` is
        not ignored.
    small_sample_correction
        Apply the Harvey-Leybourne-Newbold correction and refer the statistic
        to :math:`t_{n-1}` rather than the normal. On by default: with ~70
        evaluation points the uncorrected test over-rejects noticeably, and this
        module's whole premise is not over-claiming.

    Returns
    -------
    dict
        ``{'statistic', 'p_value', 'mean_loss_differential', 'n', 'lags',
        'loss', 'horizon', 'favours', 'hln_correction', 'distribution'}``.

    Raises
    ------
    ValueError
        If the error arrays differ in length, or fewer than three paired
        observations survive.

    Notes
    -----
    The test compares *accuracy*, not economic value, and it assumes the
    evaluation points are the same for both models — which the harness enforces
    by pairing on ``(quarter, horizon, as_of)``. A statistic that fails to reject
    is the common outcome at this sample size and should be reported as such
    rather than read as evidence of equality.
    """
    a = np.asarray(errors_a, dtype=float).reshape(-1)
    b = np.asarray(errors_b, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError(
            f"error arrays must be the same length, got {a.size} and {b.size}."
        )
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = a.size
    if n < 3:
        raise ValueError(
            f"Diebold-Mariano needs at least 3 paired observations, got {n}."
        )
    if loss == "squared":
        differential = a**2 - b**2
    elif loss == "absolute":
        differential = np.abs(a) - np.abs(b)
    else:
        raise ValueError(f"loss must be 'squared' or 'absolute', got {loss!r}.")

    if lags is None:
        automatic = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lags = max(horizon - 1, automatic)
    variance = newey_west_variance(differential, lags)
    mean = float(differential.mean())
    if variance <= 0.0:
        statistic = float("nan")
        p_value = float("nan")
        correction = float("nan")
    else:
        statistic = mean / float(np.sqrt(variance))
        correction = 1.0
        if small_sample_correction:
            h = int(horizon)
            factor = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
            correction = float(np.sqrt(max(factor, 1e-12)))
            statistic *= correction
        p_value = float(2.0 * stats.t.sf(abs(statistic), df=max(n - 1, 1)))

    favours: str
    if mean == 0.0:
        # Identical losses are a tie, which is informative; "undetermined" is
        # reserved for a non-zero differential whose variance cannot be
        # estimated. Checked before finiteness because an exactly zero
        # differential has zero variance and so leaves the statistic undefined.
        favours = "tie"
    elif not np.isfinite(statistic):
        favours = "undetermined"
    elif mean < 0:
        favours = "a"
    else:
        favours = "b"

    return {
        "statistic": statistic,
        "p_value": p_value,
        "mean_loss_differential": mean,
        "n": int(n),
        "lags": int(lags),
        "loss": loss,
        "horizon": int(horizon),
        "favours": favours,
        "hln_correction": correction,
        "distribution": f"t({max(n - 1, 1)})"
        if small_sample_correction
        else "normal",
    }


# ──────────────────────────────────────────────────────────────────
# Nested-selection verification — CHECKPOINT 4
# ──────────────────────────────────────────────────────────────────


def _selection_quarters(model: Any) -> set[str]:
    """Every quarter a model's hyperparameter selection actually looked at.

    Reads the model's own ``fit_diagnostics['selection']['detail']``, where each
    candidate configuration records the ``selection_index`` its inner
    rolling-origin split scored over. Returns an empty set for a model that does
    no selection.
    """
    try:
        diagnostics = model.fit_diagnostics
    except (AttributeError, RuntimeError):
        return set()
    selection = diagnostics.get("selection") or {}
    detail = selection.get("detail") or {}
    seen: set[str] = set()
    for entry in detail.values():
        if isinstance(entry, dict):
            seen.update(str(q) for q in entry.get("selection_index", ()))
    return seen


def _assert_nested_selection(
    model: Any,
    info: InformationSet,
    targets: Sequence[pd.Period],
    *,
    model_name: str,
) -> dict[str, Any]:
    """Confirm the outer evaluation quarters never entered the inner split.

    Two independent checks, because they fail in different ways:

    1. **The selection index.** Every quarter scored by the inner rolling-origin
       split must be a published quarter, and in particular none of ``targets``
       may appear. This catches a model that reached outside its information set
       during selection.
    2. **The training span.** The last training quarter must not exceed the last
       published quarter. This catches a model that reached outside it during
       *fitting*, which the selection index would not reveal.

    Parameters
    ----------
    model
        A fitted nowcaster.
    info
        The information set it was fitted on.
    targets
        The outer evaluation quarters about to be predicted.
    model_name
        Name used in the error message.

    Returns
    -------
    dict
        ``{'n_selection_quarters', 'selection_span', 'training_quarters',
        'verified'}`` for the evaluation diagnostics.

    Raises
    ------
    NestedSelectionError
        If either check fails. This is an assertion about the *code*, not about
        the data, so it raises regardless of ``FallbackPolicy``.
    """
    target_labels = {str(q) for q in targets}
    published = {str(q) for q in pd.PeriodIndex(info.reported.index, freq="Q")}
    selection = _selection_quarters(model)

    leaked_targets = selection & target_labels
    if leaked_targets:
        raise NestedSelectionError(
            f"{model_name}: hyperparameter selection at "
            f"as_of={info.as_of.date()} scored quarter(s) "
            f"{sorted(leaked_targets)}, which are the outer evaluation targets. "
            "Selection must use only quarters published at as_of."
        )
    outside = selection - published
    if outside:
        raise NestedSelectionError(
            f"{model_name}: hyperparameter selection at "
            f"as_of={info.as_of.date()} scored quarter(s) {sorted(outside)} "
            "that are not published in this information set."
        )

    diagnostics = getattr(model, "fit_diagnostics", {}) or {}
    span = diagnostics.get("training_quarters")
    last_published = info.last_published_quarter
    if span is not None and last_published is not None:
        if str(span[1]) > str(last_published):
            raise NestedSelectionError(
                f"{model_name}: trained through {span[1]} but only "
                f"{last_published} is published at as_of={info.as_of.date()}."
            )
    return {
        "n_selection_quarters": len(selection),
        "selection_span": (
            (min(selection), max(selection)) if selection else None
        ),
        "training_quarters": span,
        "verified": True,
    }


# ──────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────


def _metrics_for(block: pd.DataFrame) -> dict[str, float]:
    r"""Accuracy and calibration metrics for one block of evaluation rows.

    Returns
    -------
    dict
        ``n``, ``rmse``, ``mae``, ``mean_error`` (bias), ``directional_hit_rate``
        (sign of the *change* from the last published value — the informative
        direction for a heavily autocorrelated series — over the rows where a
        directional call was actually made), ``directional_coverage`` (the
        fraction of rows on which a call was made; ``0`` for a carry-forward,
        which always predicts no change), ``sign_hit_rate`` (sign of the return
        level, reported because it is the conventional figure and is
        near-uninformative for a series with a positive mean), ``coverage_80``,
        ``coverage_95``, ``mean_width_80``.
    """
    if block.empty:
        return {"n": 0}
    error = block["actual"].to_numpy() - block["point"].to_numpy()
    actual = block["actual"].to_numpy()
    point = block["point"].to_numpy()
    s_last = block["s_last"].to_numpy()
    out = {
        "n": int(len(block)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_error": float(np.mean(error)),
        "coverage_80": float(block["covered_80"].mean()),
        "coverage_95": float(block["covered_95"].mean()),
        "mean_width_80": float(
            (block["upper_80"] - block["lower_80"]).mean()
        ),
    }
    change_actual = actual - s_last
    change_point = point - s_last
    # A forecast of *no* change is an abstention, not a wrong call. Scoring it as
    # a miss would hand NaiveCarryForward a hit rate of exactly 0 % by
    # construction — its point forecast is s_last, so its predicted change is
    # identically zero. Rows without a directional call are excluded from the
    # rate and counted in `directional_coverage` instead.
    called = (np.abs(change_actual) > 0) & (np.abs(change_point) > 0)
    out["directional_hit_rate"] = (
        float(np.mean(np.sign(change_point[called]) == np.sign(change_actual[called])))
        if called.any()
        else float("nan")
    )
    out["directional_coverage"] = float(np.mean(called))
    nonzero = np.abs(actual) > 0
    out["sign_hit_rate"] = (
        float(np.mean(np.sign(point[nonzero]) == np.sign(actual[nonzero])))
        if nonzero.any()
        else float("nan")
    )
    return out


def _subperiod_mask(
    quarters: pd.Series, window: tuple[str, str] | None
) -> np.ndarray:
    """Boolean mask selecting target quarters inside ``window``."""
    if window is None:
        return np.ones(len(quarters), dtype=bool)
    start, end = pd.Period(window[0], freq="Q"), pd.Period(window[1], freq="Q")
    values = pd.PeriodIndex(quarters, freq="Q")
    return np.asarray((values >= start) & (values <= end), dtype=bool)


# ──────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationResult:
    """Everything a rolling-origin evaluation produced.

    Attributes
    ----------
    rows
        One row per ``(model, quarter, horizon, as_of)``: the nowcast, its
        interval, the realisation, the last published value, and the error.
        This is the primitive everything else is derived from.
    metrics
        Per ``(subperiod, model, horizon)`` accuracy and calibration, computed on
        the ``primary_offset_days`` rows only — see the module docstring on why
        the multi-date grid is not an observation multiplier.
    dm_tests
        Diebold-Mariano tests against the benchmark, same grouping.
    diagnostics
        Evaluation dates attempted and used, skip reasons, nested-selection
        verification records, and the configuration.
    """

    rows: pd.DataFrame
    metrics: pd.DataFrame
    dm_tests: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        """Model names present in the results."""
        return sorted(self.rows["model"].unique()) if not self.rows.empty else []

    def full_sample(self) -> pd.DataFrame:
        """The full-sample metrics block."""
        if self.metrics.empty:
            return self.metrics
        return self.metrics.loc[self.metrics["subperiod"] == "full"]

    def evolution(self, quarter: pd.Period | str) -> pd.DataFrame:
        """How each model's nowcast of ``quarter`` moved as data arrived.

        Parameters
        ----------
        quarter
            Target quarter.

        Returns
        -------
        pd.DataFrame
            Rows for that quarter across every ``as_of`` in the grid, sorted by
            date — the data behind
            :func:`plot_nowcast_evolution` and the reference paper's Figures 8-9.
        """
        q = quarter if isinstance(quarter, pd.Period) else pd.Period(quarter, freq="Q")
        block = self.rows.loc[self.rows["quarter"] == q]
        return block.sort_values(["model", "as_of"]).reset_index(drop=True)

    def summary(self, *, subperiod: str = "full") -> str:
        """A plain-text report, including whether the challenger actually won.

        Parameters
        ----------
        subperiod
            Which subperiod to summarise.

        Returns
        -------
        str
        """
        lines: list[str] = []
        diag = self.diagnostics
        lines.append("Nowcast rolling-origin evaluation")
        lines.append("=" * 64)
        lines.append(
            f"data mode            : {diag.get('data_mode')}  "
            f"(evaluation_target={diag.get('evaluation_target')!r})"
        )
        lines.append(
            f"evaluation dates used: {diag.get('n_as_of_used')} of "
            f"{diag.get('n_as_of_attempted')}"
        )
        lines.append(
            f"primary offset       : {diag.get('primary_offset_days')} days "
            f"after quarter end (grid: {diag.get('as_of_offsets_days')})"
        )
        lines.append(f"benchmark            : {diag.get('benchmark')}")
        lines.append("")

        block = self.metrics.loc[self.metrics["subperiod"] == subperiod]
        if block.empty:
            lines.append(f"No evaluation rows in subperiod {subperiod!r}.")
            return "\n".join(lines)

        lines.append(f"Metrics — subperiod {subperiod!r}")
        lines.append("-" * 64)
        columns = [
            "model",
            "horizon",
            "n",
            "rmse",
            "mae",
            "mean_error",
            "directional_hit_rate",
            "coverage_80",
            "coverage_95",
        ]
        lines.append(
            block[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )
        lines.append("")

        dm = self.dm_tests.loc[self.dm_tests["subperiod"] == subperiod]
        if not dm.empty:
            lines.append(f"Diebold-Mariano vs {diag.get('benchmark')}")
            lines.append("-" * 64)
            lines.append(
                dm[
                    [
                        "model",
                        "horizon",
                        "n",
                        "rmse_ratio",
                        "statistic",
                        "p_value",
                        "lags",
                        "verdict",
                    ]
                ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
            )
            lines.append("")
            lines.append(self._verdict_paragraph(dm, subperiod))
        return "\n".join(lines)

    def _verdict_paragraph(self, dm: pd.DataFrame, subperiod: str) -> str:
        """State plainly whether anything beat the benchmark."""
        benchmark = self.diagnostics.get("benchmark")
        wins = dm.loc[(dm["rmse_ratio"] < 1.0) & (dm["p_value"] < 0.05)]
        if wins.empty:
            losers = dm.loc[dm["rmse_ratio"] >= 1.0, "model"].unique()
            text = (
                f"No model beats {benchmark} at the 5 % level in subperiod "
                f"{subperiod!r}."
            )
            if len(losers):
                text += (
                    f" {', '.join(sorted(losers))} did not improve on it at all. "
                )
            return (
                text
                + "The smoothing regression is the reduced form of the library's "
                "own AR(1)/Rudin models, so this is a substantive result rather "
                "than a failed run: at this sample size the extra features do "
                "not pay for themselves."
            )
        rows = ", ".join(
            f"{r.model} (h={r.horizon}, RMSE ratio {r.rmse_ratio:.3f}, "
            f"p={r.p_value:.3f})"
            for r in wins.itertuples()
        )
        return (
            f"Beats {benchmark} at the 5 % level in subperiod {subperiod!r}: "
            f"{rows}. Note the Diebold-Mariano p-values are not corrected for "
            f"testing {dm['model'].nunique()} model(s) across "
            f"{dm['horizon'].nunique()} horizon(s)."
        )


# ──────────────────────────────────────────────────────────────────
# The harness
# ──────────────────────────────────────────────────────────────────


def _as_model(spec: Any) -> Any:
    """Instantiate a class or factory; return an already-built instance unchanged.

    The class case must be tested first and by ``isinstance(spec, type)``, not by
    looking for a ``fit`` attribute: a nowcaster *class* has one (the unbound
    method), so an attribute check misreads ``NaiveCarryForward`` as an instance
    and calls ``NaiveCarryForward.fit(info)`` with ``info`` bound to ``self``.
    """
    if isinstance(spec, type):
        return spec()
    if callable(spec) and not hasattr(spec, "fit"):
        return spec()
    return spec


def rolling_origin_evaluation(
    models: Mapping[str, Any],
    data: EvaluationData,
    start: str | pd.Period,
    end: str | pd.Period,
    horizons: Sequence[int] = (1, 2),
    inner_validation_quarters: int = 12,
    *,
    as_of_offsets_days: Sequence[int] = (0, 30),
    primary_offset_days: int | None = None,
    min_train_quarters: int = DEFAULT_MIN_TRAIN_QUARTERS,
    n_draws: int = 1000,
    benchmark: str | None = _BENCHMARK,
    subperiods: Mapping[str, tuple[str, str]] | None = None,
    dm_loss: str = "squared",
    dm_lags: int | None = None,
    rng: np.random.Generator | int | None = 0,
    fallback_policy: FallbackPolicy | str = FallbackPolicy.WARN,
    verify_nested_selection: bool = True,
    progress: Callable[[str], None] | None = None,
) -> EvaluationResult:
    r"""Replay each model quarter by quarter and score it against the realisation.

    Parameters
    ----------
    models
        ``{name: nowcaster}``. Values may be instances (refitted in place at
        every evaluation date — ``fit`` clears all prior state) or zero-argument
        factories, which are called once per evaluation date.
    data
        The full dataset. Models never receive it; they receive an
        :class:`InformationSet` built from it.
    start, end
        First and last **target** quarter to score.
    horizons
        Horizons to score. Must be contiguous from 1.
    inner_validation_quarters
        Passed to any model exposing it, so the inner split length is set in one
        place. Models without the attribute are left alone.
    as_of_offsets_days
        Calendar days after a quarter's end at which to build an information set.
        ``(0, 30)`` by default: offset ``0`` is the instant a quarter closes,
        which with a ~100-day publication lag is the only point at which *two*
        quarters are simultaneously closed-and-unpublished — so it is the offset
        that exercises ``h=2``. Offset ``30`` is the spec's nominal nowcast date.

        **A third and fourth offset add nothing, and it is worth knowing why.**
        Once a quarter has closed, no further public factor data about *that
        quarter* can arrive, so its nowcast cannot move with the calendar. What
        does move it is a **publication event**: when the previous quarter's
        reported value is released, the target drops from ``h=2`` to ``h=1`` and
        conditions on a published value instead of a nowcast of one. So a target
        quarter has as many distinct nowcasts as there are horizons it passes
        through — two, with the default horizons — and offsets 30, 60 and 90 all
        return the same number. :func:`plot_nowcast_evolution` shows that
        progression; it is genuine evolution, just event-driven rather than
        continuous. Nowcasting a quarter *while it is still open* would give
        continuous evolution, and is not supported: a partial-quarter factor
        aggregate is not comparable to the full-quarter aggregates the models
        train on.
    primary_offset_days
        Which offset the headline metrics use. ``None`` picks the smallest
        offset, since it is the one that covers every horizon. Rows from the
        other offsets are kept in ``rows`` but excluded from ``metrics`` and the
        Diebold-Mariano test — several nowcasts of the same quarter are not
        several observations.
    min_train_quarters
        Skip evaluation dates with fewer published quarters. ``40`` per the spec.
    n_draws
        Predictive draws per nowcast, for interval coverage.
    benchmark
        Model name the Diebold-Mariano tests compare against. ``None`` skips
        them.
    subperiods
        ``{name: (start_quarter, end_quarter)}``. Defaults to
        :data:`DEFAULT_SUBPERIODS`; ``'full'`` is always added.
    dm_loss, dm_lags
        Passed to :func:`diebold_mariano`.
    rng
        Seed or generator for the predictive simulations. Derived per evaluation
        date so results do not depend on iteration order.
    fallback_policy
        ``'warn'`` (default), ``'strict'`` or ``'auto'``. Under ``'warn'`` a
        model that fails at one evaluation date is recorded and skipped; under
        ``'strict'`` the failure propagates.
    verify_nested_selection
        Run :func:`_assert_nested_selection` at every evaluation date. On by
        default — it is cheap and it is the property the whole harness exists to
        protect.
    progress
        Optional callback receiving a short status string per evaluation date.

    Returns
    -------
    EvaluationResult

    Raises
    ------
    ValueError
        If ``horizons`` is not contiguous from 1, if ``start > end``, or if no
        evaluation date yields a usable information set.
    NestedSelectionError
        If a model's hyperparameter selection is found to have seen an
        evaluation quarter.

    Notes
    -----
    Everything is re-estimated at every evaluation date: coefficients, scalers,
    PCA rotations, hyperparameters and residual pools. That is deliberately
    wasteful — reusing a global fit is the leak this function exists to avoid
    (``DESIGN_NOTES.md``, Risk 1).

    Examples
    --------
    >>> report = rolling_origin_evaluation(            # doctest: +SKIP
    ...     models={
    ...         "naive": NaiveCarryForward,
    ...         "smoothing_regression": SmoothingRegressionNowcaster,
    ...         "signature": SignatureNowcaster,
    ...     },
    ...     data=EvaluationData(reported=pe, public_factors=factors,
    ...                         vintages=vintages),
    ...     start="2010Q1", end="2026Q1",
    ... )
    >>> print(report.summary())                        # doctest: +SKIP
    """
    policy = (
        fallback_policy
        if isinstance(fallback_policy, FallbackPolicy)
        else FallbackPolicy(fallback_policy)
    )
    hs = sorted(set(int(h) for h in horizons))
    if hs != list(range(1, len(hs) + 1)):
        raise ValueError(
            f"horizons must be contiguous from 1 (recursion needs h-1); got {hs}."
        )
    if not models:
        raise ValueError("`models` is empty.")
    offsets = sorted(set(int(o) for o in as_of_offsets_days))
    if not offsets:
        raise ValueError("`as_of_offsets_days` is empty.")
    if min(offsets) < 0:
        raise ValueError(f"as_of_offsets_days must be >= 0, got {offsets!r}.")
    primary = int(primary_offset_days) if primary_offset_days is not None else offsets[0]
    if primary not in offsets:
        raise ValueError(
            f"primary_offset_days={primary} is not in as_of_offsets_days={offsets!r}."
        )

    start_q = start if isinstance(start, pd.Period) else pd.Period(start, freq="Q")
    end_q = end if isinstance(end, pd.Period) else pd.Period(end, freq="Q")
    if start_q > end_q:
        raise ValueError(f"start ({start_q}) must not exceed end ({end_q}).")

    realised = data.realisations()
    base_rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    seed_sequence = base_rng.integers(0, 2**32 - 1, size=1)[0]

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    data_modes: set[str] = set()
    n_attempted = 0

    # The as_of grid: for every quarter in range, one date per offset. A quarter
    # appears as a *target* at whichever offsets leave it closed-and-unpublished.
    grid: list[tuple[pd.Timestamp, int]] = []
    for quarter in pd.period_range(start_q - max(hs), end_q, freq="Q"):
        for offset in offsets:
            grid.append((quarter.end_time + pd.Timedelta(days=offset), offset))
    grid = sorted(set(grid))

    for as_of, offset in grid:
        n_attempted += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = data.information_set(as_of, max_horizon=max(hs))
        except (ValueError, KeyError) as exc:
            skipped.append(
                {"as_of": as_of, "offset_days": offset, "reason": str(exc)}
            )
            continue
        data_modes.add(info.data_mode)

        if info.n_published < min_train_quarters:
            skipped.append(
                {
                    "as_of": as_of,
                    "offset_days": offset,
                    "reason": (
                        f"only {info.n_published} published quarters "
                        f"(min_train_quarters={min_train_quarters})"
                    ),
                }
            )
            continue

        targets = [
            q
            for q in info.unpublished_quarters()
            if start_q <= q <= end_q and info.horizon_of(q) in hs
        ]
        if not targets:
            skipped.append(
                {
                    "as_of": as_of,
                    "offset_days": offset,
                    "reason": "no in-range unpublished quarter at this date",
                }
            )
            continue
        if progress is not None:
            progress(
                f"as_of={as_of.date()} (+{offset}d) targets="
                f"{[str(q) for q in targets]}"
            )

        for name, spec in models.items():
            model = _as_model(spec)
            if hasattr(model, "inner_validation_quarters"):
                model.inner_validation_quarters = inner_validation_quarters
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(info)
                    if verify_nested_selection:
                        verifications.append(
                            {
                                "as_of": as_of,
                                "model": name,
                                **_assert_nested_selection(
                                    model, info, targets, model_name=name
                                ),
                            }
                        )
                    draw_rng = np.random.default_rng(
                        [int(seed_sequence), int(as_of.value), abs(hash(name)) % 2**32]
                    )
                    results = model.predict(
                        info, targets, n_draws=n_draws, rng=draw_rng
                    )
            except NestedSelectionError:
                raise
            except Exception as exc:  # noqa: BLE001 - one date must not kill the run
                if policy is FallbackPolicy.STRICT:
                    raise
                skipped.append(
                    {
                        "as_of": as_of,
                        "offset_days": offset,
                        "model": name,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            s_last = float(info.reported.iloc[-1])
            for result in results:
                if result.quarter not in realised.index:
                    continue
                actual = float(realised[result.quarter])
                records.append(
                    {
                        "model": name,
                        "quarter": result.quarter,
                        "horizon": result.horizon,
                        "as_of": as_of,
                        "offset_days": offset,
                        "point": result.point,
                        "actual": actual,
                        "error": actual - result.point,
                        "s_last": s_last,
                        "lower_80": result.interval_80[0],
                        "upper_80": result.interval_80[1],
                        "lower_95": result.interval_95[0],
                        "upper_95": result.interval_95[1],
                        "covered_80": result.covers(actual, 0.80),
                        "covered_95": result.covers(actual, 0.95),
                        "interval_method": result.interval_method,
                        "n_train": result.diagnostics.get("n_train"),
                        "data_mode": info.data_mode,
                    }
                )

    rows = pd.DataFrame.from_records(records)
    if rows.empty:
        raise ValueError(
            f"no evaluation row was produced between {start_q} and {end_q}. "
            f"Attempted {n_attempted} date(s); first few skip reasons: "
            f"{[s['reason'] for s in skipped[:3]]}"
        )

    rows_by_model = rows["model"].value_counts().to_dict()
    silent = [name for name in models if rows_by_model.get(name, 0) == 0]
    if silent:
        reasons = [
            s["reason"]
            for s in skipped
            if s.get("model") in silent
        ]
        message = (
            f"model(s) {silent} produced no evaluation row at any of "
            f"{n_attempted} date(s) and are absent from the report entirely. "
            "That is a failure, not a result — a model that never fits cannot be "
            "compared to one that does. First reasons: "
            f"{reasons[:3] if reasons else 'no per-model failure recorded'}"
        )
        if policy is FallbackPolicy.STRICT:
            raise RuntimeError(message)
        warnings.warn(message, EvaluationWarning, stacklevel=2)

    windows: dict[str, tuple[str, str] | None] = {"full": None}
    windows.update(dict(subperiods if subperiods is not None else DEFAULT_SUBPERIODS))

    primary_rows = rows.loc[rows["offset_days"] == primary]
    metrics = _build_metrics(primary_rows, windows)
    dm_tests = (
        _build_dm_tests(
            primary_rows, windows, benchmark=benchmark, loss=dm_loss, lags=dm_lags
        )
        if benchmark is not None
        else pd.DataFrame()
    )

    diagnostics = {
        "n_as_of_attempted": n_attempted,
        "n_as_of_used": int(rows["as_of"].nunique()),
        "n_rows": int(len(rows)),
        "n_rows_by_model": rows_by_model,
        "models_with_no_rows": silent,
        "n_primary_rows": int(len(primary_rows)),
        "skipped": skipped,
        "n_skipped": len(skipped),
        "nested_selection_checks": len(verifications),
        "nested_selection_verified": bool(verify_nested_selection),
        "nested_selection_records": verifications[:50],
        "data_mode": "/".join(sorted(data_modes)) if data_modes else None,
        "evaluation_target": data.evaluation_target
        or ("first_print" if data.vintages is not None else "latest"),
        "as_of_offsets_days": tuple(offsets),
        "primary_offset_days": primary,
        "horizons": tuple(hs),
        "min_train_quarters": int(min_train_quarters),
        "inner_validation_quarters": int(inner_validation_quarters),
        "benchmark": benchmark,
        "subperiods": {k: v for k, v in windows.items() if v is not None},
        "n_draws": int(n_draws),
        "models": list(models),
        "fallback_policy": policy.value,
        "start": str(start_q),
        "end": str(end_q),
    }
    return EvaluationResult(
        rows=rows, metrics=metrics, dm_tests=dm_tests, diagnostics=diagnostics
    )


def _build_metrics(
    rows: pd.DataFrame, windows: Mapping[str, tuple[str, str] | None]
) -> pd.DataFrame:
    """Metrics per ``(subperiod, model, horizon)``."""
    records: list[dict[str, Any]] = []
    for subperiod, window in windows.items():
        mask = _subperiod_mask(rows["quarter"], window)
        block = rows.loc[mask]
        for (model, horizon), group in block.groupby(["model", "horizon"]):
            metrics = _metrics_for(group)
            if metrics.get("n", 0) == 0:
                continue
            records.append(
                {"subperiod": subperiod, "model": model, "horizon": int(horizon), **metrics}
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.sort_values(["subperiod", "horizon", "rmse"]).reset_index(drop=True)


def _build_dm_tests(
    rows: pd.DataFrame,
    windows: Mapping[str, tuple[str, str] | None],
    *,
    benchmark: str,
    loss: str,
    lags: int | None,
) -> pd.DataFrame:
    """Diebold-Mariano tests against ``benchmark``, paired on the same points."""
    records: list[dict[str, Any]] = []
    for subperiod, window in windows.items():
        mask = _subperiod_mask(rows["quarter"], window)
        block = rows.loc[mask]
        if benchmark not in set(block["model"]):
            continue
        for horizon in sorted(block["horizon"].unique()):
            at_h = block.loc[block["horizon"] == horizon]
            base = at_h.loc[at_h["model"] == benchmark].set_index("quarter")["error"]
            for model in sorted(at_h["model"].unique()):
                if model == benchmark:
                    continue
                other = at_h.loc[at_h["model"] == model].set_index("quarter")["error"]
                shared = base.index.intersection(other.index)
                if shared.size < 3:
                    continue
                paired_a = other.loc[shared].to_numpy()
                paired_b = base.loc[shared].to_numpy()
                try:
                    test = diebold_mariano(
                        paired_a,
                        paired_b,
                        horizon=int(horizon),
                        loss=loss,
                        lags=lags,
                    )
                except ValueError:
                    continue
                rmse_a = float(np.sqrt(np.mean(paired_a**2)))
                rmse_b = float(np.sqrt(np.mean(paired_b**2)))
                ratio = rmse_a / rmse_b if rmse_b > 0 else float("nan")
                if not np.isfinite(test["statistic"]):
                    verdict = "undetermined"
                elif test["p_value"] >= 0.05:
                    verdict = "no significant difference"
                elif ratio < 1.0:
                    verdict = f"beats {benchmark}"
                else:
                    verdict = f"worse than {benchmark}"
                records.append(
                    {
                        "subperiod": subperiod,
                        "model": model,
                        "benchmark": benchmark,
                        "horizon": int(horizon),
                        "rmse": rmse_a,
                        "rmse_benchmark": rmse_b,
                        "rmse_ratio": ratio,
                        "verdict": verdict,
                        **{
                            k: v
                            for k, v in test.items()
                            if k not in ("horizon", "loss")
                        },
                        "loss": loss,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.sort_values(["subperiod", "horizon", "rmse_ratio"]).reset_index(
        drop=True
    )


# ──────────────────────────────────────────────────────────────────
# Plots (matplotlib optional)
# ──────────────────────────────────────────────────────────────────


def _pyplot() -> Any:
    """Import ``matplotlib.pyplot``, with an actionable error if it is absent.

    Typed ``Any`` rather than a module type: matplotlib is an optional extra, so
    annotating it concretely would make this module unimportable for anyone who
    has not installed it.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib. Install it with "
            "`pip install private_assets_frequency[viz]`."
        ) from exc
    return plt


def plot_nowcast_vs_realised(
    result: EvaluationResult,
    *,
    model: str | None = None,
    horizon: int = 1,
    offset_days: int | None = None,
    ax: Any = None,
) -> Any:
    """Nowcast against realisation over time, with 80 % bands.

    Parameters
    ----------
    result
        The evaluation result.
    model
        Model to plot. ``None`` uses the benchmark if present, else the first.
    horizon
        Horizon to plot.
    offset_days
        Which ``as_of`` offset to use. ``None`` uses the primary one.
    ax
        Existing axes to draw on; created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _pyplot()
    offset = (
        offset_days
        if offset_days is not None
        else result.diagnostics.get("primary_offset_days", 0)
    )
    name = model or result.diagnostics.get("benchmark") or result.models[0]
    block = result.rows.loc[
        (result.rows["model"] == name)
        & (result.rows["horizon"] == horizon)
        & (result.rows["offset_days"] == offset)
    ].sort_values("quarter")
    if block.empty:
        raise ValueError(
            f"no rows for model={name!r}, horizon={horizon}, offset={offset}."
        )

    figure = None
    if ax is None:
        figure, ax = plt.subplots(figsize=(11, 4.5))
    else:
        figure = ax.get_figure()
    x = pd.PeriodIndex(block["quarter"], freq="Q").to_timestamp(how="end")
    ax.fill_between(
        x,
        block["lower_80"],
        block["upper_80"],
        alpha=0.25,
        label="80% predictive interval",
    )
    ax.plot(x, block["point"], marker="o", ms=3, lw=1.2, label=f"{name} nowcast")
    ax.plot(x, block["actual"], marker="s", ms=3, lw=1.2, label="realised")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_title(
        f"Nowcast vs realised — {name}, h={horizon} "
        f"(as_of = quarter end + {offset}d, "
        f"target={result.diagnostics.get('evaluation_target')})"
    )
    ax.set_ylabel("quarterly return")
    ax.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def plot_error_by_horizon(
    result: EvaluationResult, *, subperiod: str = "full", ax: Any = None
) -> Any:
    """RMSE by model and horizon, as grouped bars.

    Parameters
    ----------
    result
        The evaluation result.
    subperiod
        Subperiod to plot.
    ax
        Existing axes; created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _pyplot()
    block = result.metrics.loc[result.metrics["subperiod"] == subperiod]
    if block.empty:
        raise ValueError(f"no metrics for subperiod {subperiod!r}.")

    figure = None
    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 4.0))
    else:
        figure = ax.get_figure()
    horizons = sorted(block["horizon"].unique())
    models = sorted(block["model"].unique())
    width = 0.8 / max(len(models), 1)
    for i, name in enumerate(models):
        values = [
            float(
                block.loc[
                    (block["model"] == name) & (block["horizon"] == h), "rmse"
                ].iloc[0]
            )
            if not block.loc[
                (block["model"] == name) & (block["horizon"] == h)
            ].empty
            else np.nan
            for h in horizons
        ]
        positions = np.arange(len(horizons)) + i * width - 0.4 + width / 2
        ax.bar(positions, values, width=width, label=name)
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_ylabel("RMSE")
    ax.set_title(f"Nowcast RMSE by horizon — subperiod {subperiod!r}")
    ax.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def plot_nowcast_evolution(
    result: EvaluationResult,
    quarters: Sequence[pd.Period | str] | None = None,
    *,
    max_panels: int = 6,
) -> Any:
    """How each nowcast of a quarter moved as public data arrived.

    In the style of the reference paper's Figures 8-9: one panel per target
    quarter, nowcast on the vertical axis, ``as_of`` date on the horizontal, with
    the realisation as a horizontal line.

    Parameters
    ----------
    result
        An evaluation run with more than one entry in ``as_of_offsets_days`` —
        otherwise there is nothing to show.
    quarters
        Quarters to panel. ``None`` takes the most recent ones with more than one
        ``as_of``.
    max_panels
        Cap on the number of panels.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the run used a single ``as_of`` offset, or no quarter has more than
        one evaluation date.
    """
    plt = _pyplot()
    offsets = result.diagnostics.get("as_of_offsets_days", ())
    if len(offsets) < 2:
        raise ValueError(
            "nowcast evolution needs more than one as_of offset; this run used "
            f"{offsets}. Re-run with e.g. as_of_offsets_days=(0, 30, 60)."
        )
    counts = result.rows.groupby("quarter")["as_of"].nunique()
    candidates = list(counts.loc[counts > 1].index)
    if not candidates:
        raise ValueError("no target quarter has more than one evaluation date.")
    if quarters is None:
        chosen = sorted(candidates)[-max_panels:]
    else:
        chosen = [
            q if isinstance(q, pd.Period) else pd.Period(q, freq="Q")
            for q in quarters
        ][:max_panels]

    n = len(chosen)
    cols = min(3, n)
    rows_n = int(np.ceil(n / cols))
    figure, axes = plt.subplots(
        rows_n, cols, figsize=(4.2 * cols, 3.2 * rows_n), squeeze=False
    )
    for idx, quarter in enumerate(chosen):
        ax = axes[idx // cols][idx % cols]
        block = result.evolution(quarter)
        for name, group in block.groupby("model"):
            group = group.sort_values("as_of")
            ax.plot(
                group["as_of"], group["point"], marker="o", ms=4, lw=1.2, label=name
            )
            ax.fill_between(
                group["as_of"], group["lower_80"], group["upper_80"], alpha=0.15
            )
        if not block.empty:
            ax.axhline(
                float(block["actual"].iloc[0]),
                color="k",
                ls="--",
                lw=1.0,
                label="realised",
            )
        ax.set_title(f"{quarter}", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        if idx == 0:
            ax.legend(loc="best", fontsize=7)
    for idx in range(n, rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")
    figure.suptitle("Nowcast evolution as public data arrives", y=1.0)
    figure.tight_layout()
    return figure
