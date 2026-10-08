r"""
Handing nowcasts to the rest of the library, and taking them back when reality arrives.

Three functions and one guarantee.

:func:`extend_with_nowcasts`
    Appends provisional quarters after the last published one, each carrying the
    model that produced it, the ``as_of`` it was made at, and its 80 % interval.
    The result is a flat frame, so a provisional number can never be mistaken for
    a published one by reading the value alone.

:func:`to_pipeline_returns`
    Assembles the wide frame
    :class:`~private_assets_frequency.pipeline.runner.FrequencyPipeline` expects
    and attaches the provisional mask it needs. This is the supported way to feed
    provisional data to the pipeline; ``allow_provisional=True`` without the mask
    raises rather than guessing.

:func:`reconcile`
    Overwrites provisional rows once the official print lands, keeping the
    nowcast, the realisation and the error. That log is the raw material for the
    empirical prediction intervals in
    :mod:`~private_assets_frequency.nowcast.uncertainty` — a nowcaster's own
    history of being wrong is the best available estimate of how wrong it will be
    next time.

The guarantee
-------------
**Provisional quarters never enter an estimate.** With
``FrequencyPipeline(allow_provisional=True)``, Stage 1 (desmoothing) and Stage 2
(the factor model) are fitted on published rows only; the desmoothed series is
then extended across the provisional tail by applying the *fitted* filter, and
Stages 3-4 upsample the extended series. The estimated parameters come out
bit-for-bit identical with and without provisional rows, which is asserted in
``test_integration.py`` rather than asserted in prose.

The reason this matters is circularity, not tidiness. A nowcast is produced *by* a
factor model; feeding it back in as data would shrink the estimated λ and inflate
R² for no informational reason whatsoever, and the resulting numbers would look
better precisely to the extent that the nowcaster was confident.

What provisional rows *do* reach
--------------------------------
Stage 3's Chow-Lin nuisance parameters (``rho``, the indicator betas) are
estimated on the series handed to it, which includes the provisional tail. That
is deliberate and unavoidable: the temporal-aggregation constraint must hold on
every low-frequency row including the provisional ones, so Stage 3 cannot be
fitted on a shorter series and then extrapolated. The affected quantities are
disaggregation nuisances rather than the desmoothing or factor parameters the
spec names, and the span is recorded in
``disaggregation.diagnostics['estimation_window']`` so the exposure is visible.

Examples
--------
>>> extended = extend_with_nowcasts(pe_index, nowcasts)        # doctest: +SKIP
>>> returns = to_pipeline_returns({"us_buyout": extended})     # doctest: +SKIP
>>> result = FrequencyPipeline(                                # doctest: +SKIP
...     returns=returns,
...     factor_returns_monthly=monthly_factors,
...     configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
...     allow_provisional=True,
... ).run()
>>> # a quarter later, the official print lands
>>> extended = reconcile(extended, newly_published)            # doctest: +SKIP
>>> reconciliation_log(extended)[["quarter", "nowcast", "actual", "error"]]
...                                                            # doctest: +SKIP
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .base import NowcastResult

__all__ = [
    "EXTENDED_COLUMNS",
    "ProvisionalDataWarning",
    "extend_with_nowcasts",
    "provisional_markers",
    "provisional_quarters",
    "reconcile",
    "reconciliation_log",
    "to_pipeline_returns",
]


EXTENDED_COLUMNS: tuple[str, ...] = (
    "value",
    "is_provisional",
    "nowcast_model",
    "as_of",
    "lower_80",
    "upper_80",
)
"""Column contract of :func:`extend_with_nowcasts`, in order."""

_LOG_ATTR = "reconciliation_log"
_HORIZON_ATTR = "nowcast_horizons"
_MASK_ATTR = "provisional_mask"


class ProvisionalDataWarning(UserWarning):
    """Emitted when provisional data is ambiguous, duplicated, or discarded."""


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _as_quarters(index: pd.Index, *, what: str) -> pd.PeriodIndex:
    """Coerce an index of quarter labels to a quarterly :class:`pd.PeriodIndex`."""
    if isinstance(index, pd.PeriodIndex):
        if index.freqstr.startswith("Q"):
            return index
        raise TypeError(
            f"{what}: PeriodIndex must be quarterly, got freq={index.freqstr!r}."
        )
    if isinstance(index, pd.DatetimeIndex):
        return pd.PeriodIndex(index, freq="Q")
    try:
        return pd.PeriodIndex([pd.Period(v, freq="Q") for v in index], freq="Q")
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"{what}: cannot interpret index of dtype {index.dtype!r} as quarters."
        ) from exc


def _empty_extended() -> pd.DataFrame:
    """An extended frame with the right columns and dtypes but no rows."""
    return pd.DataFrame(
        {
            "value": pd.Series(dtype=float),
            "is_provisional": pd.Series(dtype=bool),
            "nowcast_model": pd.Series(dtype=object),
            "as_of": pd.Series(dtype="datetime64[ns]"),
            "lower_80": pd.Series(dtype=float),
            "upper_80": pd.Series(dtype=float),
        },
        index=pd.PeriodIndex([], freq="Q"),
    )


def provisional_quarters(extended: pd.DataFrame) -> pd.PeriodIndex:
    """Quarters currently marked provisional in an extended frame.

    Parameters
    ----------
    extended
        Output of :func:`extend_with_nowcasts` or :func:`reconcile`.

    Returns
    -------
    pd.PeriodIndex
    """
    _validate_extended(extended)
    mask = extended["is_provisional"].to_numpy(dtype=bool)
    return pd.PeriodIndex(extended.index[mask], freq="Q")


def _validate_extended(extended: pd.DataFrame) -> None:
    """Check an extended frame honours the column contract."""
    if not isinstance(extended, pd.DataFrame):
        raise TypeError(
            f"extended must be a DataFrame, got {type(extended).__name__}."
        )
    missing = [c for c in EXTENDED_COLUMNS if c not in extended.columns]
    if missing:
        raise KeyError(
            f"extended frame missing columns {missing!r}; expected "
            f"{list(EXTENDED_COLUMNS)}. Build it with extend_with_nowcasts()."
        )


# ──────────────────────────────────────────────────────────────────
# extend_with_nowcasts
# ──────────────────────────────────────────────────────────────────


def extend_with_nowcasts(
    reported: pd.Series,
    nowcasts: Sequence[NowcastResult],
    *,
    on_duplicate: str = "latest",
    require_contiguous: bool = True,
) -> pd.DataFrame:
    r"""Append provisional quarters to a published series, flagged as provisional.

    Parameters
    ----------
    reported
        Published reported quarterly returns. Index may be a quarterly
        ``PeriodIndex``, a ``DatetimeIndex``, or string quarter labels.
    nowcasts
        :class:`~private_assets_frequency.nowcast.base.NowcastResult` objects for
        quarters after the last published one. All must have
        ``target='reported'`` — see Raises.
    on_duplicate
        What to do when several nowcasts cover the same quarter, which happens
        whenever an evaluation grid produced more than one ``as_of`` per quarter.
        ``'latest'`` (default) keeps the one made at the latest ``as_of`` and
        warns; ``'error'`` raises.
    require_contiguous
        Require the provisional quarters to form an unbroken run starting at the
        quarter after the last published one. On by default, because
        ``FrequencyPipeline`` upsamples a contiguous low-frequency series and a
        gap would silently misalign every downstream period.

    Returns
    -------
    pd.DataFrame
        Indexed by quarterly ``pd.PeriodIndex`` over published **and** provisional
        quarters, with exactly the columns in :data:`EXTENDED_COLUMNS`. Published
        rows carry ``is_provisional=False`` and ``NaN``/``NaT`` in the nowcast
        columns.

    Raises
    ------
    ValueError
        If ``reported`` is empty or has duplicate quarters; if any nowcast has
        ``target='true'`` (the unsmoothed return is a different quantity and
        appending it to a reported series would mix two scales); if a nowcast
        quarter is already published; or if ``require_contiguous`` and the
        provisional quarters have a gap.
    TypeError
        If ``nowcasts`` contains something that is not a ``NowcastResult``.

    Notes
    -----
    The nowcast columns are retained on a row even after :func:`reconcile`
    overwrites its value, which is what lets the reconciliation log be
    reconstructed from the frame alone.

    Examples
    --------
    >>> extended = extend_with_nowcasts(pe_index, nowcasts)   # doctest: +SKIP
    >>> extended.tail(3)                                      # doctest: +SKIP
               value  is_provisional        nowcast_model     as_of  lower_80  upper_80
    2026Q1  0.034657           False                  NaN       NaT       NaN       NaN
    2026Q2  0.046809            True  smoothing_regression 2026-10-07  0.036511  0.063221
    2026Q3  0.047744            True  smoothing_regression 2026-10-07  0.034800  0.064712
    """
    if not isinstance(reported, pd.Series):
        raise TypeError(
            f"reported must be a Series, got {type(reported).__name__}."
        )
    published = reported.astype(float).copy()
    published.index = _as_quarters(published.index, what="reported")
    if published.index.has_duplicates:
        dup = published.index[published.index.duplicated()].unique()
        raise ValueError(f"reported has duplicate quarters {list(dup)!r}.")
    published = published.sort_index().dropna()
    if published.empty:
        raise ValueError("reported is empty after dropping NaN.")
    last_published = published.index[-1]

    for item in nowcasts:
        if not isinstance(item, NowcastResult):
            raise TypeError(
                "nowcasts must contain NowcastResult objects, got "
                f"{type(item).__name__}."
            )
        if item.target != "reported":
            raise ValueError(
                f"nowcast for {item.quarter} has target={item.target!r}. Only "
                "target='reported' can extend a reported series: the unsmoothed "
                "return is a different quantity on a different scale, and "
                "appending it would make the series mean two things at once."
            )

    chosen: dict[pd.Period, NowcastResult] = {}
    duplicates: list[str] = []
    for item in sorted(nowcasts, key=lambda r: (r.quarter, r.as_of)):
        if item.quarter in chosen:
            duplicates.append(str(item.quarter))
            if on_duplicate == "error":
                raise ValueError(
                    f"several nowcasts cover {item.quarter} "
                    f"(as_of {chosen[item.quarter].as_of.date()} and "
                    f"{item.as_of.date()}). Pass on_duplicate='latest' to keep "
                    "the most recent."
                )
            if on_duplicate != "latest":
                raise ValueError(
                    f"on_duplicate must be 'latest' or 'error', got "
                    f"{on_duplicate!r}."
                )
        chosen[item.quarter] = item  # sorted ascending, so this keeps the latest
    if duplicates:
        warnings.warn(
            f"{len(duplicates)} quarter(s) {sorted(set(duplicates))} had more "
            "than one nowcast; kept the one made at the latest as_of. Several "
            "nowcasts of one quarter are revisions of a single estimate, not "
            "several estimates.",
            ProvisionalDataWarning,
            stacklevel=2,
        )

    already = sorted(str(q) for q in chosen if q <= last_published)
    if already:
        raise ValueError(
            f"nowcast(s) for {already} are at or before the last published "
            f"quarter {last_published}. Provisional rows extend a series; they "
            "do not overwrite it. Use reconcile() to replace a provisional row "
            "with an official print."
        )

    quarters = sorted(chosen)
    if require_contiguous and quarters:
        expected = [last_published + i for i in range(1, len(quarters) + 1)]
        if quarters != expected:
            raise ValueError(
                f"provisional quarters {[str(q) for q in quarters]} are not a "
                f"contiguous run after {last_published} (expected "
                f"{[str(q) for q in expected]}). FrequencyPipeline upsamples a "
                "contiguous low-frequency series, so a gap would misalign every "
                "downstream period. Pass require_contiguous=False to override."
            )

    frame = pd.DataFrame(
        {
            "value": published.to_numpy(dtype=float),
            "is_provisional": np.zeros(published.size, dtype=bool),
            "nowcast_model": pd.array([None] * published.size, dtype=object),
            "as_of": pd.Series([pd.NaT] * published.size, dtype="datetime64[ns]"),
            "lower_80": np.full(published.size, np.nan),
            "upper_80": np.full(published.size, np.nan),
        },
        index=published.index,
    )
    if quarters:
        tail = pd.DataFrame(
            {
                "value": [chosen[q].point for q in quarters],
                "is_provisional": [True] * len(quarters),
                "nowcast_model": [chosen[q].model for q in quarters],
                "as_of": pd.to_datetime([chosen[q].as_of for q in quarters]),
                "lower_80": [chosen[q].interval_80[0] for q in quarters],
                "upper_80": [chosen[q].interval_80[1] for q in quarters],
            },
            index=pd.PeriodIndex(quarters, freq="Q"),
        )
        frame = pd.concat([frame, tail])
    frame.index.name = None
    frame = frame.loc[:, list(EXTENDED_COLUMNS)]
    # The horizon is not part of the six-column contract and is not recoverable
    # from the frame, so it rides along in `attrs` for reconcile() to log. The
    # error pool is meaningless pooled across horizons, which is why it is kept
    # rather than dropped.
    frame.attrs[_HORIZON_ATTR] = {q: chosen[q].horizon for q in quarters}
    return frame


# ──────────────────────────────────────────────────────────────────
# reconcile
# ──────────────────────────────────────────────────────────────────


def reconcile(
    extended: pd.DataFrame, new_reported: pd.Series
) -> pd.DataFrame:
    r"""Replace provisional rows with official prints, logging the nowcast errors.

    Parameters
    ----------
    extended
        Output of :func:`extend_with_nowcasts` (or a previous :func:`reconcile`).
    new_reported
        Newly published reported values, indexed by quarter. Quarters not present
        in ``extended`` are appended as published rows; quarters already published
        in ``extended`` are checked for agreement and warned about on a mismatch,
        since that means an earlier print was revised.

    Returns
    -------
    pd.DataFrame
        Same columns as the input. Reconciled rows have ``is_provisional=False``
        and the official ``value``, while keeping ``nowcast_model``, ``as_of`` and
        the interval bounds so the error log can be rebuilt from the frame alone.
        The log is also attached as ``result.attrs['reconciliation_log']`` and is
        read back by :func:`reconciliation_log`.

    Raises
    ------
    KeyError
        If ``extended`` does not honour the :data:`EXTENDED_COLUMNS` contract.
    TypeError
        If ``new_reported`` is not a Series.

    Notes
    -----
    The log accumulates across calls, so reconciling quarter by quarter builds the
    out-of-sample error history that
    :func:`~private_assets_frequency.nowcast.uncertainty.build_residual_pool`
    wants. It is the honest version of that pool: errors from nowcasts that were
    actually made at the time, against the values actually printed.

    Examples
    --------
    >>> extended = reconcile(extended, official_prints)        # doctest: +SKIP
    >>> reconciliation_log(extended)[["nowcast", "actual", "error", "covered_80"]]
    ...                                                        # doctest: +SKIP
    """
    _validate_extended(extended)
    if not isinstance(new_reported, pd.Series):
        raise TypeError(
            f"new_reported must be a Series, got {type(new_reported).__name__}."
        )
    incoming = new_reported.astype(float).copy()
    incoming.index = _as_quarters(incoming.index, what="new_reported")
    incoming = incoming.sort_index().dropna()

    out = extended.copy()
    previous_log = extended.attrs.get(_LOG_ATTR)
    horizons = dict(extended.attrs.get(_HORIZON_ATTR) or {})
    records: list[dict[str, Any]] = []
    revised: list[str] = []

    appended: list[pd.Period] = []
    appended_values: list[float] = []

    for quarter, actual in incoming.items():
        if quarter not in out.index:
            # Collected and concatenated once below, with explicit dtypes.
            # Assigning row-by-row via `.loc[new_key] = dict` makes pandas infer
            # dtypes from an all-NA row and emits a FutureWarning about it.
            appended.append(quarter)
            appended_values.append(float(actual))
            continue
        row = out.loc[quarter]
        if bool(row["is_provisional"]):
            nowcast = float(row["value"])
            lower, upper = float(row["lower_80"]), float(row["upper_80"])
            records.append(
                {
                    "quarter": quarter,
                    "nowcast_model": row["nowcast_model"],
                    "as_of": row["as_of"],
                    "horizon": horizons.get(quarter, np.nan),
                    "nowcast": nowcast,
                    "actual": float(actual),
                    "error": float(actual) - nowcast,
                    "lower_80": lower,
                    "upper_80": upper,
                    "covered_80": bool(lower <= float(actual) <= upper),
                }
            )
            out.loc[quarter, "value"] = float(actual)
            out.loc[quarter, "is_provisional"] = False
        elif not np.isclose(float(row["value"]), float(actual), atol=0.0, rtol=1e-12):
            revised.append(str(quarter))
            out.loc[quarter, "value"] = float(actual)

    if revised:
        warnings.warn(
            f"{len(revised)} already-published quarter(s) {revised[:5]} arrived "
            "with a different value and were overwritten. That is a revision, "
            "not a reconciliation: it does not enter the nowcast error log, "
            "because the nowcast was never scored against this vintage.",
            ProvisionalDataWarning,
            stacklevel=2,
        )

    if appended:
        n = len(appended)
        new_rows = pd.DataFrame(
            {
                "value": np.asarray(appended_values, dtype=float),
                "is_provisional": np.zeros(n, dtype=bool),
                "nowcast_model": pd.array([None] * n, dtype=object),
                "as_of": pd.Series([pd.NaT] * n, dtype="datetime64[ns]"),
                "lower_80": np.full(n, np.nan),
                "upper_80": np.full(n, np.nan),
            },
            index=pd.PeriodIndex(appended, freq="Q"),
        )
        out = pd.concat([out, new_rows])

    out = out.sort_index()
    out.index.name = None
    log = pd.DataFrame.from_records(records) if records else None
    earlier = pd.DataFrame(previous_log) if previous_log is not None else None
    if earlier is not None and earlier.empty:
        earlier = None
    # Concatenating an empty frame also trips the all-NA dtype FutureWarning, so
    # the two sides are combined only when both actually have rows.
    if log is None:
        log = earlier if earlier is not None else _empty_log()
    elif earlier is not None:
        log = pd.concat([earlier, log], ignore_index=True)
    if not log.empty:
        log = log.drop_duplicates(subset=["quarter", "as_of"], keep="last")
        log = log.sort_values(["quarter", "as_of"]).reset_index(drop=True)
    out = out.loc[:, list(EXTENDED_COLUMNS)]
    out.attrs[_LOG_ATTR] = log
    out.attrs[_HORIZON_ATTR] = horizons
    return out


def _empty_log() -> pd.DataFrame:
    """An empty reconciliation log with the documented columns."""
    return pd.DataFrame(
        columns=[
            "quarter",
            "nowcast_model",
            "as_of",
            "horizon",
            "nowcast",
            "actual",
            "error",
            "lower_80",
            "upper_80",
            "covered_80",
        ]
    )


def reconciliation_log(extended: pd.DataFrame) -> pd.DataFrame:
    """The accumulated nowcast-error log attached by :func:`reconcile`.

    Parameters
    ----------
    extended
        A frame returned by :func:`reconcile`.

    Returns
    -------
    pd.DataFrame
        Columns ``[quarter, nowcast_model, as_of, horizon, nowcast, actual,
        error, lower_80, upper_80, covered_80]``, one row per reconciled quarter.
        Empty (with those columns) when nothing has been reconciled.

    Notes
    -----
    Feed ``log['error']`` to
    :func:`~private_assets_frequency.nowcast.uncertainty.build_residual_pool` to
    build prediction intervals from the model's realised history rather than from
    a backtest. Group by ``nowcast_model`` **and** ``horizon`` first: pooling
    errors across models describes none of them, and pooling across horizons
    throws away the horizon widening the intervals exist to express.
    """
    _validate_extended(extended)
    log = extended.attrs.get(_LOG_ATTR)
    if log is None:
        return _empty_log()
    return pd.DataFrame(log).copy()


# ──────────────────────────────────────────────────────────────────
# to_pipeline_returns
# ──────────────────────────────────────────────────────────────────


def to_pipeline_returns(
    extended: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    name: str | None = None,
    index: str = "quarter_end",
) -> pd.DataFrame:
    r"""Assemble the wide ``returns`` frame ``FrequencyPipeline`` expects.

    Parameters
    ----------
    extended
        One extended frame (then ``name`` is required) or a mapping
        ``{strategy: extended_frame}``.
    name
        Strategy name when a single frame is given.
    index
        ``'quarter_end'`` (default) converts the quarterly ``PeriodIndex`` to a
        quarter-end ``DatetimeIndex``, which is what the pipeline and the factor
        panels use. ``'period'`` keeps periods, for callers doing their own
        alignment.

    Returns
    -------
    pd.DataFrame
        One column per strategy, with the provisional mask attached as
        ``result.attrs['provisional_mask']`` — a boolean frame of the same shape.
        ``FrequencyPipeline(allow_provisional=True)`` reads that attribute and
        raises if it is absent, so this function is the supported way to build
        provisional inputs.

    Raises
    ------
    ValueError
        If strategies have differing quarter coverage, if ``name`` is missing for
        a single frame, or if a strategy's provisional rows are not a contiguous
        trailing block (the pipeline requires that, and failing here gives a
        better message than failing three stages later).

    Notes
    -----
    ``DataFrame.attrs`` is metadata, and pandas does not promise to propagate it
    through every operation. Pass the returned object to the pipeline directly
    rather than slicing or concatenating it first; the pipeline reads the
    attribute in ``__post_init__``, before touching the frame.

    Examples
    --------
    >>> returns = to_pipeline_returns({"us_buyout": extended})  # doctest: +SKIP
    >>> returns.attrs["provisional_mask"].tail(3)               # doctest: +SKIP
                us_buyout
    2026-03-31      False
    2026-06-30       True
    2026-09-30       True
    """
    if isinstance(extended, pd.DataFrame) and set(EXTENDED_COLUMNS) <= set(
        extended.columns
    ):
        if name is None:
            raise ValueError(
                "to_pipeline_returns needs `name` when given a single extended "
                "frame; or pass a mapping {strategy: frame}."
            )
        frames = {name: extended}
    elif isinstance(extended, Mapping):
        frames = dict(extended)
        if not frames:
            raise ValueError("to_pipeline_returns received an empty mapping.")
    else:
        raise TypeError(
            "extended must be an extended frame (see EXTENDED_COLUMNS) or a "
            f"mapping of them, got {type(extended).__name__}."
        )

    values: dict[str, pd.Series] = {}
    masks: dict[str, pd.Series] = {}
    reference: pd.PeriodIndex | None = None
    for strategy, frame in frames.items():
        _validate_extended(frame)
        quarters = _as_quarters(frame.index, what=f"extended[{strategy!r}]")
        if reference is None:
            reference = quarters
        elif not quarters.equals(reference):
            raise ValueError(
                f"strategy {strategy!r} covers {quarters[0]}..{quarters[-1]} but "
                f"another covers {reference[0]}..{reference[-1]}. The pipeline "
                "needs one shared low-frequency index; align the strategies "
                "first."
            )
        mask = frame["is_provisional"].to_numpy(dtype=bool)
        if mask.any():
            first = int(np.argmax(mask))
            if not mask[first:].all():
                raise ValueError(
                    f"strategy {strategy!r}: provisional rows must be a "
                    "contiguous trailing block, but a published row follows a "
                    "provisional one."
                )
        values[strategy] = pd.Series(
            frame["value"].to_numpy(dtype=float), index=quarters
        )
        masks[strategy] = pd.Series(mask, index=quarters)

    returns = pd.DataFrame(values)
    mask_frame = pd.DataFrame(masks)
    if index == "quarter_end":
        stamps = pd.PeriodIndex(returns.index, freq="Q").to_timestamp(
            how="end"
        ).normalize()
        returns.index = stamps
        mask_frame.index = stamps
    elif index != "period":
        raise ValueError(
            f"index must be 'quarter_end' or 'period', got {index!r}."
        )
    returns.attrs[_MASK_ATTR] = mask_frame
    return returns


# ──────────────────────────────────────────────────────────────────
# Reading provisional markers back out of a pipeline result
# ──────────────────────────────────────────────────────────────────


def provisional_markers(result: Any) -> dict[str, dict[str, pd.Series]]:
    r"""Boolean ``is_provisional`` markers for every output of a pipeline run.

    Parameters
    ----------
    result
        A :class:`~private_assets_frequency.pipeline.runner.PipelineResult` from a
        run with ``allow_provisional=True``.

    Returns
    -------
    dict
        ``{strategy: {'monthly': Series, 'daily': Series}}``, each Series indexed
        like the corresponding output and ``True`` on periods that fall inside a
        provisional quarter. ``'daily'`` is omitted when there is no daily output.

    Raises
    ------
    ValueError
        If the run did not use ``allow_provisional=True``, in which case no row is
        provisional and asking the question signals a misunderstanding.

    Notes
    -----
    The markers are derived from the quarter list the pipeline recorded, so they
    stay correct for whatever high-frequency index Stage 3 or Stage 4 produced.
    """
    diagnostics = getattr(result, "diagnostics", {}) or {}
    if not diagnostics.get("allow_provisional"):
        raise ValueError(
            "this PipelineResult was produced with allow_provisional=False, so "
            "no output row is provisional. Re-run with allow_provisional=True "
            "and a returns frame built by to_pipeline_returns()."
        )
    out: dict[str, dict[str, pd.Series]] = {}
    for strategy, strat in result.per_strategy.items():
        quarters = pd.PeriodIndex(
            strat.disaggregation.diagnostics.get("provisional_quarters", []),
            freq="Q",
        )
        markers: dict[str, pd.Series] = {}
        monthly = strat.disaggregation.high_frequency
        markers["monthly"] = pd.Series(
            pd.PeriodIndex(monthly.index, freq="Q").isin(quarters),
            index=monthly.index,
            name="is_provisional",
        )
        if strat.daily is not None:
            daily = strat.daily.daily_returns
            markers["daily"] = pd.Series(
                pd.PeriodIndex(daily.index, freq="Q").isin(quarters),
                index=daily.index,
                name="is_provisional",
            )
        out[strategy] = markers
    return out
