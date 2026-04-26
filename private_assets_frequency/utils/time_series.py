"""
Time-series utilities for ``private_assets_frequency``.

This module supplies the calendar / frequency / windowing primitives used
throughout the pipeline:

    * :func:`infer_frequency` — best-effort detection of native frequency
      from a :class:`pandas.DatetimeIndex`.
    * :func:`align_factors_to_returns` — index-align factor data to a return
      series, raising a clear error if the factor coverage is inadequate.
    * :func:`rolling_compound_window` — rolling K-period compounded return
      indexed at every observation. The K=4 case (quarterly) is the
      "rolling annual return" used by the AR(1) Bayesian desmoother to break
      Q4 mark-to-market seasonality (Stage 1 spec point 1).
    * :func:`four_offset_annual_streams` — the four *non-overlapping* annual
      return series, each starting at a different quarter offset, returned
      as a ``dict[int, Series]`` since their quarter-end stamps don't align
      into a single rectangular index.
    * :func:`compound_returns_panel` — multi-strategy version of compounding
      for use with cross-sectional fits.

Frequency aliases
-----------------
The library follows the ``Frequency`` enum in :mod:`core.protocols`:
``'quarterly' | 'monthly' | 'weekly' | 'daily'``. ``infer_frequency`` is
robust to either pandas 1.x (``'M'``, ``'Q'``) or 2.x (``'ME'``, ``'QE'``)
inferred frequency strings, plus a fallback based on median day-spacing.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..core.protocols import Frequency
from .returns import compound_returns

# ──────────────────────────────────────────────────────────────────
# Pandas-version-compatible frequency aliases
# ──────────────────────────────────────────────────────────────────
#
# Pandas 2.2 renamed the month-end / quarter-end aliases from 'M' / 'Q' to
# 'ME' / 'QE' and emits a FutureWarning when the old names are used. Older
# 2.x releases (2.0, 2.1) only recognise the short forms. We pick the right
# alias at import time so the rest of the library can use a stable name.

_PD_VERSION = tuple(int(p) for p in pd.__version__.split(".")[:2])

if _PD_VERSION >= (2, 2):
    MONTH_END_FREQ = "ME"
    QUARTER_END_FREQ = "QE"
else:
    MONTH_END_FREQ = "M"
    QUARTER_END_FREQ = "Q"


# ──────────────────────────────────────────────────────────────────
# Frequency inference
# ──────────────────────────────────────────────────────────────────


def infer_frequency(index: pd.DatetimeIndex) -> Frequency:
    """Best-effort native-frequency detection from a :class:`DatetimeIndex`.

    Tries :func:`pandas.infer_freq` first; falls back to the median day-spacing
    between adjacent timestamps. Raises if the result is ambiguous.

    Examples
    --------
    >>> idx = pd.date_range('2020-01-31', periods=12, freq='ME')
    >>> infer_frequency(idx) is Frequency.MONTHLY
    True
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"infer_frequency requires a DatetimeIndex, got {type(index).__name__}")
    if len(index) < 3:
        raise ValueError(
            f"infer_frequency requires at least 3 observations, got {len(index)}."
        )

    # Try pandas inference
    inferred = pd.infer_freq(index)
    if inferred is not None:
        prefix = inferred.split("-")[0]  # strip year-end suffix like 'QE-DEC'
        if prefix in ("Q", "QE", "BQ", "BQE", "QS", "BQS"):
            return Frequency.QUARTERLY
        if prefix in ("M", "ME", "BM", "BME", "MS", "BMS"):
            return Frequency.MONTHLY
        if prefix.startswith("W"):
            return Frequency.WEEKLY
        if prefix in ("B", "D", "C"):
            return Frequency.DAILY

    # Fallback: median day-spacing
    deltas = np.diff(index.values).astype("timedelta64[D]").astype(int)
    if len(deltas) == 0:
        raise ValueError("infer_frequency: cannot compute deltas (single observation).")
    median = float(np.median(deltas))
    if 80.0 <= median <= 100.0:
        return Frequency.QUARTERLY
    if 27.0 <= median <= 32.0:
        return Frequency.MONTHLY
    if 6.0 <= median <= 8.0:
        return Frequency.WEEKLY
    if 1.0 <= median <= 5.0:
        return Frequency.DAILY
    raise ValueError(
        f"infer_frequency: cannot infer frequency (median Δdays = {median:.1f}); "
        "supply Frequency explicitly."
    )


def is_period_end_aligned(index: pd.DatetimeIndex, frequency: Frequency) -> bool:
    """Check that every timestamp in ``index`` is a period-end of ``frequency``.

    Useful as a sanity assertion in the pipeline: quarterly factor returns
    that aren't quarter-end-aligned are a common silent source of off-by-one
    errors when joining with PE returns.
    """
    if frequency is Frequency.QUARTERLY:
        return bool((index == index + pd.offsets.QuarterEnd(0)).all())
    if frequency is Frequency.MONTHLY:
        return bool((index == index + pd.offsets.MonthEnd(0)).all())
    if frequency is Frequency.WEEKLY:
        # Week-end varies by convention; require Friday or Sunday end.
        weekdays = set(index.weekday.tolist())
        return weekdays.issubset({4, 6})
    # Daily: any business day is fine; we just check non-weekend.
    return bool((index.weekday < 7).all())


# ──────────────────────────────────────────────────────────────────
# Alignment
# ──────────────────────────────────────────────────────────────────


def align_factors_to_returns(
    factors: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    *,
    how: str = "inner",
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    """Align ``factors`` and ``returns`` on their common index.

    Parameters
    ----------
    factors
        Factor return matrix.
    returns
        Return series or panel.
    how
        Pandas join type. Default ``'inner'`` requires complete overlap on
        every date both sides have. ``'right'`` keeps every return date and
        forward-fills missing factor values (use sparingly — silent
        forward-fill can hide data-quality issues).

    Returns
    -------
    aligned_factors, aligned_returns
        Two pandas objects sharing the same DatetimeIndex.

    Raises
    ------
    ValueError
        If the resulting index is empty or shorter than 12 observations
        (below which any downstream fit will be unstable).
    """
    if not isinstance(factors, pd.DataFrame):
        raise TypeError(f"factors must be a DataFrame, got {type(factors).__name__}")
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"returns must be a Series or DataFrame, got {type(returns).__name__}"
        )
    if not isinstance(factors.index, pd.DatetimeIndex):
        raise TypeError("factors must have a DatetimeIndex.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex.")

    if how == "inner":
        common = factors.index.intersection(returns.index)
    elif how == "right":
        common = returns.index
    else:
        raise ValueError(f"how must be 'inner' or 'right', got {how!r}.")

    if len(common) == 0:
        raise ValueError(
            "align_factors_to_returns: factors and returns share no dates. "
            f"factor span: [{factors.index.min()}, {factors.index.max()}]; "
            f"return span: [{returns.index.min()}, {returns.index.max()}]."
        )
    if len(common) < 12:
        raise ValueError(
            f"align_factors_to_returns: only {len(common)} overlapping observations; "
            "need >= 12 for a stable fit."
        )

    factors_aligned = factors.reindex(common).sort_index()
    if how == "right" and factors_aligned.isna().any().any():
        factors_aligned = factors_aligned.ffill()
        if factors_aligned.isna().any().any():
            raise ValueError(
                "align_factors_to_returns(how='right'): factor coverage starts later "
                "than returns; cannot ffill leading NaNs."
            )

    returns_aligned = returns.reindex(common).sort_index()
    return factors_aligned, returns_aligned


# ──────────────────────────────────────────────────────────────────
# Rolling windows
# ──────────────────────────────────────────────────────────────────


def rolling_compound_window(
    returns: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling W-period compounded return at every observation.

    The series at index ``t`` equals ``∏_{i=t-W+1}^{t}(1 + r_i) − 1``. The
    first ``W − 1`` observations are NaN.

    Parameters
    ----------
    returns
        1-D simple-return series.
    window
        Number of periods to compound.

    Returns
    -------
    pandas.Series
        Same index as ``returns``, leading ``window - 1`` entries NaN.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a Series, got {type(returns).__name__}")
    if not (isinstance(window, (int, np.integer)) and window >= 1):
        raise ValueError(f"window must be an integer >= 1, got {window!r}")
    if (returns.dropna() <= -1.0).any():
        raise ValueError("rolling_compound_window: input contains values ≤ -1.")
    log_ret = np.log1p(returns.to_numpy(dtype=float))
    log_sum = pd.Series(log_ret, index=returns.index).rolling(window).sum()
    out = np.expm1(log_sum)
    out.name = f"rolling_{window}_compound" if returns.name is None else f"{returns.name}_r{window}"
    return out


def rolling_four_quarter_returns(quarterly_returns: pd.Series) -> pd.Series:
    """Convenience wrapper: rolling 4-quarter compounded return at every quarter.

    Equivalent to ``rolling_compound_window(quarterly_returns, 4)``. The first
    three entries are ``NaN``; from quarter index 3 onwards each value is
    the compound return of the trailing four quarters.

    This is the series the AR(1) Bayesian desmoother fits on (Stage 1 spec
    point 1), to break the Q4 mark-to-market seasonality of raw quarterly
    smoothing.
    """
    return rolling_compound_window(quarterly_returns, 4)


def four_offset_annual_streams(
    quarterly_returns: pd.Series,
) -> dict[int, pd.Series]:
    """Return four non-overlapping annual return streams from quarterly data.

    For quarterly returns of length ``T``, this returns a ``dict`` keyed by
    offset ``k ∈ {0, 1, 2, 3}``. Each value is a :class:`pandas.Series` of
    non-overlapping compounded 4-quarter returns starting at offset ``k``::

        stream_k[i] = compound(r_{4i + k}, r_{4i + k + 1},
                                r_{4i + k + 2}, r_{4i + k + 3})

    Each stream is indexed at the *last* quarter of its 4-quarter block.
    The four streams have similar but not identical lengths (differing by
    at most one) and do not share a common index — the offsets shift by one
    quarter, so their quarter-end timestamps differ. Returning a dict keeps
    each stream rectangular without forcing artificial alignment.

    Stage 1 spec point 1: collectively the four streams realise the full
    rolling 4-quarter return cover, but each stream individually has iid-
    enough properties for a separate fit when desired.

    Examples
    --------
    >>> q = pd.Series(
    ...     0.01, index=pd.date_range('1990-03-31', periods=12, freq='QE')
    ... )
    >>> streams = four_offset_annual_streams(q)
    >>> sorted(streams)
    [0, 1, 2, 3]
    >>> len(streams[0]), len(streams[3])
    (3, 2)

    Raises
    ------
    ValueError
        If ``len(quarterly_returns) < 4`` or any value is ≤ -1.
    """
    if not isinstance(quarterly_returns, pd.Series):
        raise TypeError(
            f"quarterly_returns must be a Series, got {type(quarterly_returns).__name__}"
        )
    if (quarterly_returns.dropna() <= -1.0).any():
        raise ValueError("four_offset_annual_streams: contains values ≤ -1.")

    n = len(quarterly_returns)
    if n < 4:
        raise ValueError(
            f"four_offset_annual_streams: need >= 4 quarters, got {n}."
        )

    arr = quarterly_returns.to_numpy(dtype=float)
    log_arr = np.log1p(arr)

    streams: dict[int, pd.Series] = {}
    for k in range(4):
        n_k = (n - k) // 4
        if n_k == 0:
            streams[k] = pd.Series(
                dtype=float,
                index=pd.DatetimeIndex([], name=quarterly_returns.index.name),
                name=f"offset_{k}",
            )
            continue
        sub = log_arr[k : k + 4 * n_k].reshape(n_k, 4)
        annual = np.expm1(sub.sum(axis=1))
        last_indices = quarterly_returns.index[k + 3 :: 4][:n_k]
        streams[k] = pd.Series(annual, index=last_indices, name=f"offset_{k}")
    return streams


# ──────────────────────────────────────────────────────────────────
# Panel utilities
# ──────────────────────────────────────────────────────────────────


def compound_returns_panel(panel: pd.DataFrame) -> pd.Series:
    """Compound each column of ``panel`` into a single total return per strategy.

    Returns a :class:`pandas.Series` indexed by the columns of ``panel``.
    Wraps :func:`utils.returns.compound_returns` for convenience.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"panel must be a DataFrame, got {type(panel).__name__}")
    return compound_returns(panel, axis=0)  # type: ignore[return-value]


def common_index(*objects: pd.Series | pd.DataFrame) -> pd.DatetimeIndex:
    """Return the intersection of the DatetimeIndexes of all arguments."""
    if not objects:
        raise ValueError("common_index: at least one argument required.")
    idx = objects[0].index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("common_index: all arguments must have a DatetimeIndex.")
    for obj in objects[1:]:
        if not isinstance(obj.index, pd.DatetimeIndex):
            raise TypeError("common_index: all arguments must have a DatetimeIndex.")
        idx = idx.intersection(obj.index)
    return idx.sort_values()


def trim_to_index(
    objects: Iterable[pd.Series | pd.DataFrame],
    index: pd.DatetimeIndex,
) -> list:
    """Reindex each object to ``index`` and sort, raising on missing values."""
    out = []
    for obj in objects:
        sub = obj.reindex(index).sort_index()
        if sub.isna().any().any() if isinstance(sub, pd.DataFrame) else sub.isna().any():
            raise ValueError("trim_to_index: missing values after reindex.")
        out.append(sub)
    return out


__all__ = [
    "MONTH_END_FREQ",
    "QUARTER_END_FREQ",
    "align_factors_to_returns",
    "common_index",
    "compound_returns_panel",
    "four_offset_annual_streams",
    "infer_frequency",
    "is_period_end_aligned",
    "rolling_compound_window",
    "rolling_four_quarter_returns",
    "trim_to_index",
]
