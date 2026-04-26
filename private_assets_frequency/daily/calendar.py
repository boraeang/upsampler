"""
Business-day calendar utilities for Stage 4 daily disaggregation.

The pipeline operates on monthly returns (post-Stage-3) and pushes them to a
daily frequency using a business-day calendar. The two key shapes that
Stage 4 needs from this module are:

    * A :class:`pandas.DatetimeIndex` of business days spanning a target
      window — :func:`business_day_index`.
    * The per-month count of business days inside that window —
      :func:`business_days_per_month`. The Kalman smoother and the simulator
      both operate on irregular monthly blocks (months have between ~19 and
      ~23 business days depending on holidays / month length).

Calendar selection
------------------
* ``'NYSE'`` (default) — uses :mod:`pandas_market_calendars` when installed
  (correct holiday set), otherwise falls back to :func:`pandas.bdate_range`
  which is Monday-Friday only. The fallback is acceptable for synthetic /
  testing use; production callers should install the optional
  ``pandas_market_calendars`` extra to get the proper holiday calendar.
* ``'BDAY'`` — explicit Mon–Fri calendar (no holidays). Useful in tests so
  results are reproducible across machines regardless of which optional
  packages are installed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def business_day_index(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    calendar: Literal["NYSE", "BDAY"] = "NYSE",
) -> pd.DatetimeIndex:
    """Generate a :class:`pandas.DatetimeIndex` of business days from ``start`` to ``end``.

    Parameters
    ----------
    start, end
        Inclusive bounds. Either pandas Timestamps or strings parsable by
        :func:`pandas.Timestamp`.
    calendar
        ``'NYSE'`` (default) or ``'BDAY'``. ``'NYSE'`` uses
        :mod:`pandas_market_calendars` when available; otherwise falls back
        to a Monday-Friday calendar. ``'BDAY'`` is always plain Mon-Fri.

    Returns
    -------
    pandas.DatetimeIndex
        Sorted, unique daily timestamps (normalised to midnight).
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError(f"end {end_ts.date()} < start {start_ts.date()}")

    if calendar == "NYSE":
        try:  # pragma: no cover — only exercised when the optional dep is installed
            import pandas_market_calendars as mcal

            cal = mcal.get_calendar("NYSE")
            schedule = cal.schedule(start_date=start_ts, end_date=end_ts)
            return pd.DatetimeIndex(schedule.index.normalize())
        except ImportError:
            return pd.bdate_range(start_ts, end_ts)
    if calendar == "BDAY":
        return pd.bdate_range(start_ts, end_ts)
    raise ValueError(f"unsupported calendar {calendar!r}")


def business_days_per_month(daily_index: pd.DatetimeIndex) -> pd.Series:
    """Count business days falling in each calendar month of ``daily_index``.

    Returns a :class:`pandas.Series` indexed at the *last business day* of
    each month with the count of daily timestamps as values. Months with no
    daily observations are absent.
    """
    if not isinstance(daily_index, pd.DatetimeIndex):
        raise TypeError(
            f"daily_index must be a DatetimeIndex, got {type(daily_index).__name__}"
        )
    if len(daily_index) == 0:
        raise ValueError("daily_index is empty.")
    # Group by year-month, keep the last index per group as the anchor
    df = pd.DataFrame({"day": daily_index}, index=daily_index)
    grouped = df.groupby(daily_index.to_period("M"))
    counts = grouped.size()
    last_days = grouped["day"].max()
    out = pd.Series(counts.to_numpy(), index=pd.DatetimeIndex(last_days), name="n_business_days")
    return out.sort_index()


def block_sizes_for_months(
    daily_index: pd.DatetimeIndex,
    monthly_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Return the per-month count of daily timestamps aligned to ``monthly_index``.

    Each entry in ``monthly_index`` is interpreted as a *month-end* anchor;
    the count returned at position ``k`` is the number of daily observations
    falling in the calendar month of ``monthly_index[k]``.

    Raises
    ------
    ValueError
        If ``monthly_index`` is not in ascending order, contains months
        absent from ``daily_index``, or if the implied total daily count
        does not equal ``len(daily_index)`` (i.e. there are extra daily
        observations outside the requested months).
    """
    if not isinstance(daily_index, pd.DatetimeIndex):
        raise TypeError("daily_index must be a DatetimeIndex.")
    if not isinstance(monthly_index, pd.DatetimeIndex):
        raise TypeError("monthly_index must be a DatetimeIndex.")
    if len(monthly_index) == 0:
        raise ValueError("monthly_index is empty.")
    if not monthly_index.is_monotonic_increasing:
        raise ValueError("monthly_index must be in ascending order.")

    counts_by_month = business_days_per_month(daily_index)
    requested_periods = monthly_index.to_period("M")
    available_periods = counts_by_month.index.to_period("M")
    period_to_count = dict(zip(available_periods, counts_by_month.to_numpy()))
    out = np.empty(len(monthly_index), dtype=int)
    for i, p in enumerate(requested_periods):
        if p not in period_to_count:
            raise ValueError(
                f"month {p} requested but no daily observations fall inside it."
            )
        out[i] = period_to_count[p]
    if int(out.sum()) != len(daily_index):
        # Daily observations outside the requested months — silently dropping
        # them would invite confusion, so we surface the mismatch.
        raise ValueError(
            f"daily_index has {len(daily_index)} entries but only "
            f"{int(out.sum())} fall inside the requested {len(monthly_index)} months; "
            "filter daily_index to the months you want before disaggregating."
        )
    return out


def irregular_aggregation_matrix(block_sizes: np.ndarray | list[int]) -> np.ndarray:
    """Build a block-summing aggregation matrix with variable block sizes.

    Returns a matrix ``C`` of shape ``(len(block_sizes), sum(block_sizes))``
    where row ``b`` contains ones in the columns corresponding to block ``b``
    and zeros elsewhere — i.e. ``C @ x_high`` sums each block of ``x_high``.

    Used by Stage 4 disaggregation when block sizes (business days per month)
    differ across periods.
    """
    sizes = np.asarray(block_sizes, dtype=int)
    if sizes.ndim != 1:
        raise ValueError(f"block_sizes must be 1-D, got ndim={sizes.ndim}")
    if (sizes < 1).any():
        raise ValueError("block_sizes entries must all be >= 1.")
    n_low = sizes.size
    n_high = int(sizes.sum())
    C = np.zeros((n_low, n_high))
    start = 0
    for b, sz in enumerate(sizes):
        C[b, start : start + sz] = 1.0
        start += int(sz)
    return C


__all__ = [
    "block_sizes_for_months",
    "business_day_index",
    "business_days_per_month",
    "irregular_aggregation_matrix",
]
