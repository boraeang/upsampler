"""
Stage 0 reporting-lag adjustment for hedge-fund index returns.

Hedge-fund index providers (HFRI, CS) publish a return for month *t* whose
underlying constituent reporting in fact reflects month *t − k* economics
for some integer ``k ∈ {1, 2}``. The Getmansky-Lo-Makarov MA(q) smoothing
process operates on top of this mechanical reporting delay, so the spec
requires us to *un-shift* the series before fitting MA(q): otherwise the
estimated θ weights absorb a chunk of plain-old reporting delay and the
smoothing-only interpretation downstream is wrong (sanity anchors on
volatility / correlation will be biased).

This module exposes one function:

    * :func:`adjust_reporting_lag` — shift a return series backward by
      ``lag_months``, cropping the resulting NaN tail.

The adjustment is purely mechanical: ``adjusted[t] = raw[t + k]``. The
function preserves the input frequency and is a no-op for ``lag_months=0``.

Notes
-----
* If the data vendor has *already* applied a lag adjustment, calling this
  function with a non-zero lag will *over-shift* the series. The pipeline
  emits a diagnostic when lag-adjusted autocorrelation is implausibly high
  or low so users can spot double-counting.
* The function works at any frequency but is named ``_months`` because that
  is how vendor lags are quoted. Pass the lag in *periods* of the input
  frequency.
"""

from __future__ import annotations

import pandas as pd


def adjust_reporting_lag(
    returns: pd.Series | pd.DataFrame,
    lag_months: int,
    *,
    drop_tail: bool = True,
) -> pd.Series | pd.DataFrame:
    r"""Shift a return series backward by ``lag_months`` to remove mechanical reporting delay.

    Parameters
    ----------
    returns
        Series or DataFrame of returns indexed by a :class:`pandas.DatetimeIndex`.
    lag_months
        Number of periods (typically months for hedge-fund data) to shift
        the series backward by. ``0`` is a no-op — the series is returned
        unchanged. Must be ``>= 0``.
    drop_tail
        If ``True`` (default), drop the trailing ``lag_months`` rows whose
        post-shift values would be NaN. If ``False``, keep the original
        index length with NaN tails.

    Returns
    -------
    Same container type as the input. The new index is the leading
    ``len(returns) - lag_months`` rows of the original index when
    ``drop_tail=True``.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range('2020-01-31', periods=4, freq='M')
    >>> s = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx, name='r')
    >>> adjust_reporting_lag(s, lag_months=1).tolist()
    [0.02, 0.03, 0.04]

    Raises
    ------
    ValueError
        If ``lag_months`` is negative or exceeds the length of ``returns``.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"returns must be a Series or DataFrame, got {type(returns).__name__}"
        )
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex.")
    if not isinstance(lag_months, int) or isinstance(lag_months, bool):
        raise TypeError(f"lag_months must be an int, got {type(lag_months).__name__}")
    if lag_months < 0:
        raise ValueError(f"lag_months must be >= 0, got {lag_months}")
    if lag_months >= len(returns):
        raise ValueError(
            f"lag_months={lag_months} >= len(returns)={len(returns)}; "
            "would discard all observations."
        )
    if lag_months == 0:
        return returns.copy()

    shifted = returns.shift(-lag_months)
    if drop_tail:
        shifted = shifted.iloc[:-lag_months]
    return shifted


__all__ = ["adjust_reporting_lag"]
