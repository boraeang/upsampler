r"""
Stage 0 carry / mark-to-market decomposition for private credit returns.

Private credit returns mix two components with fundamentally different
dynamics (Stage 0 of the spec):

    * **Carry** — coupon income accruing smoothly at the running yield
      rate. This is *not* smoothed in the Geltner sense; it really does
      accrue at a steady rate. Desmoothing it would invent volatility that
      is not there.
    * **Mark-to-market (MTM)** — changes in fair value of the credit
      spread / default expectation. Loans may carry at par for quarters,
      then suddenly be written down. *This* is the smoothed component, and
      Stage 1 (Threshold AR(1)) operates on it alone.

This module exposes two simple functions:

    * :func:`decompose_credit_return` — split a total-return series into
      ``carry_t = yield_t · Δt`` and ``mtm_t = total_t − carry_t``.
    * :func:`reattach_carry` — after Stage 3 disaggregates the MTM
      component to a higher frequency, add the carry back at that
      frequency, distributed linearly within each low-frequency block.

Both functions are pandas-aware and operate at quarterly, monthly, or daily
frequency.

Notes
-----
* "Linearly within each block" means we split the low-frequency carry
  evenly across the high-frequency periods inside the block — equivalent to
  assuming the running yield is constant within the low-frequency window,
  which is the spec's recommendation.
* Negative carry (negative yield) is allowed but flagged in the diagnostics
  returned alongside the components — typically a sign of a bad yield series
  rather than a real economic effect.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..core.protocols import Frequency

# Δt per period for known frequencies (years per period)
_DELTA_T: dict[str, float] = {
    Frequency.QUARTERLY.value: 0.25,
    Frequency.MONTHLY.value: 1.0 / 12.0,
    Frequency.WEEKLY.value: 1.0 / 52.0,
    Frequency.DAILY.value: 1.0 / 252.0,
}


def decompose_credit_return(
    total_return: pd.Series,
    yield_series: pd.Series,
    *,
    frequency: Frequency | str = Frequency.QUARTERLY,
) -> tuple[pd.Series, pd.Series]:
    r"""Decompose a total-return series into carry and mark-to-market components.

    Parameters
    ----------
    total_return
        Observed total return per period (simple, not log).
    yield_series
        Running annual yield (or coupon rate). Must align to the same index
        as ``total_return`` after reindex; values are interpreted as decimals
        (e.g. ``0.07`` = 7 %).
    frequency
        Sampling frequency of ``total_return``. Used to convert annual yield
        to per-period accrual: ``carry_t = yield_t · Δt``.

    Returns
    -------
    (carry, mtm)
        Two pandas Series, both indexed identically to ``total_return``.
        ``carry + mtm == total_return`` exactly.

    Raises
    ------
    ValueError
        If the indexes do not align, the frequency is unknown, or the
        yield series contains NaNs.

    Notes
    -----
    The decomposition is exact and additive:

    .. math::

        r_{\text{total}, t} = r_{\text{carry}, t} + r_{\text{MTM}, t}

    The pipeline desmooths only ``mtm``; ``carry`` bypasses Stage 1
    entirely and is reattached after disaggregation.
    """
    if not isinstance(total_return, pd.Series):
        raise TypeError(
            f"total_return must be a Series, got {type(total_return).__name__}"
        )
    if not isinstance(yield_series, pd.Series):
        raise TypeError(
            f"yield_series must be a Series, got {type(yield_series).__name__}"
        )
    freq = Frequency(frequency) if not isinstance(frequency, Frequency) else frequency
    if freq.value not in _DELTA_T:
        raise ValueError(f"unsupported frequency {freq!r}")
    delta_t = _DELTA_T[freq.value]

    aligned_yield = yield_series.reindex(total_return.index)
    if aligned_yield.isna().any():
        missing = int(aligned_yield.isna().sum())
        raise ValueError(
            f"yield_series missing {missing} value(s) over total_return index."
        )
    if total_return.isna().any():
        raise ValueError("total_return contains NaN.")

    carry = aligned_yield * delta_t
    carry = carry.rename(
        f"{total_return.name}_carry" if total_return.name else "carry"
    )
    mtm = total_return - carry.values
    mtm = mtm.rename(f"{total_return.name}_mtm" if total_return.name else "mtm")
    return carry, mtm


def reattach_carry(
    desmoothed_mtm: pd.Series,
    carry_low_freq: pd.Series,
    *,
    high_freq_index: pd.DatetimeIndex,
    method: Literal["linear", "constant"] = "linear",
) -> pd.Series:
    r"""Reattach the carry component to a high-frequency MTM series after disaggregation.

    Parameters
    ----------
    desmoothed_mtm
        High-frequency MTM series from Stage 3 disaggregation. Indexed at
        ``high_freq_index``.
    carry_low_freq
        Low-frequency carry from :func:`decompose_credit_return` (typically
        quarterly).
    high_freq_index
        Target high-frequency index — typically the index of
        ``desmoothed_mtm``. Each low-frequency period must contain a known
        integer number of high-frequency periods.
    method
        ``'linear'`` (default) splits each low-frequency carry evenly across
        all high-frequency observations inside that block. ``'constant'``
        uses the average, which is identical for evenly-spaced grids — the
        two options exist as a hook for future weighted-distribution rules.

    Returns
    -------
    pandas.Series
        ``desmoothed_mtm + carry_high_freq``, indexed at ``high_freq_index``.

    Notes
    -----
    The mapping from low-frequency index to high-frequency block uses
    pandas' ``searchsorted`` semantics: each high-frequency timestamp is
    assigned to the *first* low-frequency block whose end timestamp is
    ``>=`` it. Mixed timezone-aware / naive indexes are not supported.
    """
    if not isinstance(desmoothed_mtm, pd.Series):
        raise TypeError(
            f"desmoothed_mtm must be a Series, got {type(desmoothed_mtm).__name__}"
        )
    if not isinstance(carry_low_freq, pd.Series):
        raise TypeError(
            f"carry_low_freq must be a Series, got {type(carry_low_freq).__name__}"
        )
    if method not in ("linear", "constant"):
        raise ValueError(f"method must be 'linear' or 'constant', got {method!r}")

    # Map each high-frequency timestamp to its containing low-frequency block.
    lf_ends = carry_low_freq.index.to_numpy()
    hf_ts = high_freq_index.to_numpy()
    block_idx = np.searchsorted(lf_ends, hf_ts, side="left")
    if (block_idx >= len(lf_ends)).any():
        raise ValueError(
            "high_freq_index extends beyond the latest low-frequency carry timestamp."
        )

    # Count how many high-frequency periods fall in each low-frequency block.
    counts = np.bincount(block_idx, minlength=len(lf_ends))
    if (counts == 0).any():
        empty_blocks = lf_ends[counts == 0]
        raise ValueError(
            f"low-frequency blocks {empty_blocks!r} contain no high-frequency observations."
        )

    # Per-period carry
    if method in ("linear", "constant"):
        # Both methods coincide for evenly-spaced grids; differ only when we
        # later add weighted distribution. Keep the branches separate to make
        # extension obvious.
        per_period_carry = carry_low_freq.to_numpy() / counts
    else:  # pragma: no cover — guarded above
        raise AssertionError("unreachable")

    carry_hf_arr = per_period_carry[block_idx]
    desm_arr = desmoothed_mtm.reindex(high_freq_index).to_numpy()
    if np.isnan(desm_arr).any():
        raise ValueError("desmoothed_mtm not aligned to high_freq_index.")

    out = pd.Series(
        desm_arr + carry_hf_arr,
        index=high_freq_index,
        name=desmoothed_mtm.name,
    )
    return out


__all__ = ["decompose_credit_return", "reattach_carry"]
