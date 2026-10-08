r"""
The information set available at a nowcast date.

This module is the correctness foundation of the whole ``nowcast`` package.
Every nowcasting model receives an :class:`InformationSet` and *nothing else*.
The untruncated data never becomes an attribute of that object, so there is no
code path by which a model — or a careless future edit to a model — can reach
an observation it should not have seen.

The information set at a date ``as_of`` is defined to be exactly:

* public factor observations with timestamp ``<= as_of``;
* reported private index values whose **publication date** ``<= as_of``
  — *not* whose reference quarter ends before ``as_of``;
* optional early-signal observations with timestamp ``<= as_of``.

The distinction in the second bullet is the whole point. A private markets
index for 2026Q2 has a reference quarter that ended on 30 June 2026 but is not
published until roughly October. On 7 October 2026 the quarter is *closed and
unpublished*, which is precisely the object this package nowcasts.

Data modes
----------
**Vintage mode** (preferred). The caller supplies a long frame ``vintages``
with columns ``[quarter, release_date, value]``, one row per published value of
each quarter *including revisions*. The value known at ``as_of`` for quarter
``Q`` is the row with the latest ``release_date <= as_of``. Lagged predictors
therefore carry the vintage a forecaster would actually have had, not the
final revised number.

**Lag-rule mode** (fallback). The caller supplies only the final revised series
plus a ``publication_lag`` (default 100 calendar days after quarter end).
Quarter ``Q`` is treated as known once ``Q.end_time + lag <= as_of``. This is
*pseudo-real-time*: the values are the revised ones, used as if they had been
available at first release. Revisions in private markets indices are
mean-reverting as late-reporting funds arrive, which makes the revised series
systematically easier to predict than the first print — so validation in this
mode is optimistic, and :func:`build_information_set` says so via
:class:`NowcastDataModeWarning`.

Horizons and the ragged edge
----------------------------
``horizon h`` is the number of quarters between the last published quarter and
the quarter being nowcast. In the motivating example — ``as_of`` = 7 Oct 2026,
last published quarter 2026Q1 — 2026Q2 is ``h=1`` and 2026Q3 is ``h=2``. The
open quarter 2026Q4 is excluded unless ``include_open_quarter=True``.

References
----------
.. [1] Cohen, Mantoan, Nesheim, de Paula, Turrell & Yang (2023) —
       "Nowcasting using regression on signatures," arXiv:2305.10256v3.
       Section 2 on the real-time information set and the ragged edge.
.. [2] Croushore & Stark (2001) — "A real-time data set for macroeconomists,"
       *Journal of Econometrics*. The canonical treatment of why
       pseudo-real-time validation flatters a model.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..core.config import AssetClassConfig, FallbackPolicy

__all__ = [
    "DEFAULT_PUBLICATION_LAG_DAYS",
    "EvaluationTarget",
    "InformationSet",
    "NowcastDataModeWarning",
    "build_information_set",
    "first_print_values",
    "latest_values",
    "quarterly_from_high_frequency",
    "resolve_factor_columns",
]


DEFAULT_PUBLICATION_LAG_DAYS = 100
"""Calendar days after quarter end at which a quarter is assumed published."""

EvaluationTarget = Literal["first_print", "latest"]

_VINTAGE_COLUMNS = ("quarter", "release_date", "value")


class NowcastDataModeWarning(UserWarning):
    """Emitted when the data mode makes validation optimistic.

    Raised as an exception instead under :attr:`FallbackPolicy.STRICT`, since a
    strict run asking for a defensible number should not silently receive a
    pseudo-real-time one.
    """


# ──────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────


def _as_quarter_index(index: pd.Index, *, what: str) -> pd.PeriodIndex:
    """Coerce an index of quarter labels to a quarterly :class:`pd.PeriodIndex`.

    Accepts a ``PeriodIndex`` (any quarterly freq alias), a ``DatetimeIndex``
    of quarter-end (or in-quarter) timestamps, or an object index of strings
    such as ``'2026Q2'``.

    Parameters
    ----------
    index
        The index to coerce.
    what
        Name used in error messages.

    Returns
    -------
    pd.PeriodIndex
        Quarterly periods, in the same order as ``index``.

    Raises
    ------
    TypeError
        If ``index`` cannot be interpreted as quarterly labels.
    """
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


def _as_timestamp(value: Any, *, what: str) -> pd.Timestamp:
    """Coerce to a tz-naive :class:`pd.Timestamp`, raising a clear error."""
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    if pd.isna(ts):
        raise ValueError(f"{what} must be a valid timestamp, got {value!r}.")
    return ts


def _as_lag(
    publication_lag: int | float | pd.Timedelta | pd.DateOffset | None,
) -> pd.Timedelta | pd.DateOffset | None:
    """Normalise a publication lag to something addable to a ``Timestamp``.

    An integer or float is interpreted as **calendar days**, matching the
    module default of 100 days. ``Timedelta`` and ``DateOffset`` pass through.
    """
    if publication_lag is None:
        return None
    if isinstance(publication_lag, (pd.Timedelta, pd.DateOffset)):
        return publication_lag
    if isinstance(publication_lag, (int, float, np.integer, np.floating)):
        if publication_lag < 0:
            raise ValueError(
                f"publication_lag must be >= 0 days, got {publication_lag!r}."
            )
        return pd.Timedelta(days=float(publication_lag))
    raise TypeError(
        "publication_lag must be days (int/float), Timedelta, DateOffset or None, "
        f"got {type(publication_lag).__name__}."
    )


def _last_closed_quarter(as_of: pd.Timestamp) -> pd.Period:
    """Return the last quarter that has fully ended at or before ``as_of``."""
    current = pd.Period(as_of, freq="Q")
    if as_of >= current.end_time:
        return current
    return current - 1


def resolve_factor_columns(
    config: AssetClassConfig | None,
    available: Sequence[str] | pd.Index,
    requested: Sequence[str] | None = None,
) -> list[str]:
    r"""Decide which columns of a factor panel a strategy should use.

    The parent library has a documented naming tension: ``AssetClassConfig``
    carries both ``beta_priors`` (keyed by *economic* factor names such as
    ``'equity_market'``) and ``default_factors`` (data column names such as
    ``'SP500'``). The code as built resolves this in favour of
    **``beta_priors`` keys being the actual factor column names** — see
    ``ar1_bayesian._validate_priors``, which requires a prior for *every*
    column of the supplied ``factor_returns``, and ``tests/test_pipeline_hf.py``,
    whose fixture sets ``default_factors=('equity_market', 'smb')`` to match its
    priors. ``default_factors`` is, in the shipped code, unused metadata.

    This helper respects that resolution while still honouring
    ``default_factors`` when it happens to name real columns, so a config
    written to either convention works:

    1. ``requested``, if given — all names must exist.
    2. ``beta_priors`` keys present in ``available``.
    3. ``default_factors`` entries present in ``available``.

    Parameters
    ----------
    config
        Strategy configuration, or ``None`` to require ``requested``.
    available
        Column names actually present in the factor panel.
    requested
        Explicit override.

    Returns
    -------
    list[str]
        Factor column names, in a deterministic order.

    Raises
    ------
    KeyError
        If ``requested`` names a column that does not exist, or if neither
        ``beta_priors`` nor ``default_factors`` matches any available column.
    """
    available_list = list(available)
    available_set = set(available_list)

    if requested is not None:
        requested = list(requested)
        missing = [c for c in requested if c not in available_set]
        if missing:
            raise KeyError(
                f"requested factor columns {missing!r} are not in the factor panel "
                f"(available: {available_list!r})."
            )
        if not requested:
            raise KeyError("requested factor columns is empty.")
        return requested

    if config is None:
        raise KeyError("resolve_factor_columns needs either a config or `requested`.")

    by_prior = [c for c in config.beta_priors if c in available_set]
    if by_prior:
        return by_prior

    by_default = [c for c in config.default_factors if c in available_set]
    if by_default:
        return by_default

    raise KeyError(
        f"none of beta_priors={list(config.beta_priors)!r} or "
        f"default_factors={list(config.default_factors)!r} match the factor panel "
        f"(available: {available_list!r}). Pass `requested` explicitly."
    )


def quarterly_from_high_frequency(
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    r"""Compound high-frequency factor returns into quarterly returns.

    Uses the library convention :math:`\prod(1 + r) - 1`, computed in log
    space as ``expm1(sum(log1p(r)))`` for numerical stability, consistent with
    :func:`utils.returns.compound_returns`.

    Parameters
    ----------
    factors
        Daily or monthly simple returns with a ``DatetimeIndex``. Must already
        be truncated to the information set.

    Returns
    -------
    quarterly : pd.DataFrame
        Compounded returns indexed by quarterly ``pd.PeriodIndex``.
    n_obs : pd.Series
        Number of high-frequency observations contributing to each quarter.
    coverage : pd.Series
        Fraction of each quarter's calendar span covered by the available
        observations, ``(last_obs - quarter_start) / (quarter_end - quarter_start)``.
        Equals ~1.0 for a quarter whose data runs to quarter end and is < 1 for
        the ragged edge. Reported as a diagnostic; completeness itself is
        decided by the calendar (``quarter.end_time <= as_of``), since
        truncation guarantees a closed quarter has all of its observations.

    Notes
    -----
    A quarter with any ``NaN`` in a factor column yields ``NaN`` for that
    column, rather than silently compounding a shorter window. Callers decide
    whether to drop or impute; models in this package drop.
    """
    if not isinstance(factors, pd.DataFrame):
        raise TypeError(
            f"factors must be a DataFrame, got {type(factors).__name__}."
        )
    if not isinstance(factors.index, pd.DatetimeIndex):
        raise TypeError("factors must have a DatetimeIndex.")
    if factors.empty:
        empty_q = pd.PeriodIndex([], freq="Q")
        return (
            pd.DataFrame(index=empty_q, columns=factors.columns, dtype=float),
            pd.Series(index=empty_q, dtype=int),
            pd.Series(index=empty_q, dtype=float),
        )

    arr = factors.to_numpy(dtype=float)
    if np.any(arr <= -1.0):
        raise ValueError(
            "factor returns contain values <= -1; cannot compound in log space."
        )

    quarters = pd.PeriodIndex(factors.index, freq="Q")
    log1p = pd.DataFrame(
        np.log1p(arr), index=factors.index, columns=factors.columns
    )
    grouped = log1p.groupby(quarters, sort=True)
    quarterly = np.expm1(grouped.sum(min_count=1))
    quarterly.index = pd.PeriodIndex(quarterly.index, freq="Q")
    quarterly.index.name = None

    n_obs = factors.groupby(quarters, sort=True).size()
    n_obs.index = pd.PeriodIndex(n_obs.index, freq="Q")

    last_obs = pd.Series(factors.index, index=quarters).groupby(level=0).max()
    last_obs.index = pd.PeriodIndex(last_obs.index, freq="Q")
    starts = last_obs.index.start_time
    ends = last_obs.index.end_time
    span = (ends - starts).to_numpy(dtype="timedelta64[ns]").astype(float)
    elapsed = (last_obs.to_numpy() - starts.to_numpy()).astype(
        "timedelta64[ns]"
    ).astype(float)
    coverage = pd.Series(
        np.clip(elapsed / np.where(span > 0, span, np.nan), 0.0, 1.0),
        index=last_obs.index,
        name="coverage",
    )
    return quarterly, n_obs.rename("n_obs"), coverage


# ──────────────────────────────────────────────────────────────────
# Vintage utilities
# ──────────────────────────────────────────────────────────────────


def _validate_vintages(vintages: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a long vintage frame.

    Returns a copy with ``quarter`` as quarterly periods, ``release_date`` as
    timestamps, ``value`` as float, stably sorted by ``(quarter, release_date)``
    with the original row order preserved within ties.
    """
    if not isinstance(vintages, pd.DataFrame):
        raise TypeError(
            f"vintages must be a DataFrame, got {type(vintages).__name__}."
        )
    missing = [c for c in _VINTAGE_COLUMNS if c not in vintages.columns]
    if missing:
        raise KeyError(
            f"vintages missing required columns {missing!r}; "
            f"expected {list(_VINTAGE_COLUMNS)}."
        )
    out = vintages.loc[:, list(_VINTAGE_COLUMNS)].copy()
    out["quarter"] = _as_quarter_index(
        pd.Index(out["quarter"]), what="vintages['quarter']"
    )
    out["release_date"] = pd.to_datetime(out["release_date"])
    if out["release_date"].dt.tz is not None:
        out["release_date"] = out["release_date"].dt.tz_localize(None)
    out["value"] = out["value"].astype(float)

    if out["release_date"].isna().any():
        raise ValueError("vintages['release_date'] contains NaT.")
    if out["value"].isna().any():
        raise ValueError("vintages['value'] contains NaN.")

    # A release cannot precede the quarter it reports on.
    quarter_ends = pd.PeriodIndex(out["quarter"]).end_time
    early = out["release_date"].to_numpy() < quarter_ends.to_numpy()
    if early.any():
        bad = out.loc[early, ["quarter", "release_date"]].head(3)
        raise ValueError(
            "vintages contains releases dated before the end of the quarter they "
            f"report on, which would leak future information:\n{bad}"
        )

    # Stable sort keeps original order within identical (quarter, release_date).
    out["_order"] = np.arange(len(out))
    out = out.sort_values(
        ["quarter", "release_date", "_order"], kind="stable"
    ).drop(columns="_order")
    return out.reset_index(drop=True)


def _known_from_vintages(
    vintages: pd.DataFrame, as_of: pd.Timestamp
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Latest release of each quarter as at ``as_of``.

    Parameters
    ----------
    vintages
        Normalised vintage frame (output of :func:`_validate_vintages`).
    as_of
        Information-set date.

    Returns
    -------
    values : pd.Series
        Value known at ``as_of``, indexed by quarter.
    release_dates : pd.Series
        The ``release_date`` each known value came from.
    n_releases : pd.Series
        Number of releases seen at ``as_of`` for each quarter — ``1`` means the
        value in hand is still a first print.
    """
    visible = vintages.loc[vintages["release_date"] <= as_of]
    if visible.empty:
        empty_q = pd.PeriodIndex([], freq="Q")
        return (
            pd.Series(index=empty_q, dtype=float, name="reported"),
            pd.Series(index=empty_q, dtype="datetime64[ns]", name="release_date"),
            pd.Series(index=empty_q, dtype=int, name="n_releases"),
        )
    # `visible` is already sorted by (quarter, release_date, original order),
    # so `.last()` is the latest release and ties resolve to the later row.
    grouped = visible.groupby("quarter", sort=True)
    values = grouped["value"].last()
    release_dates = grouped["release_date"].last()
    n_releases = grouped.size()
    for obj, name in (
        (values, "reported"),
        (release_dates, "release_date"),
        (n_releases, "n_releases"),
    ):
        obj.index = pd.PeriodIndex(obj.index, freq="Q")
        obj.index.name = None
        obj.name = name
    return values, release_dates, n_releases


def first_print_values(vintages: pd.DataFrame) -> pd.Series:
    """Earliest published value of each quarter, for scoring nowcasts.

    Parameters
    ----------
    vintages
        Long frame ``[quarter, release_date, value]``.

    Returns
    -------
    pd.Series
        First print per quarter, indexed by quarterly ``pd.PeriodIndex``.

    Notes
    -----
    This reads the **full, untruncated** vintage table and is therefore for the
    evaluation harness only — it answers "what was actually printed", which is
    the realisation a nowcast is scored against. Never call it on data destined
    for a model: it is not part of any information set.
    """
    v = _validate_vintages(vintages)
    out = v.groupby("quarter", sort=True)["value"].first()
    out.index = pd.PeriodIndex(out.index, freq="Q")
    out.index.name = None
    return out.rename("first_print")


def latest_values(vintages: pd.DataFrame) -> pd.Series:
    """Latest revised value of each quarter, for scoring nowcasts.

    Parameters
    ----------
    vintages
        Long frame ``[quarter, release_date, value]``.

    Returns
    -------
    pd.Series
        Latest revision per quarter, indexed by quarterly ``pd.PeriodIndex``.

    Notes
    -----
    Same caveat as :func:`first_print_values` — untruncated, evaluation only.
    """
    v = _validate_vintages(vintages)
    out = v.groupby("quarter", sort=True)["value"].last()
    out.index = pd.PeriodIndex(out.index, freq="Q")
    out.index.name = None
    return out.rename("latest")


# ──────────────────────────────────────────────────────────────────
# InformationSet
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InformationSet:
    r"""Everything knowable at a nowcast date, and nothing else.

    Built only by :func:`build_information_set`, which truncates every input at
    ``as_of`` and stores private copies. Models receive this object and never
    the caller's frames, so a model cannot reach a post-``as_of`` observation
    even by accident.

    Attributes
    ----------
    as_of
        The information-set date. All data below is dated at or before it.
    reported
        Reported quarterly returns *as known at* ``as_of``, indexed by
        quarterly ``pd.PeriodIndex``, sorted ascending and contiguous-checked.
        In vintage mode these are the vintages in force at ``as_of``, not the
        final revised values.
    release_dates
        Publication date each entry of ``reported`` came from. In lag-rule mode
        these are the *assumed* dates ``quarter_end + publication_lag``.
    n_releases
        Number of releases seen for each reported quarter at ``as_of``
        (vintage mode); ``1`` throughout in lag-rule mode.
    public_factors
        High-frequency (daily or monthly) public factor returns with
        ``index <= as_of``, restricted to ``factor_columns``.
    factors_quarterly
        ``public_factors`` compounded to quarters, :math:`\prod(1+r)-1`.
        Includes the ragged final quarter(s); use ``factor_quarter_complete``
        to tell which are whole.
    factor_quarter_complete
        ``True`` where the quarter had fully ended at ``as_of`` and therefore
        every one of its high-frequency observations is present.
    factor_quarter_coverage
        Fraction of each quarter's calendar span spanned by available data.
    factor_columns
        Factor column names in use, in a deterministic order.
    early_signals
        Optional irregular series that may proxy private marks (listed PE index
        levels, NAV announcements, preliminary prints), ``index <= as_of``.
        ``None`` when not supplied.
    data_mode
        ``'vintage'`` or ``'lag_rule'``.
    evaluation_target
        ``'first_print'`` or ``'latest'`` — which realisation nowcasts built
        from this information set should be scored against. Carried here so a
        results table always records it.
    publication_lag
        The lag used in lag-rule mode (and for vintage backfill); ``None`` in
        pure vintage mode.
    vintages
        The vintage rows with ``release_date <= as_of``, or ``None`` in
        lag-rule mode. Present so a model may use revision history as a
        feature; already truncated.
    max_horizon
        Largest horizon :meth:`unpublished_quarters` will return.
    diagnostics
        Provenance: data mode, counts, backfilled quarters, warnings emitted.
    """

    as_of: pd.Timestamp
    reported: pd.Series
    release_dates: pd.Series
    n_releases: pd.Series
    public_factors: pd.DataFrame
    factors_quarterly: pd.DataFrame
    factor_quarter_complete: pd.Series
    factor_quarter_coverage: pd.Series
    factor_columns: tuple[str, ...]
    early_signals: pd.DataFrame | None
    data_mode: Literal["vintage", "lag_rule"]
    evaluation_target: EvaluationTarget
    publication_lag: pd.Timedelta | pd.DateOffset | None
    vintages: pd.DataFrame | None
    max_horizon: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ── Published / unpublished bookkeeping ────────────────────────

    @property
    def published_quarters(self) -> pd.PeriodIndex:
        """Quarters with a value known at ``as_of``, ascending."""
        return pd.PeriodIndex(self.reported.index, freq="Q")

    @property
    def last_published_quarter(self) -> pd.Period | None:
        """Most recent quarter with a value known at ``as_of``."""
        if self.reported.empty:
            return None
        return self.published_quarters[-1]

    @property
    def last_closed_quarter(self) -> pd.Period:
        """Most recent quarter that had fully ended at ``as_of``."""
        return _last_closed_quarter(self.as_of)

    @property
    def n_published(self) -> int:
        """Number of quarters with a known value — the training sample size."""
        return int(self.reported.size)

    def unpublished_quarters(
        self, *, include_open_quarter: bool = False
    ) -> list[pd.Period]:
        """Closed-but-unpublished quarters at ``as_of``, in horizon order.

        Parameters
        ----------
        include_open_quarter
            If ``True``, also return the quarter currently in progress (whose
            factor data is necessarily partial). Default ``False``.

        Returns
        -------
        list[pd.Period]
            Ascending; element ``i`` has horizon ``i + 1``. Truncated to
            ``max_horizon`` entries.

        Notes
        -----
        With ``as_of = 2026-10-07`` and 2026Q1 as the last published quarter,
        this returns ``[2026Q2, 2026Q3]`` — horizons 1 and 2 — which is the
        motivating case in the module spec. 2026Q4 is still open and appears
        only under ``include_open_quarter=True``.
        """
        last_pub = self.last_published_quarter
        if last_pub is None:
            return []
        last = self.last_closed_quarter
        if include_open_quarter:
            last = pd.Period(self.as_of, freq="Q")
        if last <= last_pub:
            return []
        out = [last_pub + h for h in range(1, (last - last_pub).n + 1)]
        return out[: self.max_horizon]

    def horizon_of(self, quarter: pd.Period | str) -> int:
        """Horizon of ``quarter``: quarters since the last published one.

        Parameters
        ----------
        quarter
            Target quarter.

        Returns
        -------
        int
            ``1`` for the first unpublished quarter, ``2`` for the next, and so
            on. ``<= 0`` would mean the quarter is already published, which
            raises instead.

        Raises
        ------
        ValueError
            If ``quarter`` is already published at ``as_of``, or if nothing is
            published at all.
        """
        q = quarter if isinstance(quarter, pd.Period) else pd.Period(quarter, freq="Q")
        last_pub = self.last_published_quarter
        if last_pub is None:
            raise ValueError(
                f"no quarter is published at as_of={self.as_of.date()}; "
                "horizon is undefined."
            )
        h = (q - last_pub).n
        if h <= 0:
            raise ValueError(
                f"quarter {q} is already published at as_of={self.as_of.date()} "
                f"(last published {last_pub}); it is not a nowcast target."
            )
        return int(h)

    # ── Factor access ──────────────────────────────────────────────

    def factor_row(
        self, quarter: pd.Period | str, *, require_complete: bool = True
    ) -> pd.Series:
        """Compounded factor returns for one quarter.

        Parameters
        ----------
        quarter
            Target quarter.
        require_complete
            If ``True`` (default) raise when the quarter had not fully ended at
            ``as_of``, since its compounded return is then a partial-quarter
            figure that is not comparable to the training rows.

        Returns
        -------
        pd.Series
            One entry per factor column.

        Raises
        ------
        KeyError
            If no factor data exists for the quarter.
        ValueError
            If ``require_complete`` and the quarter is incomplete.
        """
        q = quarter if isinstance(quarter, pd.Period) else pd.Period(quarter, freq="Q")
        if q not in self.factors_quarterly.index:
            raise KeyError(
                f"no public factor data for {q} at as_of={self.as_of.date()}."
            )
        if require_complete and not bool(self.factor_quarter_complete.get(q, False)):
            cov = float(self.factor_quarter_coverage.get(q, float("nan")))
            raise ValueError(
                f"factor data for {q} is incomplete at as_of={self.as_of.date()} "
                f"(coverage {cov:.2f}); pass require_complete=False to use the "
                "partial-quarter compound."
            )
        return self.factors_quarterly.loc[q]

    def training_frame(self, *, n_lags: int = 1) -> pd.DataFrame:
        r"""Aligned design frame for the smoothing regression, published rows only.

        Builds the rows of

        .. math::
            s_Q = a + \sum_k b_k F_{k,Q} + \sum_{j=1}^{q} c_j s_{Q-j} + e_Q

        over quarters where the reported value, every factor column, and every
        required lag are all available inside the information set.

        Parameters
        ----------
        n_lags
            Number of lagged reported-return terms, ``q`` above.

        Returns
        -------
        pd.DataFrame
            Columns ``['s']`` + ``factor_columns`` + ``['s_lag1', ...]``,
            indexed by quarterly ``pd.PeriodIndex``. Rows with any missing
            entry are dropped.
        """
        if n_lags < 0:
            raise ValueError(f"n_lags must be >= 0, got {n_lags}.")
        s = self.reported.astype(float).rename("s")
        frame = pd.DataFrame({"s": s})
        factors = self.factors_quarterly.reindex(frame.index)
        complete = self.factor_quarter_complete.reindex(frame.index).fillna(False)
        mask = pd.DataFrame(
            np.repeat(complete.to_numpy()[:, None], factors.shape[1], axis=1),
            index=factors.index,
            columns=factors.columns,
        )
        factors = factors.where(mask)
        for col in self.factor_columns:
            frame[col] = factors[col]
        for j in range(1, n_lags + 1):
            frame[f"s_lag{j}"] = s.shift(j)
        return frame.dropna(how="any")

    # ── Representation ─────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        last_pub = self.last_published_quarter
        unpub = self.unpublished_quarters()
        return (
            f"InformationSet(as_of={self.as_of.date()}, mode={self.data_mode!r}, "
            f"n_published={self.n_published}, last_published={last_pub}, "
            f"unpublished={[str(q) for q in unpub]}, "
            f"factors={list(self.factor_columns)!r})"
        )


# ──────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────


def build_information_set(
    reported: pd.Series | None,
    public_factors: pd.DataFrame,
    as_of: pd.Timestamp | str,
    vintages: pd.DataFrame | None = None,
    publication_lag: int | float | pd.Timedelta | pd.DateOffset | None = None,
    early_signals: pd.DataFrame | pd.Series | None = None,
    *,
    factor_columns: Sequence[str] | None = None,
    config: AssetClassConfig | None = None,
    evaluation_target: EvaluationTarget | None = None,
    max_horizon: int = 4,
    include_open_quarter: bool = False,
    vintage_backfill: bool = True,
    fallback_policy: FallbackPolicy | str = FallbackPolicy.WARN,
) -> InformationSet:
    """Build the information set available at ``as_of``.

    This is the only constructor of :class:`InformationSet`. It truncates every
    input at ``as_of`` and stores copies, so the returned object contains no
    reference to post-``as_of`` data and mutating the caller's frames afterwards
    cannot change a nowcast.

    Parameters
    ----------
    reported
        Reported quarterly returns of the private index. Index may be a
        quarterly ``PeriodIndex``, a ``DatetimeIndex``, or string quarter
        labels. In vintage mode this is optional (pass ``None``) and is used
        only to backfill quarters that predate the vintage table; in lag-rule
        mode it is required.
    public_factors
        Daily or monthly public factor simple returns with a ``DatetimeIndex``.
    as_of
        The nowcast date. The information set is everything dated at or before
        it, with reported values gated on *publication* date.
    vintages
        Long frame ``[quarter, release_date, value]``, one row per published
        value including revisions. Supplying it selects **vintage mode**, the
        preferred path.
    publication_lag
        Days after quarter end at which a quarter is assumed published
        (default :data:`DEFAULT_PUBLICATION_LAG_DAYS` = 100 when ``vintages``
        is absent). Also accepts ``Timedelta`` / ``DateOffset``. Used in
        lag-rule mode, and for vintage backfill.
    early_signals
        Optional irregularly timestamped series that may proxy private marks.
        A ``Series`` is promoted to a one-column frame.
    factor_columns
        Explicit factor column selection. Overrides ``config``.
    config
        Strategy config from which to resolve factor columns — see
        :func:`resolve_factor_columns` for how the ``beta_priors`` /
        ``default_factors`` tension is resolved. If both this and
        ``factor_columns`` are ``None``, every column of ``public_factors`` is
        used.
    evaluation_target
        ``'first_print'`` (default in vintage mode) or ``'latest'`` (the only
        option in lag-rule mode, where no revision history exists).
    max_horizon
        Largest horizon :meth:`InformationSet.unpublished_quarters` reports.
    include_open_quarter
        Recorded in diagnostics and used to validate ``max_horizon`` against
        the ragged edge; the per-call default for
        :meth:`InformationSet.unpublished_quarters` is still ``False``.
    vintage_backfill
        In vintage mode, treat quarters *earlier* than the vintage table's
        first quarter as known from ``reported`` under the lag rule. Default
        ``True``: vintage tables for private indices typically start years
        after the index itself, and silently discarding that history is worse
        than using it under a flagged lag rule. Backfilled quarters are listed
        in ``diagnostics['backfilled_quarters']``.
    fallback_policy
        ``'warn'`` (default), ``'strict'`` or ``'auto'``. Under ``'strict'``,
        the lag-rule and backfill warnings escalate to
        :class:`NowcastDataModeWarning` raised as an exception.

    Returns
    -------
    InformationSet

    Raises
    ------
    ValueError
        If no data precedes ``as_of``, if the reported series has duplicate or
        non-monotonic quarters, or if a strict run would rely on
        pseudo-real-time data.
    KeyError
        If factor columns cannot be resolved.

    Notes
    -----
    **Which reported values are known at** ``as_of``. This is the decision the
    whole module rests on, and it is made in exactly one place — here.

    *Vintage mode.* For each quarter, take the rows with
    ``release_date <= as_of`` and keep the one with the latest ``release_date``
    (ties resolve to the later row in the caller's original ordering). That
    value, and not the final revised one, is what enters ``reported``. A
    quarter with no visible release is simply absent, which is what makes it a
    nowcast target.

    *Lag-rule mode.* Quarter ``Q`` is known iff
    ``Q.end_time + publication_lag <= as_of``. The value used is the final
    revised one from ``reported``, which is why this mode warns.

    Examples
    --------
    >>> info = build_information_set(                      # doctest: +SKIP
    ...     reported=pe_index,
    ...     public_factors=daily_factors,
    ...     as_of="2026-10-07",
    ...     vintages=pe_vintages,
    ... )
    >>> info.last_published_quarter, info.unpublished_quarters()  # doctest: +SKIP
    (Period('2026Q1', 'Q-DEC'), [Period('2026Q2', 'Q-DEC'), Period('2026Q3', 'Q-DEC')])
    """
    policy = (
        fallback_policy
        if isinstance(fallback_policy, FallbackPolicy)
        else FallbackPolicy(fallback_policy)
    )
    as_of_ts = _as_timestamp(as_of, what="as_of")
    if max_horizon < 1:
        raise ValueError(f"max_horizon must be >= 1, got {max_horizon}.")

    emitted: list[str] = []

    def _flag(message: str) -> None:
        """Warn, or raise under a strict policy."""
        emitted.append(message)
        if policy is FallbackPolicy.STRICT:
            raise NowcastDataModeWarning(message)
        warnings.warn(message, NowcastDataModeWarning, stacklevel=3)

    # ── Reported series (final revised; may be None in vintage mode) ──
    final_revised: pd.Series | None = None
    if reported is not None:
        if not isinstance(reported, pd.Series):
            raise TypeError(
                f"reported must be a Series, got {type(reported).__name__}."
            )
        final_revised = reported.astype(float).copy()
        final_revised.index = _as_quarter_index(
            final_revised.index, what="reported"
        )
        if final_revised.index.has_duplicates:
            dup = final_revised.index[final_revised.index.duplicated()].unique()
            raise ValueError(f"reported has duplicate quarters {list(dup)!r}.")
        final_revised = final_revised.sort_index()
        final_revised = final_revised.dropna()

    # ── Mode selection ────────────────────────────────────────────
    lag = _as_lag(publication_lag)
    if vintages is not None:
        data_mode: Literal["vintage", "lag_rule"] = "vintage"
        if evaluation_target is None:
            evaluation_target = "first_print"
    else:
        data_mode = "lag_rule"
        if final_revised is None:
            raise ValueError(
                "lag-rule mode requires `reported`; pass it, or supply `vintages` "
                "to use vintage mode."
            )
        if lag is None:
            lag = pd.Timedelta(days=DEFAULT_PUBLICATION_LAG_DAYS)
        if evaluation_target is None:
            evaluation_target = "latest"
        elif evaluation_target == "first_print":
            raise ValueError(
                "evaluation_target='first_print' needs a vintage table; lag-rule "
                "mode has only the final revised series."
            )
        _flag(
            "Nowcast information set built in lag-rule mode: the final revised "
            "index values are being used as if they had been available at first "
            "release. Private markets index revisions are mean-reverting as "
            "late-reporting funds arrive, so the revised series is systematically "
            "easier to predict than the first print — validation in this mode is "
            "pseudo-real-time and likely optimistic. Supply `vintages` "
            "([quarter, release_date, value]) for a real-time evaluation."
        )

    if evaluation_target not in ("first_print", "latest"):
        raise ValueError(
            "evaluation_target must be 'first_print' or 'latest', got "
            f"{evaluation_target!r}."
        )

    # ── Known reported values at as_of ────────────────────────────
    backfilled: list[str] = []
    truncated_vintages: pd.DataFrame | None = None

    if data_mode == "vintage":
        v_all = _validate_vintages(vintages)
        truncated_vintages = v_all.loc[v_all["release_date"] <= as_of_ts].reset_index(
            drop=True
        )
        known, release_dates, n_releases = _known_from_vintages(v_all, as_of_ts)

        if vintage_backfill and final_revised is not None and len(v_all):
            first_vintage_quarter = pd.PeriodIndex(v_all["quarter"], freq="Q").min()
            lag_bf = lag if lag is not None else pd.Timedelta(
                days=DEFAULT_PUBLICATION_LAG_DAYS
            )
            pre = final_revised.loc[final_revised.index < first_vintage_quarter]
            if not pre.empty:
                pre_release = pd.PeriodIndex(pre.index, freq="Q").end_time + lag_bf
                visible = pre_release <= as_of_ts
                pre = pre.loc[visible]
                pre_release = pre_release[visible]
                if not pre.empty:
                    backfilled = [str(q) for q in pre.index]
                    known = pd.concat([pre.rename("reported"), known]).sort_index()
                    release_dates = pd.concat(
                        [
                            pd.Series(
                                pre_release, index=pre.index, name="release_date"
                            ),
                            release_dates,
                        ]
                    ).sort_index()
                    n_releases = pd.concat(
                        [
                            pd.Series(1, index=pre.index, name="n_releases"),
                            n_releases,
                        ]
                    ).sort_index()
                    _flag(
                        f"Vintage table starts at {first_vintage_quarter}; "
                        f"{len(backfilled)} earlier quarter(s) were backfilled from "
                        "the final revised series under the "
                        f"{lag_bf} publication-lag rule. Those rows are "
                        "pseudo-real-time. Pass vintage_backfill=False to drop them "
                        "instead (at the cost of training history)."
                    )
    else:
        assert final_revised is not None and lag is not None
        release_all = pd.PeriodIndex(final_revised.index, freq="Q").end_time + lag
        visible = release_all <= as_of_ts
        known = final_revised.loc[visible].rename("reported")
        release_dates = pd.Series(
            release_all[visible], index=known.index, name="release_date"
        )
        n_releases = pd.Series(1, index=known.index, name="n_releases")

    known.index = pd.PeriodIndex(known.index, freq="Q")
    known = known.sort_index()
    release_dates = release_dates.reindex(known.index)
    n_releases = n_releases.reindex(known.index).fillna(1).astype(int)

    if known.empty:
        raise ValueError(
            f"no reported value is published at as_of={as_of_ts.date()} "
            f"(data_mode={data_mode!r}); the information set would be empty."
        )

    gaps = _interior_gaps(pd.PeriodIndex(known.index, freq="Q"))
    if gaps:
        _flag(
            f"reported series has interior gaps at {gaps[:5]!r}"
            f"{' …' if len(gaps) > 5 else ''}; lagged-return features will drop "
            "the rows adjacent to each gap."
        )

    # ── Public factors ────────────────────────────────────────────
    if not isinstance(public_factors, pd.DataFrame):
        raise TypeError(
            f"public_factors must be a DataFrame, got "
            f"{type(public_factors).__name__}."
        )
    if not isinstance(public_factors.index, pd.DatetimeIndex):
        raise TypeError("public_factors must have a DatetimeIndex.")
    cols = (
        list(factor_columns)
        if factor_columns is not None
        else (
            resolve_factor_columns(config, public_factors.columns)
            if config is not None
            else list(public_factors.columns)
        )
    )
    missing_cols = [c for c in cols if c not in public_factors.columns]
    if missing_cols:
        raise KeyError(f"public_factors missing columns {missing_cols!r}.")
    if not cols:
        raise KeyError("no factor columns selected.")

    pf = public_factors.loc[:, cols].copy()
    if pf.index.tz is not None:
        pf.index = pf.index.tz_localize(None)
    pf = pf.sort_index()
    pf = pf.loc[pf.index <= as_of_ts]
    if pf.empty:
        raise ValueError(
            f"no public factor observation at or before as_of={as_of_ts.date()}."
        )

    factors_q, n_obs_q, coverage_q = quarterly_from_high_frequency(pf)
    complete_q = pd.Series(
        [bool(q.end_time <= as_of_ts) for q in factors_q.index],
        index=factors_q.index,
        name="complete",
    )

    # ── Early signals ─────────────────────────────────────────────
    es: pd.DataFrame | None = None
    if early_signals is not None:
        if isinstance(early_signals, pd.Series):
            es = early_signals.to_frame(
                name=early_signals.name or "early_signal"
            )
        elif isinstance(early_signals, pd.DataFrame):
            es = early_signals.copy()
        else:
            raise TypeError(
                "early_signals must be a Series, DataFrame or None, got "
                f"{type(early_signals).__name__}."
            )
        if not isinstance(es.index, pd.DatetimeIndex):
            raise TypeError("early_signals must have a DatetimeIndex.")
        if es.index.tz is not None:
            es.index = es.index.tz_localize(None)
        es = es.sort_index()
        es = es.loc[es.index <= as_of_ts]
        if es.empty:
            _flag(
                f"early_signals has no observation at or before "
                f"as_of={as_of_ts.date()}; it will be ignored."
            )
            es = None

    # ── Assemble ──────────────────────────────────────────────────
    info = InformationSet(
        as_of=as_of_ts,
        reported=known.rename("reported"),
        release_dates=release_dates,
        n_releases=n_releases,
        public_factors=pf,
        factors_quarterly=factors_q,
        factor_quarter_complete=complete_q,
        factor_quarter_coverage=coverage_q,
        factor_columns=tuple(cols),
        early_signals=es,
        data_mode=data_mode,
        evaluation_target=evaluation_target,
        publication_lag=lag,
        vintages=truncated_vintages,
        max_horizon=int(max_horizon),
        diagnostics={
            "data_mode": data_mode,
            "evaluation_target": evaluation_target,
            "as_of": as_of_ts,
            "n_published": int(known.size),
            "first_published_quarter": str(known.index[0]),
            "last_published_quarter": str(known.index[-1]),
            "last_closed_quarter": str(_last_closed_quarter(as_of_ts)),
            "publication_lag": lag,
            "n_factor_observations": int(len(pf)),
            "last_factor_observation": pf.index[-1],
            "factor_columns": tuple(cols),
            "n_factor_quarters_complete": int(complete_q.sum()),
            "factor_quarter_n_obs": n_obs_q.to_dict(),
            "backfilled_quarters": backfilled,
            "include_open_quarter": bool(include_open_quarter),
            "early_signal_columns": tuple(es.columns) if es is not None else (),
            "fallback_policy": policy.value,
            "warnings": emitted,
        },
    )
    return info


def _interior_gaps(quarters: pd.PeriodIndex) -> list[str]:
    """Quarters missing between the first and last entry of ``quarters``."""
    if quarters.size < 2:
        return []
    full = pd.period_range(quarters[0], quarters[-1], freq="Q")
    return [str(q) for q in full.difference(quarters)]
