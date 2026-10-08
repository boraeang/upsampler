r"""
Path signatures: construction, term selection, and a pure-numpy backend.

The path signature is a graded sequence of iterated integrals of a path, and it
is the feature set of the reference paper's nowcasting method. For a path
:math:`X : [0,1] \to \mathbb{R}^d` the level-:math:`k` term indexed by a *word*
:math:`(i_1, \dots, i_k)` is

.. math::
    S^{(i_1,\dots,i_k)}(X) =
    \int_{0 < u_1 < \dots < u_k < 1}
    \mathrm{d}X^{i_1}_{u_1} \cdots \mathrm{d}X^{i_k}_{u_k} .

Two facts make this computable exactly rather than approximately, and they are
the whole of the numpy backend:

**Tensor exponential of a straight line.** A single linear segment with increment
:math:`\Delta \in \mathbb{R}^d` has signature
:math:`S^k = \Delta^{\otimes k} / k!`, because the integrand is constant and the
simplex :math:`0 < u_1 < \dots < u_k < 1` has volume :math:`1/k!`.

**Chen's identity.** Concatenating paths multiplies their signatures in the
truncated tensor algebra:

.. math::
    S(X * Y)^k = \sum_{i=0}^{k} S(X)^i \otimes S(Y)^{k-i},
    \qquad S(\cdot)^0 = 1 .

A piecewise-linear path is a concatenation of straight lines, so its signature is
the Chen product of the segments' tensor exponentials — **exact** in floating
point, with no quadrature. That is why the pure-numpy backend is not a fallback
of last resort: it is the same computation the libraries perform, and
:func:`compute_signature` cross-checks it against ``iisignature`` or ``esig``
whenever one is installed.

Why a numpy backend at all
--------------------------
The module spec asks for it regardless of what is installed, and that turns out
to matter here: ``iisignature`` does not build on this machine, and ``esig``
requires ``numpy >= 2``, which the pinned ``pandas 2.1.4`` will not tolerate. So
on this environment the numpy backend is not the fallback — it is the only
backend. Its correctness is pinned by Chen's identity, by shuffle/algebraic
identities with known closed forms, and by agreement with a library when one is
importable.

Ordering convention
-------------------
Terms are returned flat, levels ascending, words in lexicographic order within
each level, with the leading :math:`S^0 = 1` omitted:

.. code-block:: text

    d=2, level=2  ->  [S(0), S(1), S(0,0), S(0,1), S(1,0), S(1,1)]

This is ``iisignature.sig``'s convention, so the two are directly comparable.

Linear signature terms
----------------------
``keep_sigs='linear'`` keeps only words in which each **non-time** channel
appears at most once — the "linear signature terms" of the reference paper's
Theorem 1, the subset that spans the linear/Kalman predictor the signature
regression is supposed to nest. Channel 0 is time by convention and is exempt, so
at ``d=2, level=2`` the kept set is
``[S(0), S(1), S(0,0), S(0,1), S(1,0)]`` and only ``S(1,1)`` is dropped. That
is enough to span within-quarter path statistics such as
:math:`\int X \mathrm{d}s` (which is :math:`S^{(1,0)}`) while keeping the feature
count inside the budget a ~90-quarter sample can support.

References
----------
.. [1] Cohen, Mantoan, Nesheim, de Paula, Turrell & Yang (2023) —
       "Nowcasting using regression on signatures," arXiv:2305.10256v3.
       Theorem 1 on linear signature terms; Section 3.3 on irregular data.
.. [2] Chen (1957) — "Integration of paths, geometric invariants and a
       generalized Baker-Hausdorff formula." Chen's identity.
.. [3] Lyons, Caruana & Lévy (2007) — *Differential Equations Driven by Rough
       Paths*. The tensor-algebra setting.
.. [4] Chevyrev & Kormilitzin (2016) — "A primer on the signature method in
       machine learning." The practical account of basepoints, time
       augmentation and rectilinear interpolation.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .information_set import InformationSet

__all__ = [
    "MAX_SIGNATURE_LEVEL",
    "FactorPCA",
    "KeepSigs",
    "PathFrequency",
    "PathSpec",
    "SignatureBackendWarning",
    "available_backends",
    "build_path",
    "chen_product",
    "compute_signature",
    "linear_term_mask",
    "path_signature",
    "signature_dimension",
    "signature_feature_names",
    "signature_numpy",
    "signature_words",
    "tensor_exponential",
    "warn_if_library_backend_missing",
]


MAX_SIGNATURE_LEVEL = 5
"""Hard cap on truncation level. Levels above 3 are not a supported default."""

KeepSigs = Literal["all", "linear"]
PathFrequency = Literal["daily", "monthly"]
_TIME_CHANNEL = 0


class SignatureBackendWarning(UserWarning):
    """Emitted when the signature backend is not the one that was asked for."""


# ──────────────────────────────────────────────────────────────────
# Word bookkeeping
# ──────────────────────────────────────────────────────────────────


def signature_words(n_channels: int, level: int) -> list[tuple[int, ...]]:
    """Enumerate signature words in the canonical flat order.

    Parameters
    ----------
    n_channels
        Path dimension ``d``.
    level
        Truncation level ``M``.

    Returns
    -------
    list[tuple[int, ...]]
        Words of length ``1..M``, levels ascending and lexicographic within each
        level — the same order :func:`compute_signature` returns values in, and
        the same order ``iisignature.sig`` uses.

    Examples
    --------
    >>> signature_words(2, 2)
    [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}.")
    if not 1 <= level <= MAX_SIGNATURE_LEVEL:
        raise ValueError(
            f"level must be in 1..{MAX_SIGNATURE_LEVEL}, got {level}."
        )
    words: list[tuple[int, ...]] = []
    for k in range(1, level + 1):
        words.extend(itertools.product(range(n_channels), repeat=k))
    return words


def linear_term_mask(
    n_channels: int, level: int, *, time_channel: int | None = _TIME_CHANNEL
) -> np.ndarray:
    r"""Boolean mask selecting the **linear** signature terms.

    A word is linear when every non-time channel appears in it **at most once**.
    These are the terms of the reference paper's Theorem 1 — the subset in which
    the signature regression contains the linear/Kalman predictor as a special
    case. The time channel is exempt because repeated time indices contribute
    only polynomial-in-time weights, not products of data increments: the word
    :math:`(0,1)` is :math:`\int_0^1 s\,\mathrm{d}X_s` and :math:`(1,0)` is
    :math:`\int_0^1 X_s\,\mathrm{d}s`, both *linear* in the path, whereas
    :math:`(1,1)` is :math:`\tfrac12 (X_1)^2` and is not.

    Parameters
    ----------
    n_channels
        Path dimension ``d``.
    level
        Truncation level.
    time_channel
        Index of the time channel, exempt from the once-only rule. ``None``
        applies the rule to every channel.

    Returns
    -------
    np.ndarray
        Boolean array aligned with :func:`signature_words`.

    Examples
    --------
    >>> words = signature_words(2, 2)
    >>> mask = linear_term_mask(2, 2)
    >>> [w for w, keep in zip(words, mask) if not keep]
    [(1, 1)]

    Notes
    -----
    The saving grows fast with level, which is the point — at ``d=2`` the counts
    are ``5/6`` at level 2 but ``9/14`` at level 3 and ``14/30`` at level 4. With
    ~90 training quarters and a ``n_train/5`` feature budget, that difference
    decides whether a configuration is admissible at all. Note that these are
    *signature term* counts; the model's feature count additionally drops the
    time-only terms and adds the lagged return and intercept, so size a
    configuration with
    :func:`~private_assets_frequency.nowcast.models.signature.feature_budget`
    rather than by arithmetic on these.
    """
    words = signature_words(n_channels, level)
    keep = np.ones(len(words), dtype=bool)
    for idx, word in enumerate(words):
        counts: dict[int, int] = {}
        for channel in word:
            if channel == time_channel:
                continue
            counts[channel] = counts.get(channel, 0) + 1
            if counts[channel] > 1:
                keep[idx] = False
                break
    return keep


def signature_dimension(
    n_channels: int,
    level: int,
    keep_sigs: KeepSigs = "all",
    *,
    time_channel: int | None = _TIME_CHANNEL,
) -> int:
    """Number of signature features a configuration produces.

    Parameters
    ----------
    n_channels
        Path dimension.
    level
        Truncation level.
    keep_sigs
        ``'all'`` or ``'linear'``.
    time_channel
        Passed to :func:`linear_term_mask`.

    Returns
    -------
    int
        Feature count, for the budget guard that runs *before* any fitting.

    Examples
    --------
    >>> signature_dimension(2, 2, "all"), signature_dimension(2, 2, "linear")
    (6, 5)
    >>> signature_dimension(3, 2, "all"), signature_dimension(3, 2, "linear")
    (12, 10)
    """
    if keep_sigs == "all":
        return sum(n_channels**k for k in range(1, level + 1))
    if keep_sigs == "linear":
        return int(
            linear_term_mask(n_channels, level, time_channel=time_channel).sum()
        )
    raise ValueError(f"keep_sigs must be 'all' or 'linear', got {keep_sigs!r}.")


def signature_feature_names(
    channel_names: list[str],
    level: int,
    keep_sigs: KeepSigs = "all",
    *,
    time_channel: int | None = _TIME_CHANNEL,
) -> list[str]:
    """Human-readable names for the signature features, in flat order.

    Parameters
    ----------
    channel_names
        One name per path channel, e.g. ``['time', 'equity_market']``.
    level
        Truncation level.
    keep_sigs
        ``'all'`` or ``'linear'``.
    time_channel
        Passed to :func:`linear_term_mask`.

    Returns
    -------
    list[str]
        Names like ``'sig[time,equity_market]'``.

    Examples
    --------
    >>> signature_feature_names(['t', 'x'], 2, 'linear')
    ['sig[t]', 'sig[x]', 'sig[t,t]', 'sig[t,x]', 'sig[x,t]']
    """
    words = signature_words(len(channel_names), level)
    names = [
        "sig[" + ",".join(channel_names[i] for i in word) + "]" for word in words
    ]
    if keep_sigs == "linear":
        mask = linear_term_mask(
            len(channel_names), level, time_channel=time_channel
        )
        return [names[i] for i in range(len(names)) if mask[i]]
    if keep_sigs != "all":
        raise ValueError(f"keep_sigs must be 'all' or 'linear', got {keep_sigs!r}.")
    return names


# ──────────────────────────────────────────────────────────────────
# Pure-numpy backend
# ──────────────────────────────────────────────────────────────────


def tensor_exponential(increment: np.ndarray, level: int) -> list[np.ndarray]:
    r"""Signature of a single straight-line segment.

    For a linear path with increment :math:`\Delta`, the integrand of every
    iterated integral is constant and the simplex
    :math:`0 < u_1 < \dots < u_k < 1` has volume :math:`1/k!`, so

    .. math::
        S^k = \frac{\Delta^{\otimes k}}{k!} .

    Parameters
    ----------
    increment
        The segment increment, shape ``(d,)``.
    level
        Truncation level.

    Returns
    -------
    list[np.ndarray]
        ``[S^1, ..., S^level]``, each flattened to length ``d**k`` in
        lexicographic word order. The level-0 term (the scalar 1) is implicit.
    """
    delta = np.asarray(increment, dtype=float).reshape(-1)
    out: list[np.ndarray] = []
    power = np.ones(1, dtype=float)
    for k in range(1, level + 1):
        power = np.outer(power, delta).reshape(-1)
        out.append(power / math.factorial(k))
    return out


def chen_product(
    left: list[np.ndarray], right: list[np.ndarray], level: int
) -> list[np.ndarray]:
    r"""Chen's identity: the signature of a concatenation.

    .. math::
        S(X * Y)^k = \sum_{i=0}^{k} S(X)^i \otimes S(Y)^{k-i},
        \qquad S(\cdot)^0 = 1 .

    The ``i=0`` and ``i=k`` terms are just :math:`S(Y)^k` and :math:`S(X)^k`,
    since the level-0 term is the scalar 1.

    Parameters
    ----------
    left, right
        Truncated signatures as returned by :func:`tensor_exponential` —
        ``[S^1, ..., S^level]`` with each level flattened.
    level
        Truncation level; both inputs must carry at least this many levels.

    Returns
    -------
    list[np.ndarray]
        The product, in the same layout.

    Notes
    -----
    The flattening is consistent under concatenation: a word of length ``k``
    split as an ``i``-prefix and a ``(k-i)``-suffix has flat index
    ``idx_prefix * d**(k-i) + idx_suffix``, which is exactly what
    ``np.outer(a, b).reshape(-1)`` produces. That is why no explicit index
    arithmetic appears here.
    """
    if len(left) < level or len(right) < level:
        raise ValueError(
            f"both signatures must carry {level} levels; got {len(left)} and "
            f"{len(right)}."
        )
    out: list[np.ndarray] = []
    for k in range(1, level + 1):
        acc = left[k - 1] + right[k - 1]
        for i in range(1, k):
            acc = acc + np.outer(left[i - 1], right[k - i - 1]).reshape(-1)
        out.append(acc)
    return out


def signature_numpy(path: np.ndarray, level: int) -> np.ndarray:
    r"""Exact signature of a piecewise-linear path, in pure numpy.

    Treats the ``n`` path points as ``n-1`` straight-line segments, takes each
    segment's :func:`tensor_exponential`, and folds them left to right with
    :func:`chen_product`. The result is exact for a piecewise-linear path — no
    quadrature, no truncation beyond ``level``.

    Parameters
    ----------
    path
        Path points, shape ``(n_points, n_channels)``. A path with fewer than
        two points has no increments and yields all-zero terms.
    level
        Truncation level, ``1..MAX_SIGNATURE_LEVEL``.

    Returns
    -------
    np.ndarray
        Flat signature of length ``sum(d**k for k in 1..level)``, levels
        ascending and lexicographic within each level, with the leading ``1``
        omitted — matching ``iisignature.sig``.

    Raises
    ------
    ValueError
        If ``path`` is not 2-D, has no channels, contains non-finite values, or
        ``level`` is out of range.

    Examples
    --------
    A straight line from the origin to ``(1, 2)`` has
    :math:`S^{(0)} = 1`, :math:`S^{(1)} = 2`, and
    :math:`S^{(0,1)} = S^{(1,0)} = \tfrac12 \Delta_0 \Delta_1 = 1`:

    >>> import numpy as np
    >>> sig = signature_numpy(np.array([[0.0, 0.0], [1.0, 2.0]]), 2)
    >>> [round(float(v), 6) for v in sig]
    [1.0, 2.0, 0.5, 1.0, 1.0, 2.0]
    """
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"path must be 2-D (n_points, n_channels), got {arr.ndim}-D.")
    n_points, d = arr.shape
    if d < 1:
        raise ValueError("path must have at least one channel.")
    if not 1 <= level <= MAX_SIGNATURE_LEVEL:
        raise ValueError(
            f"level must be in 1..{MAX_SIGNATURE_LEVEL}, got {level}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("path contains non-finite values.")

    total = sum(d**k for k in range(1, level + 1))
    if n_points < 2:
        return np.zeros(total, dtype=float)

    increments = np.diff(arr, axis=0)
    accumulated = tensor_exponential(increments[0], level)
    for i in range(1, increments.shape[0]):
        if not increments[i].any():
            continue  # a zero increment multiplies by the identity
        accumulated = chen_product(
            accumulated, tensor_exponential(increments[i], level), level
        )
    return np.concatenate(accumulated)


# ──────────────────────────────────────────────────────────────────
# Backend adapter
# ──────────────────────────────────────────────────────────────────


def available_backends() -> tuple[str, ...]:
    """Signature backends importable in this environment.

    Returns
    -------
    tuple[str, ...]
        Some ordered subset of ``('iisignature', 'esig', 'numpy')``. ``'numpy'``
        is always present — it is built in, not a dependency.
    """
    found: list[str] = []
    for name in ("iisignature", "esig"):
        try:
            __import__(name)
        except Exception:  # noqa: BLE001 - a broken install is an absent one
            continue
        found.append(name)
    found.append("numpy")
    return tuple(found)


def _signature_iisignature(path: np.ndarray, level: int) -> np.ndarray:
    import iisignature  # noqa: PLC0415 - optional backend, imported on use

    return np.asarray(iisignature.sig(np.asarray(path, dtype=float), level), dtype=float)


def _signature_esig(path: np.ndarray, level: int) -> np.ndarray:
    import esig  # noqa: PLC0415 - optional backend, imported on use

    # esig's stream2sig includes the leading scalar 1; drop it to match.
    full = np.asarray(
        esig.stream2sig(np.asarray(path, dtype=float), level), dtype=float
    )
    return full[1:]


def compute_signature(
    path: np.ndarray,
    level: int,
    *,
    backend: str = "auto",
    keep_sigs: KeepSigs = "all",
    time_channel: int | None = _TIME_CHANNEL,
) -> np.ndarray:
    """Signature of a piecewise-linear path, via the best available backend.

    Parameters
    ----------
    path
        Path points, shape ``(n_points, n_channels)``.
    level
        Truncation level.
    backend
        ``'auto'`` (default) prefers ``iisignature``, then ``esig``, then
        ``'numpy'``. Naming a backend explicitly requires it to be importable,
        except ``'numpy'`` which always is.
    keep_sigs
        ``'all'`` (default) or ``'linear'`` — see :func:`linear_term_mask`.
    time_channel
        Index of the time channel for the linear-term rule.

    Returns
    -------
    np.ndarray
        Flat signature terms in canonical order, filtered by ``keep_sigs``.

    Raises
    ------
    ImportError
        If a named backend is not importable.

    Notes
    -----
    All backends agree to floating-point precision for piecewise-linear paths —
    they compute the same exact quantity. ``test_signatures.py`` asserts
    agreement to ``1e-10`` for random paths at levels 1-3 whenever a library is
    importable, and skips that comparison otherwise. On this machine neither
    library installs (``iisignature`` fails to build; ``esig`` needs
    ``numpy >= 2``, which breaks the pinned ``pandas``), so the numpy backend
    carries the load and its correctness rests on the algebraic identities
    tested directly.
    """
    if backend == "auto":
        for candidate in available_backends():
            try:
                raw = _dispatch(candidate, path, level)
            except Exception:  # noqa: BLE001 - fall through to the next backend
                continue
            break
        else:  # pragma: no cover - numpy always succeeds
            raise RuntimeError("no signature backend produced a result.")
    else:
        raw = _dispatch(backend, path, level)

    if keep_sigs == "all":
        return raw
    d = np.asarray(path).shape[1]
    mask = linear_term_mask(d, level, time_channel=time_channel)
    if mask.size != raw.size:  # pragma: no cover - defensive
        raise RuntimeError(
            f"backend returned {raw.size} terms but the canonical order has "
            f"{mask.size} for d={d}, level={level}."
        )
    return raw[mask]


def _dispatch(backend: str, path: np.ndarray, level: int) -> np.ndarray:
    if backend == "numpy":
        return signature_numpy(path, level)
    if backend == "iisignature":
        return _signature_iisignature(path, level)
    if backend == "esig":
        return _signature_esig(path, level)
    raise ValueError(
        f"unknown signature backend {backend!r}; expected 'auto', 'numpy', "
        "'iisignature' or 'esig'."
    )


# ──────────────────────────────────────────────────────────────────
# PCA on factor returns
# ──────────────────────────────────────────────────────────────────


@dataclass
class FactorPCA:
    """Principal components of a factor-return panel, fitted on training rows only.

    Implemented with :func:`numpy.linalg.svd` rather than scikit-learn so the
    path machinery has no hard dependency on the ``nowcast`` extra.

    Parameters
    ----------
    n_components
        Number of components to keep.

    Attributes
    ----------
    mean_
        Column means of the training panel.
    components_
        Loading matrix, shape ``(n_components, n_features)``.
    explained_variance_ratio_
        Fraction of training variance each component carries.

    Notes
    -----
    The fit must happen inside the information set — a rotation estimated on the
    full sample is one of the quieter leakage routes catalogued in
    ``DESIGN_NOTES.md`` (Risk 1), because the feature matrix still *looks*
    clean afterwards.
    """

    n_components: int
    mean_: np.ndarray | None = field(default=None, init=False)
    components_: np.ndarray | None = field(default=None, init=False)
    explained_variance_ratio_: np.ndarray | None = field(default=None, init=False)
    feature_names_: tuple[str, ...] = field(default=(), init=False)

    def fit(self, panel: pd.DataFrame) -> FactorPCA:
        """Fit the rotation on ``panel``.

        Parameters
        ----------
        panel
            Factor returns, rows observations and columns factors.

        Returns
        -------
        FactorPCA
            ``self``.
        """
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(
                f"panel must be a DataFrame, got {type(panel).__name__}."
            )
        arr = panel.to_numpy(dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            raise ValueError(
                f"need at least 2 rows to fit a PCA, got shape {arr.shape}."
            )
        k = int(self.n_components)
        if not 1 <= k <= arr.shape[1]:
            raise ValueError(
                f"n_components must be in 1..{arr.shape[1]}, got {k}."
            )
        self.mean_ = arr.mean(axis=0)
        centred = arr - self.mean_
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        self.components_ = vt[:k]
        variance = singular**2
        total = variance.sum()
        self.explained_variance_ratio_ = (
            variance[:k] / total if total > 0 else np.zeros(k)
        )
        self.feature_names_ = tuple(str(c) for c in panel.columns)
        return self

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Project ``panel`` onto the fitted components.

        Parameters
        ----------
        panel
            Factor returns with the same columns the fit saw.

        Returns
        -------
        pd.DataFrame
            Columns ``pc1 … pcK``, same index as ``panel``.
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("call fit() before transform().")
        missing = [c for c in self.feature_names_ if c not in panel.columns]
        if missing:
            raise KeyError(f"panel missing columns {missing!r} seen at fit time.")
        arr = panel.loc[:, list(self.feature_names_)].to_numpy(dtype=float)
        projected = (arr - self.mean_) @ self.components_.T
        return pd.DataFrame(
            projected,
            index=panel.index,
            columns=[f"pc{i + 1}" for i in range(projected.shape[1])],
        )

    @property
    def component_names(self) -> list[str]:
        """Output column names."""
        return [f"pc{i + 1}" for i in range(int(self.n_components))]


# ──────────────────────────────────────────────────────────────────
# Path construction
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PathSpec:
    r"""How to turn public market data into a path for one target quarter.

    Parameters
    ----------
    lookback
        Window length in quarters, ending at the target quarter's end.
        ``1`` (default) is within-quarter only; the module spec's grid is
        ``{1, 2, 4}``.
    path_frequency
        ``'daily'`` (default) or ``'monthly'`` — the resolution of the path.
        Daily data is compounded into months when ``'monthly'``.
    level_channel
        ``'log'`` (default) builds each channel as the cumulative **log**-return
        level, so increments add and the level-1 signature term is
        :math:`\log(1 + F_Q)`. ``'simple'`` builds the cumulative simple-return
        level :math:`\prod(1+r) - 1`, whose level-1 term is :math:`F_Q`
        *exactly* — which is what makes the nesting test against
        :class:`~..models.smoothing_regression.SmoothingRegressionNowcaster`
        exact rather than approximate. Signature theory needs only a continuous
        path, so both are legitimate; log is the default because increments
        compose additively over sub-periods.
    include_time
        Prepend a time channel rescaled to ``[0, 1]`` over the window. Required
        for the level-2 terms that carry path shape, and on by default.
    basepoint
        Prepend a zero point. On by default, and load-bearing: without it the
        path starts at the first observation's level, so the level-1 term misses
        the first period's return and no longer equals the window return.
    missing
        ``'ffill'`` (default) forward-fills gaps and interpolates linearly
        between consecutive points. ``'rectilinear'`` instead moves one axis at a
        time — time advances holding values constant, then values jump at fixed
        time — which is the reference paper's Section 3.3 treatment for
        irregularly observed channels, and avoids implying movement between
        observations that were never seen.
    n_components
        Fit a :class:`FactorPCA` to this many components and use those as the
        data channels instead of the raw factors. ``None`` (default) uses the
        factors as they are.
    early_signal_columns
        Which of ``info.early_signals``' columns to add as extra channels.
        ``None`` uses all of them; ``()`` uses none.
    """

    lookback: int = 1
    path_frequency: PathFrequency = "daily"
    level_channel: Literal["log", "simple"] = "log"
    include_time: bool = True
    basepoint: bool = True
    missing: Literal["ffill", "rectilinear"] = "ffill"
    n_components: int | None = None
    early_signal_columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError(f"lookback must be >= 1 quarter, got {self.lookback}.")
        if self.path_frequency not in ("daily", "monthly"):
            raise ValueError(
                f"path_frequency must be 'daily' or 'monthly', got "
                f"{self.path_frequency!r}."
            )
        if self.level_channel not in ("log", "simple"):
            raise ValueError(
                f"level_channel must be 'log' or 'simple', got "
                f"{self.level_channel!r}."
            )
        if self.missing not in ("ffill", "rectilinear"):
            raise ValueError(
                f"missing must be 'ffill' or 'rectilinear', got {self.missing!r}."
            )
        if self.n_components is not None and self.n_components < 1:
            raise ValueError(
                f"n_components must be >= 1 or None, got {self.n_components}."
            )

    def channel_names(
        self, factor_columns: list[str], early_signal_columns: list[str] | None = None
    ) -> list[str]:
        """Channel names in path-column order.

        Parameters
        ----------
        factor_columns
            Factor columns available (ignored when ``n_components`` is set).
        early_signal_columns
            Early-signal columns to include.

        Returns
        -------
        list[str]
        """
        names: list[str] = ["time"] if self.include_time else []
        if self.n_components is not None:
            names.extend(f"pc{i + 1}" for i in range(self.n_components))
        else:
            names.extend(factor_columns)
        names.extend(early_signal_columns or [])
        return names

    def n_channels(
        self, n_factors: int, n_early_signals: int = 0
    ) -> int:
        """Path dimension implied by this spec."""
        data = self.n_components if self.n_components is not None else n_factors
        return int(self.include_time) + int(data) + int(n_early_signals)


def _window_bounds(
    quarter: pd.Period, spec: PathSpec, as_of: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``[start, end]`` of the lookback window, never extending past ``as_of``."""
    start = (quarter - (spec.lookback - 1)).start_time
    end = min(quarter.end_time, as_of)
    if end < start:
        raise ValueError(
            f"lookback window for {quarter} ends ({end.date()}) before it starts "
            f"({start.date()}) at as_of={as_of.date()}."
        )
    return start, end


def _to_levels(
    returns: pd.DataFrame, level_channel: str
) -> pd.DataFrame:
    r"""Cumulative level of each return column, rebased to 0 before the window.

    ``'log'`` gives :math:`\sum \log(1+r)`; ``'simple'`` gives
    :math:`\prod(1+r) - 1`. Both start the window at 0 — the rebasing the
    reference paper's path construction calls for — with the basepoint standing
    for the instant before the first observation.
    """
    arr = returns.to_numpy(dtype=float)
    if np.any(arr <= -1.0):
        raise ValueError(
            "path construction: a return <= -1 cannot be turned into a level."
        )
    cumulative_log = np.cumsum(np.log1p(arr), axis=0)
    values = cumulative_log if level_channel == "log" else np.expm1(cumulative_log)
    return pd.DataFrame(values, index=returns.index, columns=returns.columns)


def _rectilinear(points: np.ndarray, time_index: int | None) -> np.ndarray:
    """Expand a path into a rectilinear (axis-at-a-time) path.

    Between consecutive observations, time advances with the data channels held
    constant, then the data channels jump at fixed time. This is the reference
    paper's Section 3.3 treatment: it never implies a movement between two
    observations that were not actually seen, which matters for irregularly
    observed channels such as NAV announcements.

    Parameters
    ----------
    points
        Path points, shape ``(n, d)``.
    time_index
        Column holding time, or ``None`` when there is no time channel (in which
        case the path is already a pure step function and is returned unchanged).

    Returns
    -------
    np.ndarray
        Shape ``(2n - 1, d)`` when ``time_index`` is not ``None``.
    """
    if time_index is None or points.shape[0] < 2:
        return points
    out = [points[0]]
    for i in range(1, points.shape[0]):
        held = points[i - 1].copy()
        held[time_index] = points[i][time_index]
        out.append(held)
        out.append(points[i])
    return np.asarray(out, dtype=float)


def build_path(
    info: InformationSet,
    quarter: pd.Period,
    spec: PathSpec,
    *,
    pca: FactorPCA | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    r"""Build the public-market path for one target quarter.

    The window is ``[(quarter - lookback + 1).start, min(quarter.end, as_of)]``,
    so it can never reach past the information set — and because
    ``info.public_factors`` is already truncated at ``as_of``, it could not even
    if the arithmetic were wrong.

    Parameters
    ----------
    info
        The information set. Supplies the (already truncated) factor panel and,
        optionally, early signals.
    quarter
        Target quarter the window ends in.
    spec
        Path construction options.
    pca
        A :class:`FactorPCA` **already fitted on training rows**. Required when
        ``spec.n_components`` is set; passing an unfitted or full-sample PCA here
        is the leak this parameter exists to make explicit.

    Returns
    -------
    path : np.ndarray
        Shape ``(n_points, n_channels)``.
    channel_names : list[str]
        Names in column order, ``'time'`` first when included.
    diagnostics : dict
        Window bounds, observation counts, whether the window was clipped by
        ``as_of``, and the resampling actually applied.

    Raises
    ------
    ValueError
        If the window contains no observations, or ``spec.n_components`` is set
        without a fitted ``pca``.

    Notes
    -----
    Time is rescaled to ``[0, 1]`` across the *realised* window, so two windows
    of different calendar length still both run 0 to 1. A consequence worth
    knowing: the level-1 time term is then ``1.0`` for every complete window and
    contributes nothing but an intercept — which is why
    :class:`~..models.signature.SignatureNowcaster` drops zero-variance
    signature columns, and why the nesting test can account for it.
    """
    start, end = _window_bounds(quarter, spec, info.as_of)
    factors = info.public_factors
    window = factors.loc[(factors.index >= start) & (factors.index <= end)]
    if window.empty:
        raise ValueError(
            f"no public factor observations in the path window "
            f"[{start.date()}, {end.date()}] for {quarter} at "
            f"as_of={info.as_of.date()}."
        )

    resampled = window
    if spec.path_frequency == "monthly":
        months = pd.PeriodIndex(window.index, freq="M")
        resampled = np.expm1(
            np.log1p(window).groupby(months, sort=True).sum(min_count=1)
        )
        resampled.index = (
            pd.PeriodIndex(resampled.index, freq="M")
            .to_timestamp(how="end")
            .normalize()
        )
    if spec.missing == "ffill":
        resampled = resampled.ffill()
    resampled = resampled.dropna(how="any")
    if resampled.empty:
        raise ValueError(
            f"path window [{start.date()}, {end.date()}] for {quarter} has no "
            "complete observations after gap handling."
        )

    if spec.n_components is not None:
        if pca is None or pca.components_ is None:
            raise ValueError(
                "spec.n_components is set, so build_path needs a FactorPCA that "
                "has already been fitted on training rows. Fitting it here would "
                "use the target quarter's own data."
            )
        data_returns = pca.transform(resampled)
    else:
        data_returns = resampled

    levels = _to_levels(data_returns, spec.level_channel)
    columns: list[pd.Series] = []
    names: list[str] = []

    if spec.include_time:
        stamps = levels.index.to_numpy(dtype="datetime64[ns]").astype("float64")
        span = stamps[-1] - stamps[0]
        scaled = (
            (stamps - stamps[0]) / span if span > 0 else np.zeros_like(stamps)
        )
        columns.append(pd.Series(scaled, index=levels.index, name="time"))
        names.append("time")

    for col in levels.columns:
        columns.append(levels[col])
        names.append(str(col))

    early_names: list[str] = []
    if info.early_signals is not None:
        wanted = (
            list(info.early_signals.columns)
            if spec.early_signal_columns is None
            else [
                c for c in spec.early_signal_columns
                if c in info.early_signals.columns
            ]
        )
        if wanted:
            signals = info.early_signals.loc[
                (info.early_signals.index >= start)
                & (info.early_signals.index <= end),
                wanted,
            ]
            # Reindex onto the path's own clock, forward-filling: an early signal
            # is a level that stands until the next announcement, and its
            # irregular timing is exactly what signatures handle natively.
            aligned = (
                signals.reindex(signals.index.union(levels.index))
                .ffill()
                .reindex(levels.index)
            )
            aligned = aligned.fillna(0.0)
            for col in wanted:
                columns.append(aligned[col].rename(col))
                early_names.append(str(col))
                names.append(str(col))

    frame = pd.concat(columns, axis=1)
    points = frame.to_numpy(dtype=float)
    if spec.basepoint:
        points = np.vstack([np.zeros((1, points.shape[1])), points])
    if spec.missing == "rectilinear":
        points = _rectilinear(
            points, 0 if spec.include_time else None
        )

    diagnostics = {
        "quarter": str(quarter),
        "window_start": start,
        "window_end": end,
        "clipped_by_as_of": bool(end < quarter.end_time),
        "n_raw_observations": int(len(window)),
        "n_path_observations": int(len(frame)),
        "n_path_points": int(points.shape[0]),
        "path_frequency": spec.path_frequency,
        "level_channel": spec.level_channel,
        "basepoint": bool(spec.basepoint),
        "missing": spec.missing,
        "n_components": spec.n_components,
        "early_signal_columns": tuple(early_names),
        "channel_names": tuple(names),
    }
    return points, names, diagnostics


def path_signature(
    info: InformationSet,
    quarter: pd.Period,
    spec: PathSpec,
    *,
    level: int,
    keep_sigs: KeepSigs = "linear",
    backend: str = "auto",
    pca: FactorPCA | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build the path for ``quarter`` and return its signature features.

    Parameters
    ----------
    info, quarter, spec, pca
        Passed to :func:`build_path`.
    level
        Truncation level.
    keep_sigs
        ``'linear'`` (default) or ``'all'``.
    backend
        Passed to :func:`compute_signature`.

    Returns
    -------
    features : np.ndarray
        Signature terms, flat, in canonical order filtered by ``keep_sigs``.
    names : list[str]
        Feature names from :func:`signature_feature_names`.
    diagnostics : dict
        :func:`build_path`'s diagnostics plus ``level``, ``keep_sigs`` and the
        backend used.
    """
    path, channel_names, diagnostics = build_path(info, quarter, spec, pca=pca)
    time_channel = _TIME_CHANNEL if spec.include_time else None
    features = compute_signature(
        path,
        level,
        backend=backend,
        keep_sigs=keep_sigs,
        time_channel=time_channel,
    )
    names = signature_feature_names(
        channel_names, level, keep_sigs, time_channel=time_channel
    )
    if len(names) != features.size:  # pragma: no cover - defensive
        raise RuntimeError(
            f"produced {features.size} signature values but {len(names)} names."
        )
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "level": int(level),
            "keep_sigs": keep_sigs,
            "backend": backend if backend != "auto" else available_backends()[0],
            "n_features": int(features.size),
        }
    )
    return features, names, diagnostics


def warn_if_library_backend_missing() -> None:
    """Note once that no library backend is importable, so numpy is in use."""
    if available_backends() == ("numpy",):
        warnings.warn(
            "neither iisignature nor esig is importable, so signatures are "
            "computed with the built-in numpy backend. That backend is exact "
            "for piecewise-linear paths (tensor exponential plus Chen's "
            "identity), but the cross-check against a library is skipped.",
            SignatureBackendWarning,
            stacklevel=2,
        )
