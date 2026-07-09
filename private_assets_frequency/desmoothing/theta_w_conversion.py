r"""
Bidirectional conversion between the θ and w parameterisations of the
Rudin-Mao-Zhang-Fink (2019) reparameterised unsmoothing model.

Notation
--------
The paper uses two equivalent representations of the smoothing dynamics.

* **w parameterisation (Getmansky-Lo-Makarov / Conner / Pedersen-Page-He form):**
  the *observed* smooth return is a moving average of current and past *true*
  economic returns,

  .. math::

     \tilde r^o_t = \sum_{j=0}^{Q_w} w_j \, r^E_{t-j}
     \qquad (\text{Eq. 1 of Rudin et al.})

  with :math:`\sum_j w_j = 1`.

* **θ parameterisation (Stefek-Suryanarayanan / Rudin form):** the *true*
  return is a linear combination of the current and past *observed* returns,

  .. math::

     r^E_t = \sum_{j=0}^{Q_\theta} \theta_j \, \tilde r^o_{t-j}
     \qquad (\text{Eq. 3 of Rudin et al.})

  with :math:`\sum_j \theta_j = 1`.

In lag-operator terms Eq. 1 says :math:`\tilde r^o = W(L) \, r^E` and Eq. 3
says :math:`r^E = \Theta(L) \, \tilde r^o`, so the two polynomials are formal
inverses,

.. math::

    \Theta(L) \cdot W(L) = 1 .

A *finite* polynomial and its formal inverse cannot both be finite (except in
the trivial :math:`Q = 0` case where both are the scalar 1), so converting a
truncated :math:`\theta_0,\dots,\theta_Q` to :math:`w`-space produces an
*infinite* series that we truncate at a user-chosen order, and vice versa.
The tail is exponentially decaying whenever the roots of the source
polynomial lie outside the unit circle — which the paper's estimated θ vectors
comfortably satisfy for the reported cases (Q ∈ {1, 2}, small higher-lag
coefficients).

The recursion driving both directions is the standard formal-series
convolution: if :math:`A(L) \cdot B(L) = 1` and :math:`A_0 \neq 0`, then

.. math::

   b_0 &= 1 / a_0 \\
   b_k &= -\frac{1}{a_0} \sum_{j=1}^{\min(k, \deg A)} a_j \, b_{k-j},
   \qquad k \ge 1 .

This is implemented by :func:`_polynomial_inverse`; the public
:func:`theta_to_w` and :func:`w_to_theta` are thin wrappers that add
domain-specific validation and diagnostics.

References
----------
Rudin, A., Mao, J., Zhang, N. R., & Fink, A.-M. (2019). "Fitting Private
Equity into the Total Portfolio Framework." *Journal of Portfolio Management*
46(2), 60-77. See in particular Equations 1 and 3, and the discussion of the
equivalence between the two parameterisations.
Getmansky, M., Lo, A. W., & Makarov, I. (2004). "An econometric model of
serial correlation and illiquidity in hedge fund returns." *Journal of
Financial Economics* 74, 529-609.
Stefek, D., & Suryanarayanan, R. (2012). "Private and Public Real Estate:
What is the Link?" MSCI Research Insight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default truncation for the inverse series. 200 is generous: for a Q=1 model
# with |θ_1/θ_0| ≈ 0.5 the tail w_j decays like 0.5^j, so |w_200| ≈ 6e-61.
_DEFAULT_TRUNCATION_LAGS = 200

# Floor on |a_0| below which polynomial inversion is unstable; we raise rather
# than silently produce a garbage series.
_A0_MIN_ABS = 1e-12


# ──────────────────────────────────────────────────────────────────
# Result container for diagnostic-heavy callers
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConversionDiagnostics:
    """Truncation-quality diagnostics for a θ ↔ w conversion.

    Attributes
    ----------
    truncation_lags
        Number of lags after position 0 in the returned array; the array has
        length ``truncation_lags + 1``.
    tail_abs_max
        Maximum absolute value among the last ``max(5, (truncation_lags+1)//10)``
        entries of the truncated series. A small value means truncation error
        is negligible.
    sum
        Sum of the returned coefficients. Should be close to 1 when the source
        satisfies the sum-to-one constraint and truncation is adequate.
    sum_error
        ``|sum - 1|`` — direct measure of how well the sum constraint transfers
        through the (finite) truncation.
    truncation_warning
        Free-form warning string if the tail is not visibly negligible
        (``tail_abs_max > 1e-6``), else empty.
    """

    truncation_lags: int
    tail_abs_max: float
    sum: float
    sum_error: float
    truncation_warning: str = ""


# ──────────────────────────────────────────────────────────────────
# Core polynomial-inversion primitive
# ──────────────────────────────────────────────────────────────────


def _polynomial_inverse(
    a: np.ndarray,
    truncation_lags: int,
) -> np.ndarray:
    r"""Return the first ``truncation_lags + 1`` coefficients of ``1 / A(L)``.

    Given :math:`A(L) = a_0 + a_1 L + \dots + a_Q L^Q`, compute
    :math:`B(L) = 1 / A(L) = b_0 + b_1 L + b_2 L^2 + \dots` up to
    ``truncation_lags`` in the lag operator. The recursion is

    .. math::

       b_0 &= 1 / a_0 \\
       b_k &= -\frac{1}{a_0} \sum_{j=1}^{\min(k, Q)} a_j \, b_{k-j},
       \qquad k \ge 1 .

    Parameters
    ----------
    a
        One-dimensional coefficient array of the source polynomial. ``a[0]``
        must be non-zero (checked via ``_A0_MIN_ABS``).
    truncation_lags
        Non-negative integer. The returned array has length
        ``truncation_lags + 1``.

    Returns
    -------
    numpy.ndarray
        The truncated inverse-series coefficients.

    Raises
    ------
    ValueError
        If ``a`` is not one-dimensional, is empty, has ``|a[0]| < 1e-12``, or
        ``truncation_lags`` is negative.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"a must be 1-D, got ndim={a.ndim}")
    if a.size < 1:
        raise ValueError("a must have length >= 1")
    if abs(a[0]) < _A0_MIN_ABS:
        raise ValueError(
            f"a[0] must satisfy |a[0]| >= {_A0_MIN_ABS}; got {a[0]!r}. "
            "Polynomial inversion is undefined when the leading coefficient "
            "is zero (equivalent to no contemporaneous weight)."
        )
    if truncation_lags < 0:
        raise ValueError(f"truncation_lags must be >= 0, got {truncation_lags}")

    Q = a.size - 1
    N = truncation_lags + 1
    b = np.zeros(N, dtype=float)
    b[0] = 1.0 / a[0]
    for k in range(1, N):
        upper = min(k, Q)
        if upper >= 1:
            # b[k] = -(a[1]*b[k-1] + a[2]*b[k-2] + ... + a[upper]*b[k-upper]) / a[0]
            # b[k-upper:k] = [b[k-upper], ..., b[k-1]]; reversed → [b[k-1], ..., b[k-upper]]
            s = float(np.dot(a[1 : upper + 1], b[k - upper : k][::-1]))
        else:
            s = 0.0
        b[k] = -s / a[0]
    return b


# ──────────────────────────────────────────────────────────────────
# Public conversions
# ──────────────────────────────────────────────────────────────────


def theta_to_w(
    theta: np.ndarray,
    truncation_lags: int | None = None,
) -> np.ndarray:
    r"""Convert θ (Eq. 3) weights to w (Eq. 1) weights.

    Formally :math:`W(L) = 1 / \Theta(L)`. The inverse is an infinite formal
    series in general; only the first ``truncation_lags + 1`` coefficients are
    returned.

    Parameters
    ----------
    theta
        Coefficient array ``[θ_0, θ_1, ..., θ_Q]`` from Eq. 3.
    truncation_lags
        Maximum lag order to return. Defaults to 200. The tail is monitored
        via :func:`conversion_diagnostics`.

    Returns
    -------
    numpy.ndarray
        The truncated ``[w_0, w_1, ..., w_{truncation_lags}]`` coefficients.

    Notes
    -----
    * The sum constraint transfers: if :math:`\sum \theta_j = 1` then
      :math:`\sum_{j=0}^{\infty} w_j = 1`, but the finite-truncation sum
      differs from 1 by the tail mass. Use
      :func:`conversion_diagnostics` to quantify this.
    * When ``theta`` has length 1 (Q = 0, no unsmoothing) then
      ``w == [1/theta[0]]``, which equals ``[1.0]`` under the sum constraint
      ``θ_0 = 1``.

    Examples
    --------
    >>> import numpy as np
    >>> w = theta_to_w(np.array([1.5, -0.5]))
    >>> np.isclose(w[0], 2.0 / 3.0)
    True
    """
    theta = np.asarray(theta, dtype=float)
    if truncation_lags is None:
        truncation_lags = _DEFAULT_TRUNCATION_LAGS
    return _polynomial_inverse(theta, truncation_lags)


def w_to_theta(
    w: np.ndarray,
    truncation_lags: int | None = None,
) -> np.ndarray:
    r"""Convert w (Eq. 1) weights to θ (Eq. 3) weights.

    Formally :math:`\Theta(L) = 1 / W(L)`. Symmetric to :func:`theta_to_w`.

    Parameters
    ----------
    w
        Coefficient array ``[w_0, w_1, ..., w_{Q_w}]`` from Eq. 1.
    truncation_lags
        Maximum lag order to return. Defaults to 200.

    Returns
    -------
    numpy.ndarray
        The truncated ``[θ_0, θ_1, ..., θ_{truncation_lags}]`` coefficients.

    Notes
    -----
    If ``w`` has length 1 (no smoothing) then ``θ == [1/w[0]]``, which equals
    ``[1.0]`` when ``w`` satisfies the sum constraint ``w_0 = 1``.
    """
    w = np.asarray(w, dtype=float)
    if truncation_lags is None:
        truncation_lags = _DEFAULT_TRUNCATION_LAGS
    return _polynomial_inverse(w, truncation_lags)


# ──────────────────────────────────────────────────────────────────
# Constraint checks and round-trip diagnostics
# ──────────────────────────────────────────────────────────────────


def sum_to_one_error(coefficients: np.ndarray) -> float:
    """Return ``|Σ coefficients - 1|``.

    A convenience for the constraint :math:`\\sum_j \\theta_j = 1` in Eq. 3
    (equivalently :math:`\\sum_j w_j = 1` in Eq. 1). Values close to 0 mean
    the sum constraint holds; large values flag either model misspecification
    (before conversion) or aggressive truncation (after conversion).
    """
    coefficients = np.asarray(coefficients, dtype=float)
    return float(abs(coefficients.sum() - 1.0))


def conversion_diagnostics(
    converted: np.ndarray,
    *,
    tail_warn_threshold: float = 1e-6,
) -> ConversionDiagnostics:
    """Summarise the truncation quality of a converted coefficient vector.

    Inspects the tail of the returned series (the last ~10 % of entries) and
    reports the sum of the coefficients (which should equal 1 when the sum
    constraint holds and the truncation is generous enough).

    Parameters
    ----------
    converted
        Output of :func:`theta_to_w` or :func:`w_to_theta`.
    tail_warn_threshold
        Maximum tolerated absolute value of any tail entry; above this a
        warning string is populated.

    Returns
    -------
    ConversionDiagnostics
    """
    converted = np.asarray(converted, dtype=float)
    N = converted.size
    tail_len = max(5, N // 10) if N > 1 else 1
    tail_len = min(tail_len, N)
    tail_slice = converted[-tail_len:]
    tail_max = float(np.max(np.abs(tail_slice))) if tail_slice.size else 0.0
    total = float(converted.sum())
    warn = ""
    if tail_max > tail_warn_threshold:
        warn = (
            f"Truncation tail |{tail_max:.3e}| exceeds threshold "
            f"{tail_warn_threshold:.1e} — increase truncation_lags to reduce "
            "residual mass beyond the reported window."
        )
    return ConversionDiagnostics(
        truncation_lags=N - 1,
        tail_abs_max=tail_max,
        sum=total,
        sum_error=abs(total - 1.0),
        truncation_warning=warn,
    )


def roundtrip_error(
    theta: np.ndarray,
    *,
    truncation_lags: int = _DEFAULT_TRUNCATION_LAGS,
) -> float:
    """Maximum absolute recovery error under θ → w → θ.

    Runs ``theta_to_w`` then ``w_to_theta`` at ``truncation_lags`` in both
    directions and reports the largest absolute difference between the input
    ``theta`` and the leading ``len(theta)`` entries of the reconstruction.

    Parameters
    ----------
    theta
        Source coefficients (Eq. 3).
    truncation_lags
        Truncation used in both directions of the round trip.

    Returns
    -------
    float
    """
    theta = np.asarray(theta, dtype=float)
    w = theta_to_w(theta, truncation_lags=truncation_lags)
    theta_rec = w_to_theta(w, truncation_lags=truncation_lags)
    n = theta.size
    return float(np.max(np.abs(theta - theta_rec[:n])))


__all__ = [
    "ConversionDiagnostics",
    "conversion_diagnostics",
    "roundtrip_error",
    "sum_to_one_error",
    "theta_to_w",
    "w_to_theta",
]
