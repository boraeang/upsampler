"""Tests for the θ ↔ w bidirectional conversion utilities.

Verifies:

* the polynomial-inversion recursion against hand-computed values,
* round-trip fidelity θ → w → θ (and the symmetric w → θ → w),
* preservation of the sum-to-one constraint under generous truncation,
* the Q = 0 identity case,
* input-validation raises for zero leading coefficient and bad shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from private_assets_frequency.desmoothing.theta_w_conversion import (
    ConversionDiagnostics,
    conversion_diagnostics,
    roundtrip_error,
    sum_to_one_error,
    theta_to_w,
    w_to_theta,
)


# ──────────────────────────────────────────────────────────────────
# Analytic sanity checks
# ──────────────────────────────────────────────────────────────────


class TestPolynomialInversionAnalytics:
    """Compare to hand-computed values for small Q."""

    def test_q0_theta_is_identity(self) -> None:
        """θ = [1] ↦ w = [1, 0, 0, ...]. This is the no-unsmoothing base case."""
        w = theta_to_w(np.array([1.0]), truncation_lags=10)
        assert w.shape == (11,)
        assert w[0] == pytest.approx(1.0)
        # Q=0 means no non-zero lag coefficients in the inverse either
        assert np.allclose(w[1:], 0.0)

    def test_q0_w_is_identity(self) -> None:
        """w = [1] ↦ θ = [1, 0, 0, ...]. Symmetric case."""
        theta = w_to_theta(np.array([1.0]), truncation_lags=10)
        assert theta.shape == (11,)
        assert theta[0] == pytest.approx(1.0)
        assert np.allclose(theta[1:], 0.0)

    def test_q1_recursion_matches_closed_form(self) -> None:
        r"""For Θ(L) = θ_0 + θ_1 L the inverse is a geometric series:

        W(L) = 1/θ_0 · Σ_{k≥0} (-θ_1/θ_0)^k · L^k
        """
        theta0, theta1 = 1.2, -0.2
        w = theta_to_w(np.array([theta0, theta1]), truncation_lags=8)
        ratio = -theta1 / theta0
        expected = np.array([(1.0 / theta0) * ratio**k for k in range(9)])
        np.testing.assert_allclose(w, expected, atol=1e-12)

    def test_leading_coefficient_reciprocal(self) -> None:
        """w[0] = 1 / θ[0] regardless of higher-lag entries."""
        theta = np.array([0.7, 0.2, 0.1])
        w = theta_to_w(theta, truncation_lags=5)
        assert w[0] == pytest.approx(1.0 / 0.7)


# ──────────────────────────────────────────────────────────────────
# Round-trip fidelity
# ──────────────────────────────────────────────────────────────────


class TestRoundTrip:
    """θ → w → θ must recover the input under generous truncation."""

    @pytest.mark.parametrize(
        "theta",
        [
            np.array([1.0]),                                # Q=0
            np.array([1.5, -0.5]),                          # Q=1, sum=1
            np.array([1.2, -0.15, -0.05]),                  # Q=2, sum=1
            np.array([1.4, -0.3, -0.05, -0.05]),            # Q=3, sum=1
            np.array([1.5, -0.25, -0.15, -0.05, -0.05]),    # Q=4, sum=1
            np.array([2.0, -0.7, -0.3]),                    # heavy unsmoothing
        ],
    )
    def test_theta_to_w_to_theta_recovers(self, theta: np.ndarray) -> None:
        err = roundtrip_error(theta, truncation_lags=300)
        assert err < 1e-9, f"round-trip error {err:.3e} too large for theta={theta}"

    @pytest.mark.parametrize(
        "w",
        [
            np.array([0.6, 0.3, 0.1]),                       # classic MA(2) weights
            np.array([0.5, 0.25, 0.15, 0.1]),                # MA(3)
            np.array([0.4, 0.3, 0.2, 0.05, 0.05]),           # MA(4)
        ],
    )
    def test_w_to_theta_to_w_recovers(self, w: np.ndarray) -> None:
        theta = w_to_theta(w, truncation_lags=300)
        w_rec = theta_to_w(theta, truncation_lags=300)
        np.testing.assert_allclose(w, w_rec[: w.size], atol=1e-9)


# ──────────────────────────────────────────────────────────────────
# Sum-to-one constraint transfer
# ──────────────────────────────────────────────────────────────────


class TestSumConstraint:
    """Verify Σθ_j = 1 ⟺ Σw_j = 1 in the limit of no truncation."""

    def test_sum_error_helper(self) -> None:
        assert sum_to_one_error(np.array([0.6, 0.3, 0.1])) == pytest.approx(0.0)
        assert sum_to_one_error(np.array([0.5, 0.3, 0.1])) == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "theta",
        [
            np.array([1.5, -0.5]),
            np.array([1.2, -0.15, -0.05]),
            np.array([1.4, -0.3, -0.05, -0.05]),
        ],
    )
    def test_sum_transfers_theta_to_w(self, theta: np.ndarray) -> None:
        """Σθ = 1 should imply Σw ≈ 1 once truncation error is negligible."""
        assert sum_to_one_error(theta) < 1e-12
        w = theta_to_w(theta, truncation_lags=400)
        assert sum_to_one_error(w) < 1e-8

    @pytest.mark.parametrize(
        "w",
        [
            np.array([0.6, 0.3, 0.1]),
            np.array([0.5, 0.25, 0.15, 0.1]),
        ],
    )
    def test_sum_transfers_w_to_theta(self, w: np.ndarray) -> None:
        assert sum_to_one_error(w) < 1e-12
        theta = w_to_theta(w, truncation_lags=400)
        assert sum_to_one_error(theta) < 1e-8

    def test_non_unit_sum_preserved_up_to_series(self) -> None:
        """If Σθ ≠ 1, the reconstructed sum equals 1/Σθ (formal-series limit at L=1)."""
        theta = np.array([0.8, 0.1])  # sum = 0.9, not 1
        w = theta_to_w(theta, truncation_lags=400)
        assert w.sum() == pytest.approx(1.0 / 0.9, rel=1e-6)


# ──────────────────────────────────────────────────────────────────
# Diagnostics container
# ──────────────────────────────────────────────────────────────────


class TestConversionDiagnostics:
    def test_dataclass_reports_tail_and_sum(self) -> None:
        theta = np.array([1.5, -0.5])
        w = theta_to_w(theta, truncation_lags=300)
        diag = conversion_diagnostics(w)
        assert isinstance(diag, ConversionDiagnostics)
        assert diag.truncation_lags == 300
        assert diag.tail_abs_max < 1e-30  # 0.5^300 is astronomically small
        assert diag.sum == pytest.approx(1.0, abs=1e-10)
        assert diag.sum_error < 1e-10
        assert diag.truncation_warning == ""

    def test_diagnostics_warns_on_heavy_tail(self) -> None:
        """A very short truncation on a slowly-decaying inverse triggers a warning."""
        # θ_1/θ_0 ≈ 0.9 → w_k decays like 0.9^k; even at k=10 it's still ~0.35.
        theta = np.array([1.0, -0.9])
        w = theta_to_w(theta, truncation_lags=10)
        diag = conversion_diagnostics(w, tail_warn_threshold=1e-6)
        assert diag.truncation_warning != ""
        assert diag.tail_abs_max > 1e-6


# ──────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────


class TestValidation:
    def test_zero_leading_coefficient_raises(self) -> None:
        with pytest.raises(ValueError, match="a\\[0\\]"):
            theta_to_w(np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="a\\[0\\]"):
            w_to_theta(np.array([0.0, 1.0]))

    def test_empty_array_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            theta_to_w(np.array([]))

    def test_non_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            theta_to_w(np.array([[1.0, 0.5], [0.5, 1.0]]))

    def test_negative_truncation_raises(self) -> None:
        with pytest.raises(ValueError, match="truncation_lags"):
            theta_to_w(np.array([1.0, -0.2]), truncation_lags=-1)


# ──────────────────────────────────────────────────────────────────
# Default truncation is generous enough for typical Rudin cases
# ──────────────────────────────────────────────────────────────────


class TestDefaultTruncation:
    def test_default_recovers_typical_rudin_case(self) -> None:
        """Rudin's paper reports Q=1 with modest |θ_1|. Default truncation should give
        round-trip error smaller than 1e-9."""
        theta = np.array([1.35, -0.35])  # broadly in line with the paper's Q=1 estimate
        assert roundtrip_error(theta) < 1e-9
