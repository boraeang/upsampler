"""Tests for ``disaggregation.aggregation``."""

from __future__ import annotations

import numpy as np
import pytest

from private_assets_frequency.core.protocols import AggregationType
from private_assets_frequency.disaggregation.aggregation import (
    aggregation_matrix,
    ar1_covariance,
    from_workspace,
    litterman_covariance,
    random_walk_covariance,
    to_workspace,
)


# ──────────────────────────────────────────────────────────────────
# Aggregation matrix (already tested in utils/returns; quick re-check)
# ──────────────────────────────────────────────────────────────────


class TestAggregationMatrix:
    def test_block_structure(self):
        C = aggregation_matrix(4, 3)
        assert C.shape == (4, 12)
        assert (C.sum(axis=1) == 3).all()
        assert (C.sum(axis=0) == 1).all()


# ──────────────────────────────────────────────────────────────────
# AR(1) covariance
# ──────────────────────────────────────────────────────────────────


class TestAR1Covariance:
    def test_diagonal_is_unconditional_variance(self):
        rho = 0.6
        V = ar1_covariance(5, rho)
        var = 1.0 / (1.0 - rho**2)
        np.testing.assert_allclose(np.diag(V), var)

    def test_off_diagonal_decays(self):
        V = ar1_covariance(6, 0.5)
        # V[0, k] should decay as 0.5^k
        for k in range(1, 6):
            np.testing.assert_allclose(
                V[0, k], V[0, 0] * 0.5**k, rtol=1e-12
            )

    def test_symmetric(self):
        V = ar1_covariance(8, -0.3)
        np.testing.assert_allclose(V, V.T)

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            ar1_covariance(5, 1.0)
        with pytest.raises(ValueError):
            ar1_covariance(5, -1.0)


# ──────────────────────────────────────────────────────────────────
# Random-walk covariance (Fernández)
# ──────────────────────────────────────────────────────────────────


class TestRandomWalkCovariance:
    def test_min_structure(self):
        V = random_walk_covariance(4)
        # V[i, j] = min(i+1, j+1)
        expected = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 2.0, 2.0, 2.0],
                [1.0, 2.0, 3.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        np.testing.assert_array_equal(V, expected)

    def test_symmetric_and_psd(self):
        V = random_walk_covariance(8)
        np.testing.assert_allclose(V, V.T)
        eigs = np.linalg.eigvalsh(V)
        assert (eigs > 0).all()


# ──────────────────────────────────────────────────────────────────
# Litterman covariance
# ──────────────────────────────────────────────────────────────────


class TestLittermanCovariance:
    def test_reduces_to_fernandez_at_rho_zero(self):
        # When ρ=0, Litterman ≡ Fernández (RW residuals) up to the unconditional
        # variance scaling of the AR(1) building block: V_AR1(ρ=0) = I·(1/(1-0²)) = I,
        # so L L' = (1)·random_walk_covariance.
        V_lit = litterman_covariance(5, 0.0)
        V_rw = random_walk_covariance(5)
        np.testing.assert_allclose(V_lit, V_rw)

    def test_symmetric_and_psd(self):
        V = litterman_covariance(6, 0.4)
        np.testing.assert_allclose(V, V.T)
        eigs = np.linalg.eigvalsh(V)
        assert (eigs > 0).all()


# ──────────────────────────────────────────────────────────────────
# Workspace converters
# ──────────────────────────────────────────────────────────────────


class TestWorkspace:
    def test_round_trip_multiplicative(self):
        x = np.array([0.01, 0.05, -0.02, 0.10])
        ws = to_workspace(x, AggregationType.MULTIPLICATIVE)
        np.testing.assert_allclose(ws, np.log1p(x))
        back = from_workspace(ws, AggregationType.MULTIPLICATIVE)
        np.testing.assert_allclose(back, x, atol=1e-15)

    def test_round_trip_additive(self):
        x = np.array([0.01, 0.05, -0.02, 0.10])
        ws = to_workspace(x, AggregationType.ADDITIVE)
        np.testing.assert_allclose(ws, x)
        back = from_workspace(ws, AggregationType.ADDITIVE)
        np.testing.assert_allclose(back, x, atol=1e-15)

    def test_multiplicative_rejects_le_neg_one(self):
        with pytest.raises(ValueError):
            to_workspace(np.array([-1.0, 0.0]), "multiplicative")

    def test_string_arg_accepted(self):
        x = np.array([0.01])
        ws = to_workspace(x, "multiplicative")
        np.testing.assert_allclose(ws, np.log1p(x))
