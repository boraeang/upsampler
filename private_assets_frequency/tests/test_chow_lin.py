"""Tests for ``disaggregation.chow_lin`` — the central Stage 3 module.

The non-negotiable correctness test is :class:`TestRoundTrip`: monthly
returns compounded back to quarterly must match the original input
within 1e-10. The pipeline runner relies on this constraint as its
Stage 3 → Stage 4 inter-stage validation gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.core.protocols import (
    AggregationType,
    DisaggregationMethod,
)
from private_assets_frequency.disaggregation.chow_lin import (
    ChowLinDisaggregator,
    DisaggregationResult,
)
from private_assets_frequency.utils.returns import (
    aggregate_returns,
    realised_volatility,
)
from private_assets_frequency.utils.time_series import (
    MONTH_END_FREQ,
    QUARTER_END_FREQ,
)


# ──────────────────────────────────────────────────────────────────
# Helpers — build a synthetic disaggregation problem with known truth
# ──────────────────────────────────────────────────────────────────


def _make_problem(
    *,
    n_quarters: int = 40,
    rho: float = 0.3,
    beta: float = 1.2,
    alpha: float = 0.001,
    sigma_eps: float = 0.005,
    factor_vol_monthly: float = 0.04,
    seed: int = 0,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Generate (low-freq returns, high-freq factor returns, true high-freq, true HF df).

    The high-frequency series is generated as r_HF = β·F_HF + α + ε with
    AR(1) idiosyncratic shocks, then aggregated multiplicatively to quarterly
    to produce the low-frequency input the disaggregator will see.
    """
    rng = np.random.default_rng(seed)
    n_months = n_quarters * 3
    midx = pd.date_range("2000-01-31", periods=n_months, freq=MONTH_END_FREQ)
    qidx = pd.date_range("2000-03-31", periods=n_quarters, freq=QUARTER_END_FREQ)
    f = rng.normal(0.0, factor_vol_monthly, size=n_months)
    eps = np.zeros(n_months)
    eps[0] = rng.normal(0.0, sigma_eps)
    for t in range(1, n_months):
        eps[t] = rho * eps[t - 1] + rng.normal(0.0, sigma_eps * np.sqrt(1.0 - rho**2))
    r_hf = beta * f + alpha + eps
    r_lf = aggregate_returns(
        r_hf, ratio=3, method=AggregationType.MULTIPLICATIVE
    )
    return (
        pd.Series(r_lf, index=qidx, name="strategy"),
        pd.DataFrame({"factor": f}, index=midx),
        pd.Series(r_hf, index=midx, name="strategy"),
        pd.DataFrame({"factor": f}, index=midx),
    )


# ──────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────


class TestChowLinConstruction:
    def test_default(self):
        d = ChowLinDisaggregator()
        assert d.method is DisaggregationMethod.CHOW_LIN
        assert d.aggregation is AggregationType.MULTIPLICATIVE
        assert d.rho_grid_size == 41

    def test_string_args_accepted(self):
        d = ChowLinDisaggregator(method="fernandez", aggregation="additive")
        assert d.method is DisaggregationMethod.FERNANDEZ
        assert d.aggregation is AggregationType.ADDITIVE

    def test_invalid_rho_grid(self):
        with pytest.raises(ValueError):
            ChowLinDisaggregator(rho_lower=-1.0)
        with pytest.raises(ValueError):
            ChowLinDisaggregator(rho_grid_size=2)


# ──────────────────────────────────────────────────────────────────
# Round-trip aggregation — the spec's Stage 3 → 4 gate
# ──────────────────────────────────────────────────────────────────


class TestRoundTrip:
    """Monthly returns compounded back to quarterly must match input < 1e-10."""

    @pytest.mark.parametrize(
        "method",
        ["chow_lin", "fernandez", "litterman"],
    )
    def test_multiplicative_round_trip(self, method: str):
        lf, ind, _, _ = _make_problem(seed=0)
        d = ChowLinDisaggregator(method=method, aggregation="multiplicative")
        res = d.fit(lf, ind, ratio=3)
        # Multiplicative compounding back to quarterly
        agg_back = aggregate_returns(
            res.high_frequency, ratio=3, method=AggregationType.MULTIPLICATIVE
        )
        max_err = float(np.max(np.abs(agg_back.to_numpy() - lf.to_numpy())))
        assert max_err < 1e-10, (
            f"{method} multiplicative round-trip max |error| = {max_err:.3e} > 1e-10"
        )
        assert res.aggregation_error < 1e-10

    @pytest.mark.parametrize(
        "method",
        ["chow_lin", "fernandez", "litterman"],
    )
    def test_additive_round_trip(self, method: str):
        lf, ind, _, _ = _make_problem(seed=0)
        d = ChowLinDisaggregator(method=method, aggregation="additive")
        res = d.fit(lf, ind, ratio=3)
        agg_back = aggregate_returns(
            res.high_frequency, ratio=3, method=AggregationType.ADDITIVE
        )
        max_err = float(np.max(np.abs(agg_back.to_numpy() - lf.to_numpy())))
        assert max_err < 1e-10, (
            f"{method} additive round-trip max |error| = {max_err:.3e} > 1e-10"
        )

    def test_round_trip_holds_across_seeds(self):
        for seed in range(10):
            lf, ind, _, _ = _make_problem(seed=seed)
            res = ChowLinDisaggregator().fit(lf, ind, ratio=3)
            agg_back = aggregate_returns(
                res.high_frequency, ratio=3, method=AggregationType.MULTIPLICATIVE
            )
            err = float(np.max(np.abs(agg_back.to_numpy() - lf.to_numpy())))
            assert err < 1e-10, f"seed {seed}: round-trip error {err:.3e}"


# ──────────────────────────────────────────────────────────────────
# Recovery of latent path
# ──────────────────────────────────────────────────────────────────


class TestRecovery:
    def test_chow_lin_recovers_correlation_with_truth(self):
        """The disaggregated path should be highly correlated with the latent truth."""
        lf, ind, true_hf, _ = _make_problem(rho=0.4, seed=42)
        res = ChowLinDisaggregator(method="chow_lin").fit(lf, ind, ratio=3)
        corr = float(res.high_frequency.corr(true_hf))
        assert corr > 0.85

    def test_fernandez_recovers_correlation(self):
        lf, ind, true_hf, _ = _make_problem(rho=0.0, seed=1)
        res = ChowLinDisaggregator(method="fernandez").fit(lf, ind, ratio=3)
        corr = float(res.high_frequency.corr(true_hf))
        assert corr > 0.80

    def test_chow_lin_recovers_rho_sign(self):
        # Strongly autocorrelated residual → estimated ρ should be positive.
        lf, ind, _, _ = _make_problem(rho=0.7, seed=3)
        res = ChowLinDisaggregator(method="chow_lin").fit(lf, ind, ratio=3)
        assert res.rho is not None and res.rho > 0.0

    def test_disaggregated_vol_consistent_with_high_frequency(self):
        lf, ind, true_hf, _ = _make_problem(seed=0)
        res = ChowLinDisaggregator().fit(lf, ind, ratio=3)
        sigma_true = realised_volatility(true_hf)
        sigma_disag = realised_volatility(res.high_frequency)
        assert 0.7 * sigma_true <= sigma_disag <= 1.4 * sigma_true


# ──────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────


class TestValidation:
    def test_rejects_non_series_lf(self):
        ind = pd.DataFrame(
            {"f": np.zeros(120)},
            index=pd.date_range("2010-01-31", periods=120, freq=MONTH_END_FREQ),
        )
        with pytest.raises(TypeError):
            ChowLinDisaggregator().fit(np.zeros(40), ind, ratio=3)

    def test_indicators_wrong_length(self):
        lf, ind, _, _ = _make_problem(n_quarters=40, seed=0)
        ind_short = ind.iloc[:60]
        with pytest.raises(ValueError, match="rows"):
            ChowLinDisaggregator().fit(lf, ind_short, ratio=3)

    def test_invalid_ratio(self):
        lf, ind, _, _ = _make_problem(seed=0)
        with pytest.raises(ValueError):
            ChowLinDisaggregator().fit(lf, ind, ratio=1)

    def test_nan_in_lf(self):
        lf, ind, _, _ = _make_problem(seed=0)
        lf_bad = lf.copy()
        lf_bad.iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            ChowLinDisaggregator().fit(lf_bad, ind, ratio=3)

    def test_nan_in_indicators(self):
        lf, ind, _, _ = _make_problem(seed=0)
        ind_bad = ind.copy()
        ind_bad.iloc[10, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            ChowLinDisaggregator().fit(lf, ind_bad, ratio=3)


# ──────────────────────────────────────────────────────────────────
# Result fields
# ──────────────────────────────────────────────────────────────────


class TestResultFields:
    def test_result_fields_populated(self):
        lf, ind, _, _ = _make_problem(seed=0)
        res = ChowLinDisaggregator().fit(lf, ind, ratio=3)
        assert isinstance(res, DisaggregationResult)
        assert res.method == "chow_lin"
        assert res.aggregation == "multiplicative"
        assert "factor" in res.betas
        assert res.sigma2 > 0.0
        assert np.isfinite(res.log_likelihood)
        assert "rho" in res.diagnostics

    def test_fernandez_no_rho(self):
        lf, ind, _, _ = _make_problem(seed=0)
        res = ChowLinDisaggregator(method="fernandez").fit(lf, ind, ratio=3)
        assert res.rho is None
