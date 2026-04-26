"""Tests for ``uncertainty_mode='full'`` posterior bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.pipeline.presets import PE_PRESETS
from private_assets_frequency.pipeline.runner import (
    FrequencyPipeline,
    UncertaintyBands,
)
from private_assets_frequency.tests.conftest import AR1SyntheticDataset


def _build_pe_inputs(ds: AR1SyntheticDataset):
    returns = pd.DataFrame({"us_buyout": ds.observed_quarterly})
    monthly = pd.DataFrame({"equity_market": ds.factor_monthly["equity_market"]})
    return returns, monthly


# ──────────────────────────────────────────────────────────────────
# Bands populated for AR(1) Bayesian
# ──────────────────────────────────────────────────────────────────


class TestUncertaintyBandsAR1:
    def test_full_mode_returns_bands(self, ar1_pe_dataset: AR1SyntheticDataset):
        returns, factors_m = _build_pe_inputs(ar1_pe_dataset)
        result = FrequencyPipeline(
            returns=returns,
            factor_returns_monthly=factors_m,
            configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
            uncertainty_mode="full",
            uncertainty_n_samples=20,
            uncertainty_seed=42,
        ).run()
        assert result.uncertainty_bands is not None
        bands = result.uncertainty_bands
        assert isinstance(bands, UncertaintyBands)
        assert bands.n_samples == 20
        assert set(bands.percentiles) == {5, 25, 50, 75, 95}

    def test_band_ordering(self, ar1_pe_dataset: AR1SyntheticDataset):
        returns, factors_m = _build_pe_inputs(ar1_pe_dataset)
        result = FrequencyPipeline(
            returns=returns,
            factor_returns_monthly=factors_m,
            configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
            uncertainty_mode="full",
            uncertainty_n_samples=30,
            uncertainty_seed=0,
        ).run()
        bands = result.uncertainty_bands
        m05 = bands.monthly[5]["us_buyout"].to_numpy()
        m25 = bands.monthly[25]["us_buyout"].to_numpy()
        m50 = bands.monthly[50]["us_buyout"].to_numpy()
        m75 = bands.monthly[75]["us_buyout"].to_numpy()
        m95 = bands.monthly[95]["us_buyout"].to_numpy()
        # Pointwise monotonicity
        assert (m05 <= m25 + 1e-12).all()
        assert (m25 <= m50 + 1e-12).all()
        assert (m50 <= m75 + 1e-12).all()
        assert (m75 <= m95 + 1e-12).all()

    def test_band_columns_match_strategies(
        self, ar1_pe_dataset: AR1SyntheticDataset
    ):
        returns, factors_m = _build_pe_inputs(ar1_pe_dataset)
        result = FrequencyPipeline(
            returns=returns,
            factor_returns_monthly=factors_m,
            configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
            uncertainty_mode="full",
            uncertainty_n_samples=10,
            uncertainty_seed=1,
        ).run()
        bands = result.uncertainty_bands
        for pct in bands.percentiles:
            df = bands.monthly[pct]
            assert list(df.columns) == ["us_buyout"]
            assert len(df) == len(factors_m)

    def test_point_estimate_unchanged_when_full_mode(self, ar1_pe_dataset):
        """The point-estimate output is the same regardless of uncertainty_mode."""
        returns, factors_m = _build_pe_inputs(ar1_pe_dataset)
        common = dict(
            returns=returns,
            factor_returns_monthly=factors_m,
            configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
        )
        point = FrequencyPipeline(
            **common, uncertainty_mode="point"
        ).run()
        full = FrequencyPipeline(
            **common,
            uncertainty_mode="full",
            uncertainty_n_samples=5,
            uncertainty_seed=0,
        ).run()
        np.testing.assert_allclose(
            point.monthly_returns.to_numpy(),
            full.monthly_returns.to_numpy(),
            rtol=1e-12,
        )

    def test_seed_reproducibility(self, ar1_pe_dataset):
        returns, factors_m = _build_pe_inputs(ar1_pe_dataset)
        common = dict(
            returns=returns,
            factor_returns_monthly=factors_m,
            configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
            uncertainty_mode="full",
            uncertainty_n_samples=10,
        )
        a = FrequencyPipeline(**common, uncertainty_seed=7).run().uncertainty_bands
        b = FrequencyPipeline(**common, uncertainty_seed=7).run().uncertainty_bands
        np.testing.assert_array_equal(
            a.monthly[50]["us_buyout"].to_numpy(),
            b.monthly[50]["us_buyout"].to_numpy(),
        )


# ──────────────────────────────────────────────────────────────────
# Models without sample_posterior fall back gracefully
# ──────────────────────────────────────────────────────────────────


class TestUncertaintyBandsFallback:
    def test_threshold_ar1_skipped_with_warning(self, threshold_credit_dataset):
        from private_assets_frequency.tests.test_pipeline_credit import (
            _CREDIT_TEST_CFG,
            _build_credit_inputs,
        )

        returns, monthly_F, yields, regime = _build_credit_inputs(
            threshold_credit_dataset
        )
        result = FrequencyPipeline(
            returns=returns,
            factor_returns_monthly=monthly_F,
            configs={"direct_lending": _CREDIT_TEST_CFG},
            yield_series=yields,
            regime_indicator=regime,
            disaggregation_method="fernandez",
            uncertainty_mode="full",
            uncertainty_n_samples=5,
        ).run()
        # Threshold AR(1) doesn't expose sample_posterior — bands should be
        # None (the only strategy was skipped) and the warnings list should
        # contain the skip message.
        assert result.uncertainty_bands is None
        assert any("does not yet support" in w for w in result.warnings)
