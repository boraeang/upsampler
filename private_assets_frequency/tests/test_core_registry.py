"""Tests for ``private_assets_frequency.core.registry``."""

from __future__ import annotations

import pytest

from private_assets_frequency.core.config import (
    AssetClassConfig,
    BetaDist,
    InverseGammaPrior,
    NormalPrior,
)
from private_assets_frequency.core.registry import (
    PresetRegistry,
    default_registry,
    filter_presets,
    get_preset,
    list_presets,
    register_preset,
    unregister_preset,
)


def _make_pe(asset_class="private_equity") -> AssetClassConfig:
    return AssetClassConfig(
        asset_class=asset_class,
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        lambda_prior=BetaDist(2.0, 2.0),
        beta_priors={"equity_market": NormalPrior(1.0, 0.5)},
        alpha_prior=NormalPrior(0.0, 0.05),
        sigma_eps_prior=InverseGammaPrior(3.0, 0.02),
    )


def _make_hf() -> AssetClassConfig:
    return AssetClassConfig(
        asset_class="hedge_fund",
        smoothing_model="ma_glm",
        native_frequency="monthly",
        beta_priors={"equity_market": NormalPrior(0.4, 0.2)},
        alpha_prior=NormalPrior(0.0, 0.03),
        ma_lags=2,
        theta_prior="ordered_dirichlet",
    )


# ──────────────────────────────────────────────────────────────────
# PresetRegistry instance behaviour
# ──────────────────────────────────────────────────────────────────


class TestPresetRegistryInstance:
    def test_register_and_get(self):
        reg = PresetRegistry()
        cfg = _make_pe()
        reg.register("foo", cfg)
        assert reg.get("foo") is cfg
        assert "foo" in reg
        assert len(reg) == 1
        assert reg.list() == ["foo"]

    def test_register_duplicate_rejected_by_default(self):
        reg = PresetRegistry()
        reg.register("foo", _make_pe())
        with pytest.raises(KeyError, match="already registered"):
            reg.register("foo", _make_pe())

    def test_register_overwrite_allowed_with_flag(self):
        reg = PresetRegistry()
        first = _make_pe()
        second = _make_pe(asset_class="private_equity")
        reg.register("foo", first)
        reg.register("foo", second, overwrite=True)
        assert reg.get("foo") is second

    def test_unregister_removes(self):
        reg = PresetRegistry()
        reg.register("foo", _make_pe())
        reg.unregister("foo")
        assert "foo" not in reg

    def test_clear_empties(self):
        reg = PresetRegistry()
        reg.register("a", _make_pe())
        reg.register("b", _make_hf())
        reg.clear()
        assert len(reg) == 0

    def test_get_unknown_empty_registry(self):
        reg = PresetRegistry()
        with pytest.raises(KeyError, match="empty"):
            reg.get("missing")

    def test_get_unknown_with_suggestion(self):
        reg = PresetRegistry()
        reg.register("us_large_buyout", _make_pe())
        reg.register("us_early_venture", _make_pe())
        with pytest.raises(KeyError, match="us_large_buyout"):
            reg.get("us_large_buyot")  # typo

    def test_invalid_name(self):
        reg = PresetRegistry()
        with pytest.raises(ValueError):
            reg.register("", _make_pe())

    def test_invalid_config_type(self):
        reg = PresetRegistry()
        with pytest.raises(TypeError):
            reg.register("foo", {"not": "a config"})  # type: ignore[arg-type]

    def test_filter_by_asset_class(self):
        reg = PresetRegistry()
        reg.register("pe1", _make_pe())
        reg.register("pe2", _make_pe())
        reg.register("hf1", _make_hf())
        only_pe = reg.filter(asset_class="private_equity")
        assert set(only_pe) == {"pe1", "pe2"}

    def test_filter_by_smoothing_model(self):
        reg = PresetRegistry()
        reg.register("pe1", _make_pe())
        reg.register("hf1", _make_hf())
        only_ar1 = reg.filter(smoothing_model="ar1_bayesian")
        assert list(only_ar1) == ["pe1"]

    def test_filter_by_frequency(self):
        reg = PresetRegistry()
        reg.register("pe1", _make_pe())
        reg.register("hf1", _make_hf())
        monthly = reg.filter(native_frequency="monthly")
        assert list(monthly) == ["hf1"]

    def test_filter_combined(self):
        reg = PresetRegistry()
        reg.register("pe1", _make_pe())
        reg.register("hf1", _make_hf())
        result = reg.filter(asset_class="private_equity", native_frequency="monthly")
        assert result == {}

    def test_iter_sorted(self):
        reg = PresetRegistry()
        reg.register("zeta", _make_pe())
        reg.register("alpha", _make_hf())
        assert list(reg) == ["alpha", "zeta"]

    def test_items_sorted(self):
        reg = PresetRegistry()
        reg.register("b", _make_pe())
        reg.register("a", _make_hf())
        keys = [k for k, _ in reg.items()]
        assert keys == ["a", "b"]


# ──────────────────────────────────────────────────────────────────
# Module-level default registry
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_default_registry():
    """Snapshot, clear, and restore the default registry around each test."""
    snapshot = dict(default_registry().items())
    default_registry().clear()
    try:
        yield default_registry()
    finally:
        default_registry().clear()
        for name, cfg in snapshot.items():
            default_registry().register(name, cfg)


class TestDefaultRegistry:
    def test_register_get_round_trip(self, clean_default_registry):
        cfg = _make_pe()
        register_preset("toy", cfg)
        assert get_preset("toy") is cfg

    def test_list_returns_sorted_names(self, clean_default_registry):
        register_preset("zeta", _make_pe())
        register_preset("alpha", _make_hf())
        assert list_presets() == ["alpha", "zeta"]

    def test_unregister_removes(self, clean_default_registry):
        register_preset("toy", _make_pe())
        unregister_preset("toy")
        with pytest.raises(KeyError):
            get_preset("toy")

    def test_filter_presets(self, clean_default_registry):
        register_preset("pe1", _make_pe())
        register_preset("hf1", _make_hf())
        only_quarterly = filter_presets(native_frequency="quarterly")
        assert list(only_quarterly) == ["pe1"]
