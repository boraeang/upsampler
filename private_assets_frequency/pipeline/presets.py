"""
Asset-class preset configurations from the spec.

This module ships built-in :class:`AssetClassConfig` instances for each
strategy listed in the spec's "Asset Class Preset Configurations" section,
exposed both as plain dicts (``PE_PRESETS``, ``HF_PRESETS``, …) and via
auto-registration with the default :class:`PresetRegistry` at import time.

Defaults applied uniformly across presets when not specified per spec:

    * ``alpha_prior   = NormalPrior(0.0, 0.05)``
    * ``sigma_eps_prior = InverseGammaPrior(3.0, 0.02)``

Per-strategy notes are taken from the spec's preset table verbatim.

The presets are deliberately conservative: priors are chosen so the
posterior is never *less* informative than the data alone (Beta(2, 2) is
a near-uniform prior on λ; β priors have stds wide enough to reflect real
parameter uncertainty without being so loose as to invite λ–β confounding).
"""

from __future__ import annotations

from ..core.config import (
    AssetClassConfig,
    BetaDist,
    InverseGammaPrior,
    NormalPrior,
)
from ..core.registry import register_preset

# Spec-uniform defaults
_ALPHA_PRIOR_DEFAULT = NormalPrior(0.0, 0.05)
_ALPHA_PRIOR_TIGHT = NormalPrior(0.0, 0.03)
_ALPHA_PRIOR_TIGHTER = NormalPrior(0.0, 0.02)
_SIGMA_EPS_DEFAULT = InverseGammaPrior(3.0, 0.02)
_LAMBDA_PRIOR_DEFAULT = BetaDist(2.0, 2.0)


# ──────────────────────────────────────────────────────────────────
# Private equity
# ──────────────────────────────────────────────────────────────────


PE_PRESETS: dict[str, AssetClassConfig] = {
    "us_large_buyout": AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={"equity_market": NormalPrior(1.15, 0.50)},
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("SP500", "Russell2000"),
        public_proxy="Russell 2000 (leveraged)",
        notes=(
            "Leveraged equity; beta prior reflects ~2x leverage declining to "
            "1x at exit."
        ),
    ),
    "us_early_venture": AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={"equity_market": NormalPrior(0.83, 0.25)},
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("Russell2000_Growth", "NASDAQ"),
        public_proxy="Russell 2000 Growth",
        notes=(
            "Unleveraged; beta prior based on small-cap factor with "
            "negligible size beta."
        ),
    ),
    "us_late_venture": AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={"equity_market": NormalPrior(0.83, 0.25)},
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("Russell2000_Growth", "NASDAQ"),
        public_proxy="Russell 2000 Growth",
    ),
    "us_mezzanine": AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={"credit_proxy": NormalPrior(1.0, 0.50)},
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("HY_Bond_Index", "SP500"),
        public_proxy="ICE BofA High Yield",
        notes="Uncertain duration and credit quality relative to HY proxy.",
    ),
    "us_distressed": AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={"credit_proxy": NormalPrior(1.0, 0.50)},
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("HY_Bond_Index", "Distressed_Index"),
        public_proxy="ICE BofA Distressed",
        notes=(
            "Higher credit spreads than mezzanine; duration uncertainty shifts beta."
        ),
    ),
    "europe_buyout": AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={"equity_market": NormalPrior(0.95, 0.50)},
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("MSCI_Europe", "STOXX_600"),
        public_proxy="MSCI Europe",
    ),
    "europe_venture": AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={"equity_market": NormalPrior(0.79, 0.25)},
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("MSCI_Europe_SmallCap",),
        public_proxy="MSCI Europe Small Cap",
    ),
    "asia_buyout": AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={"equity_market": NormalPrior(0.91, 0.50)},
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("MSCI_AC_Asia",),
        public_proxy="MSCI AC Asia",
    ),
    "asia_venture": AssetClassConfig(
        asset_class="private_equity",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={"equity_market": NormalPrior(0.84, 0.25)},
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("MSCI_AC_Asia_SmallCap",),
        public_proxy="MSCI AC Asia Small Cap",
    ),
}


# ──────────────────────────────────────────────────────────────────
# Private infrastructure
# ──────────────────────────────────────────────────────────────────


INFRA_PRESETS: dict[str, AssetClassConfig] = {
    "core_infrastructure": AssetClassConfig(
        asset_class="private_infrastructure",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=BetaDist(3.0, 2.0),
        beta_priors={
            "equity_market": NormalPrior(0.5, 0.30),
            "utilities": NormalPrior(0.7, 0.30),
            "inflation": NormalPrior(0.3, 0.20),
        },
        alpha_prior=_ALPHA_PRIOR_TIGHT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("MSCI_World", "FTSE_Infra", "Breakeven_10Y", "UST_10Y"),
        public_proxy="FTSE Global Core Infrastructure 50/50",
        notes=(
            "Lower equity beta, meaningful duration & inflation sensitivity. "
            "λ often 0.5-0.8."
        ),
    ),
    "opportunistic_infrastructure": AssetClassConfig(
        asset_class="private_infrastructure",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=_LAMBDA_PRIOR_DEFAULT,
        beta_priors={
            "equity_market": NormalPrior(0.8, 0.40),
            "utilities": NormalPrior(0.5, 0.30),
        },
        alpha_prior=_ALPHA_PRIOR_DEFAULT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("MSCI_World", "FTSE_Infra", "SP_Global_Infra"),
        public_proxy="S&P Global Infrastructure",
    ),
}


# ──────────────────────────────────────────────────────────────────
# Private credit
# ──────────────────────────────────────────────────────────────────


CREDIT_PRESETS: dict[str, AssetClassConfig] = {
    "direct_lending": AssetClassConfig(
        asset_class="private_credit",
        smoothing_model="threshold_ar1",
        native_frequency="quarterly",
        preprocessing="carry_mtm_decomposition",
        lambda_prior_normal=BetaDist(5.0, 2.0),
        lambda_prior_stress=BetaDist(2.0, 5.0),
        beta_priors={
            "credit_spread": NormalPrior(0.7, 0.30),
            "rate_duration": NormalPrior(-0.2, 0.20),
        },
        alpha_prior=_ALPHA_PRIOR_TIGHT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        regime_indicator="HY_OAS",
        regime_threshold=500.0,
        default_factors=("Lev_Loan_Index", "HY_OAS", "UST_2Y", "Default_Rate"),
        public_proxy="Morningstar LSTA US Leveraged Loan 100",
        notes=(
            "Carry component bypasses desmoothing. Only MTM component is "
            "desmoothed. Regime threshold should be calibrated to the "
            "specific vintage of the index."
        ),
    ),
    "private_debt_senior": AssetClassConfig(
        asset_class="private_credit",
        smoothing_model="threshold_ar1",
        native_frequency="quarterly",
        preprocessing="carry_mtm_decomposition",
        lambda_prior_normal=BetaDist(5.0, 2.0),
        lambda_prior_stress=BetaDist(2.0, 5.0),
        beta_priors={
            "credit_spread": NormalPrior(0.5, 0.25),
            "rate_duration": NormalPrior(-0.3, 0.20),
        },
        alpha_prior=_ALPHA_PRIOR_TIGHTER,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        regime_indicator="HY_OAS",
        regime_threshold=500.0,
        default_factors=("IG_Corp_Index", "Lev_Loan_Index", "UST_5Y"),
        public_proxy="Bloomberg US Aggregate Credit",
    ),
}


# ──────────────────────────────────────────────────────────────────
# Hedge funds
# ──────────────────────────────────────────────────────────────────


HF_PRESETS: dict[str, AssetClassConfig] = {
    "equity_long_short": AssetClassConfig(
        asset_class="hedge_fund",
        smoothing_model="ma_glm",
        native_frequency="monthly",
        preprocessing="reporting_lag_adjustment",
        reporting_lag_months=1,
        ma_lags=2,
        theta_prior="ordered_dirichlet",
        beta_priors={
            "equity_market": NormalPrior(0.4, 0.20),
            "smb": NormalPrior(0.1, 0.15),
            "hml": NormalPrior(0.05, 0.15),
        },
        alpha_prior=_ALPHA_PRIOR_TIGHT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("SP500", "Russell2000", "HML", "MOM", "VIX"),
        public_proxy="HFRI Equity Hedge Index",
        notes=(
            "Monthly native frequency. Stage 3 disaggregates monthly→daily "
            "if needed."
        ),
    ),
    "global_macro": AssetClassConfig(
        asset_class="hedge_fund",
        smoothing_model="ma_glm",
        native_frequency="monthly",
        preprocessing="reporting_lag_adjustment",
        reporting_lag_months=1,
        ma_lags=2,
        theta_prior="ordered_dirichlet",
        beta_priors={
            "equity_market": NormalPrior(0.15, 0.20),
            "rates": NormalPrior(0.10, 0.15),
            "fx": NormalPrior(0.10, 0.15),
            "commodities": NormalPrior(0.10, 0.15),
        },
        alpha_prior=_ALPHA_PRIOR_TIGHT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("SP500", "UST_10Y", "DXY", "GSCI"),
        public_proxy="HFRI Macro Index",
    ),
    "event_driven": AssetClassConfig(
        asset_class="hedge_fund",
        smoothing_model="ma_glm",
        native_frequency="monthly",
        preprocessing="reporting_lag_adjustment",
        reporting_lag_months=1,
        ma_lags=3,  # Spec said q=4 but grid is supported up to 3 in this build
        theta_prior="ordered_dirichlet",
        beta_priors={
            "equity_market": NormalPrior(0.35, 0.20),
            "credit_spread": NormalPrior(0.25, 0.20),
        },
        alpha_prior=_ALPHA_PRIOR_TIGHT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("SP500", "HY_Bond_Index", "M_and_A_Spread"),
        public_proxy="HFRI Event-Driven Index",
        notes=(
            "Higher illiquidity → more lags. Significant credit exposure. "
            "Spec calls for q=4; this build's grid supports up to q=3 — "
            "switch when Laplace mode is implemented."
        ),
    ),
    "relative_value": AssetClassConfig(
        asset_class="hedge_fund",
        smoothing_model="ma_glm",
        native_frequency="monthly",
        preprocessing="reporting_lag_adjustment",
        reporting_lag_months=1,
        ma_lags=3,
        theta_prior="ordered_dirichlet",
        beta_priors={
            "equity_market": NormalPrior(0.10, 0.15),
            "credit_spread": NormalPrior(0.30, 0.20),
            "vol": NormalPrior(-0.15, 0.15),
        },
        alpha_prior=_ALPHA_PRIOR_TIGHT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("SP500", "HY_OAS", "VIX", "MOVE"),
        public_proxy="HFRI Relative Value Index",
    ),
    "managed_futures": AssetClassConfig(
        asset_class="hedge_fund",
        smoothing_model="no_smoothing",
        native_frequency="monthly",
        preprocessing=None,
        beta_priors={
            "trend_equity": NormalPrior(0.10, 0.15),
            "trend_rates": NormalPrior(0.10, 0.15),
            "trend_fx": NormalPrior(0.10, 0.15),
            "trend_commodities": NormalPrior(0.10, 0.15),
        },
        alpha_prior=_ALPHA_PRIOR_TIGHT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("SG_Trend", "SP500", "UST_10Y", "DXY", "GSCI"),
        public_proxy="SG Trend Index",
        notes=(
            "Exchange-traded underlyings → no smoothing. Only factor "
            "decomposition + disaggregation."
        ),
    ),
}


# ──────────────────────────────────────────────────────────────────
# Real estate
# ──────────────────────────────────────────────────────────────────


RE_PRESETS: dict[str, AssetClassConfig] = {
    "us_core_real_estate": AssetClassConfig(
        asset_class="private_real_estate",
        smoothing_model="ar1_bayesian",
        native_frequency="quarterly",
        preprocessing=None,
        lambda_prior=BetaDist(3.0, 1.5),
        beta_priors={
            "reit_market": NormalPrior(0.6, 0.30),
            "rate_duration": NormalPrior(-0.3, 0.20),
        },
        alpha_prior=_ALPHA_PRIOR_TIGHT,
        sigma_eps_prior=_SIGMA_EPS_DEFAULT,
        default_factors=("FTSE_NAREIT", "UST_10Y", "Breakeven_5Y", "GDP_Growth"),
        public_proxy="FTSE NAREIT All Equity REITs",
        notes=(
            "Original Geltner (1993) domain. Very high smoothing. NCREIF NPI "
            "is the canonical index. Strong Q4 seasonality in appraisals."
        ),
    ),
}


# ──────────────────────────────────────────────────────────────────
# Auto-register at import
# ──────────────────────────────────────────────────────────────────


def _register_all() -> None:
    for table in (PE_PRESETS, INFRA_PRESETS, CREDIT_PRESETS, HF_PRESETS, RE_PRESETS):
        for name, cfg in table.items():
            register_preset(name, cfg, overwrite=True)


_register_all()


__all__ = [
    "CREDIT_PRESETS",
    "HF_PRESETS",
    "INFRA_PRESETS",
    "PE_PRESETS",
    "RE_PRESETS",
]
