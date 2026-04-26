# Claude Code Prompt — Private Assets Return Frequency Upsampling Library (Multi-Asset-Class)

## Your Role & Decision Priorities

You are acting as both **implementer and senior quant reviewer**. You must not blindly follow this specification — if any modeling choice is statistically fragile, numerically unstable, or poorly identified given realistic sample sizes, call it out and propose a concrete mitigation before implementing.

**Decision hierarchy (in order of priority):**
1. **Statistical validity > architectural purity** — a simpler model that is well-identified beats an elegant one that isn't.
2. **Numerical stability > theoretical elegance** — if a formula blows up in edge cases, fix it even if the fix is less "clean."
3. **Robustness to small samples > asymptotic correctness** — PE has ~30 years of quarterly data. Asymptotic results are irrelevant.
4. **Interpretability > marginal accuracy gains** — every parameter must have an economic interpretation the user can sanity-check.

**Before writing any code:** produce a brief critical review (as code comments in the main `__init__.py` or as a `DESIGN_NOTES.md`) listing the top 3 identifiability/fragility risks in the pipeline and how your implementation mitigates each. This forces you to surface model risk early rather than discovering it during debugging.

---

## Context & Objective

You are building a **production-grade Python library** called `private_assets_frequency` for a quantitative hedge fund. The library converts low-frequency private/alternative asset index returns (quarterly or monthly) into higher-frequency synthetic return series (monthly and/or daily) that preserve the true risk characteristics — correcting for the well-documented smoothing biases before disaggregation.

**The library must be asset-class generic.** It supports:
- **Private Equity** (buyout, venture capital, mezzanine, distressed debt)
- **Private Infrastructure** (core, core-plus, value-add, opportunistic)
- **Private Credit** (direct lending, private debt, specialty finance)
- **Hedge Funds** (equity L/S, global macro, event-driven, relative value, managed futures, multi-strategy)
- **Private Real Estate** (core, value-add, opportunistic — NCREIF-style indices)

Each asset class has a **different smoothing mechanism**, a **different factor model**, and **different priors**. The architecture must be pluggable at each of these dimensions while keeping the temporal disaggregation and simulation stages fully generic.

**Why this matters:** Private/alternative asset indices report NAV-based or appraisal-based returns that dramatically understate true volatility (often by 50%+) and correlation with public markets. Any backtest, VaR model, or simulation using raw data will produce dangerously misleading results. This library solves that with a rigorous multi-stage pipeline adaptable to any illiquid or smoothed asset class.

---

## Architecture Overview

The pipeline has five sequential stages. Stage 0 is asset-class-specific preprocessing; Stages 1–4 follow a generic flow with pluggable components.

```
[Raw Low-Frequency Returns]
        │
        ▼
┌─────────────────────────────┐
│  Stage 0: Return            │
│  Preprocessing              │
│  (asset-class-specific:     │
│   e.g., carry/MTM decomp   │
│   for credit)               │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Desmoothing       │
│  (pluggable model:          │
│   AR(1), MA(q), Threshold,  │
│   or None)                  │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 2: Factor            │
│  Decomposition              │
│  (systematic + idiosyncratic│
│   at native frequency)      │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 3: Temporal          │
│  Disaggregation             │
│  (Chow-Lin / Fernández /   │
│   Litterman — generic)      │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 4: Daily Extension   │
│  (Kalman smoother or        │
│   simulation — generic)     │
└─────────────────────────────┘
```

---

## Stage 0: Return Preprocessing

### Purpose
Some asset classes require decomposing or transforming the raw return before desmoothing. This stage is **optional** and **asset-class-specific**.

### Private Credit: Carry / Mark-to-Market Decomposition

Private credit returns have two components with fundamentally different dynamics:

1. **Carry component:** Coupon income accruing smoothly over time. This is NOT smoothed in the Geltner sense — it genuinely accrues at a steady rate. Desmoothing it would introduce spurious volatility.
2. **Mark-to-market (MTM) component:** Changes in the fair value of the credit spread and default risk. THIS is the smoothed component — loans may be carried at par for quarters, then suddenly written down.

**Implementation:**
```python
def decompose_credit_return(
    total_return: pd.Series,
    yield_series: pd.Series,          # Running yield or coupon rate
    frequency: str = 'quarterly'
) -> tuple[pd.Series, pd.Series]:
    """
    Decomposes total return into carry and MTM components.
    
    carry_t ≈ yield_t * Δt  (accrued coupon for the period)
    mtm_t = total_t - carry_t
    
    Only the MTM component is passed to Stage 1 for desmoothing.
    After desmoothing and disaggregation, the carry component is 
    added back at the target frequency (distributed linearly within 
    each quarter).
    """
```

### All Other Asset Classes
For PE, infrastructure, and real estate, Stage 0 is a pass-through — the full return is passed to Stage 1.

### Hedge Funds: Reporting Lag Adjustment
Hedge fund index returns (HFRI, CS) incorporate a 1–2 month reporting lag on top of the illiquidity-driven smoothing. If the user provides a `reporting_lag` parameter (in months), Stage 0 shifts the return series backward by that lag before desmoothing. This separates mechanical reporting delay from economic smoothing.

```python
def adjust_reporting_lag(
    returns: pd.Series,
    lag_months: int = 1
) -> pd.Series:
    """Shift returns backward by lag_months to remove mechanical reporting delay."""
```

---

## Stage 1: Desmoothing (Pluggable Smoothing Model)

### Abstraction Layer

The key architectural element: the smoothing model is a **protocol** (structural subtyping) that any desmoothing approach must implement.

```python
from typing import Protocol, NamedTuple
import numpy as np
import pandas as pd

class DesmoothedResult(NamedTuple):
    true_returns: pd.Series          # Desmoothed return series
    smoothing_params: dict           # Estimated parameters (λ, θ vector, etc.)
    posterior_summary: dict          # Posterior means, stds, credible intervals
    diagnostics: dict                # Model-specific diagnostics

class SmoothingModel(Protocol):
    """Protocol for pluggable smoothing models."""
    
    def fit(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
        priors: dict,
    ) -> DesmoothedResult:
        """Jointly estimate smoothing parameters and factor betas."""
        ...
    
    def log_likelihood(
        self,
        smoothing_params: np.ndarray,
        observed_returns: np.ndarray,
        factor_returns: np.ndarray,
    ) -> float:
        """Conditional log-likelihood given smoothing parameters."""
        ...
    
    def log_prior(
        self,
        smoothing_params: np.ndarray,
        prior_config: dict,
    ) -> float:
        """Log-prior density over smoothing parameters."""
        ...
    
    def desmooth(
        self,
        observed_returns: np.ndarray,
        smoothing_params: np.ndarray,
    ) -> np.ndarray:
        """Given known smoothing params, recover true returns."""
        ...
```

### Model 1: AR(1) Bayesian Desmoothing (PE, Infrastructure, Real Estate)

This is the primary model, following the MSCI Private Equity Factor Model methodology.

**Smoothing process:**
```
P_t = P_{t-1} + (1 - λ)(V_t - P_{t-1})
⟹ s_t = (1 - λ) · r_t + λ · s_{t-1}
⟹ r_t = (s_t - λ · s_{t-1}) / (1 - λ)
```

**Implementation requirements:**

1. **Use rolling annual returns, not raw quarterly returns.** Quarterly smoothing is seasonal (Q4 often has more mark-to-market activity than Q1–Q3). Construct four overlapping annual return series, each starting from a different quarter. This makes desmoothing robust to seasonal variation in λ.

2. **Single-step estimation.** Jointly estimate λ and the factor betas β in a single regression. The model is:
   ```
   r_PE,annual = β · F_annual + α + ε
   ```
   where `r_PE,annual = (s_t - λ · s_{t-4}) / (1 - λ)` (annual desmoothed return) and `F_annual` are annual factor returns aggregated from monthly.

3. **Bayesian estimation via numerical integration.** The parameter space per strategy is: (λ, β_1, ..., β_K, α, σ_ε).
   - **Prior on λ:** Beta distribution, Beta(2, 2), mild prior centered at 0.5 with support on [0, 1].
   - **Prior on β:** Normal, strategy-specific (see Asset Class Presets section below).
   - **Prior on α:** N(0, 0.05²) — weakly informative, centered on zero alpha.
   - **Prior on σ_ε:** Inverse-Gamma(3, 0.02).
   - **Posterior computation:** Grid over λ (50 points on [0.01, 0.95]). For each λ, the conditional posterior of (β, α, σ_ε) is analytically tractable via Normal-Inverse-Gamma conjugacy. Integrate over the λ grid using numerical quadrature (scipy.integrate.trapezoid) to get marginal posteriors and posterior expectations.

4. **Induced priors (cross-sectional shrinkage).** Use the empirical distribution of estimated parameters across peer strategies as an additional prior. Iterate: estimate all strategies independently → use cross-sectional distribution as induced priors → re-estimate → repeat until convergence.

5. **Thin-factor correction.** With small samples, the sample covariance of factor returns overestimates the true covariance. Apply Bayesian correction: `F_hat = E[F | data]`.

6. **Identifiability diagnostics (critical).** The joint identification of λ and β is weak — multiple (λ, β) combinations can explain the same data. Implement the following checks:
   - If the 90% credible interval on λ spans more than 0.4, warn that smoothing is poorly identified and the output is prior-driven.
   - If the posterior on λ is near-identical to the prior (KL divergence < 0.1 nats), warn that the data is not informative.
   - Check whether β changes sign or magnitude by more than 50% across different λ values on the grid. If so, λ and β are confounded — warn the user.
   - Report the effective sample size (accounting for autocorrelation in the rolling annual returns).

7. **Uncertainty propagation mode.** In addition to the default point-estimate pipeline, support an `uncertainty_mode='full'` option that:
   - Samples N=100 draws of (λ, β, σ_ε) from the joint posterior.
   - Runs the full Stage 2→3→4 pipeline for each draw.
   - Returns percentile bands (5th, 25th, 50th, 75th, 95th) on the monthly/daily return series.
   - This is computationally cheap (100× the base pipeline) and prevents false precision in downstream risk estimates.

### Model 2: MA(q) Desmoothing — Getmansky-Lo-Makarov (Hedge Funds)

**Smoothing process:**
```
s_t = θ_0 · r_t + θ_1 · r_{t-1} + ... + θ_q · r_{t-q}
```
where θ_0 + θ_1 + ... + θ_q = 1 (weights sum to one).

The true return is recovered by inverting the MA filter. In matrix form, if S = Θ · R where Θ is a banded Toeplitz-like matrix, then R = Θ⁻¹ · S.

**Implementation requirements:**

1. **Estimate the MA coefficients θ.** Use maximum likelihood: the observed returns s_t follow a constrained MA(q) process. The variance of s_t and its autocovariances identify the θ weights:
   ```
   Var(s_t) = σ²_r · Σ θ²_j
   Cov(s_t, s_{t-k}) = σ²_r · Σ_{j=0}^{q-k} θ_j · θ_{j+k}   for k ≤ q
   ```
   Solve this system of equations for θ given the sample autocovariances, subject to: θ_j ≥ 0, Σ θ_j = 1.

2. **Bayesian extension.** Put priors on the θ vector:
   - θ_0 should be the largest weight (the contemporaneous return gets the most weight). Prior: θ_0 ~ Beta(5, 2) rescaled.
   - θ should be decreasing: enforce via an ordered Dirichlet prior or a stick-breaking construction.
   - The number of lags q can be selected via BIC, or fixed per strategy (typically q=2 for liquid strategies like equity L/S, q=4–5 for illiquid strategies like distressed).

3. **Joint estimation with factors.** Similar to the AR(1) case, jointly estimate θ and factor betas β. The desmoothed return feeds into the factor regression. For numerical integration, the parameter space is higher-dimensional (q+1 parameters for θ). For q ≤ 3, grid integration over a simplex is feasible. For q > 3, use Laplace approximation (find the MAP via scipy.optimize.minimize, then approximate the posterior as Gaussian around the mode using the Hessian).

4. **Inversion stability.** The MA inversion amplifies noise. Regularize by:
   - Clipping θ_0 ≥ 0.2 (contemporaneous weight is at least 20%)
   - Tapering the higher-lag coefficients with an exponential decay prior
   - Monitoring the condition number of the Θ matrix

### Model 3: Threshold AR(1) (Private Credit)

**Smoothing process:**
```
s_t = (1 - λ(regime_t)) · r_t + λ(regime_t) · s_{t-1}
```
where λ takes different values depending on a regime indicator:
- **Normal regime** (no credit events): λ_high ∈ [0.5, 0.9] — loans carried at par, minimal marking
- **Stress regime** (credit events, writedowns): λ_low ∈ [0.0, 0.3] — forced marking to reality

**Implementation requirements:**

1. **Regime identification.** Two options:
   - **Exogenous:** Use an observable indicator to define regimes — e.g., credit spread level (HY OAS > threshold = stress), default rate, or VIX.
   - **Endogenous:** Estimate regimes from the data using a Markov-switching model. The regime is a latent state with transition probabilities.

2. **For the initial implementation, use the exogenous approach** — it's simpler, more robust, and more transparent. Let the user supply a binary regime indicator or specify a threshold on a credit spread series.

3. **Bayesian estimation.** Two smoothing parameters (λ_high, λ_low) plus factor betas. Grid integration over the 2D (λ_high, λ_low) space (e.g., 30×30 grid), with the same conjugate conditional posterior for β as in the AR(1) case.

4. **Only apply to the MTM component** of credit returns (the carry component from Stage 0 bypasses desmoothing entirely).

### Model 4: NoSmoothing (Pass-through)

For data that is already desmoothed, or for asset classes where smoothing is negligible. Simply passes the observed returns to Stage 2 unchanged. Useful as a baseline comparator.

---

## Stage 2: Factor Decomposition

This stage is fully generic across asset classes. The factor matrix differs, but the methodology is identical.

### Implementation Requirements

1. Decompose each return series (post-desmoothing) into:
   ```
   r_t = Σ_k β_k · F_k,t + α + ε_t
   ```
   Using OLS or WLS (with optional exponential decay weighting for more recent observations).

2. Extract the residual series ε_t.

3. Characterize the residual:
   - Ljung-Box test for autocorrelation
   - ARCH-LM test for heteroskedasticity
   - Jarque-Bera test for normality
   - Fit parametric distribution: normal, Student-t, skewed-t (Hansen, 1994)
   - Store all properties for use in Stage 3/4

4. Compute cross-strategy residual covariance matrix.

5. For hedge funds (monthly native frequency): the factor decomposition operates at monthly frequency directly. For quarterly asset classes: operates at quarterly frequency with quarterly-aggregated factor returns.

---

## Stage 3: Temporal Disaggregation (Generic)

This stage is **fully asset-class agnostic**. It takes any low-frequency series plus high-frequency indicators and produces the target-frequency output.

### Theoretical Foundation

**Chow-Lin (1971):** Given a low-frequency series y_LF and high-frequency indicator series X_HF, find high-frequency values y_HF such that:
- Temporal aggregation constraint: high-frequency values within each low-frequency period aggregate to the observed value
- y_HF is related to X_HF via a regression with autocorrelated residuals

The GLS solution is:
```
y_HF = X_HF · β + V · C' · (C · V · C')⁻¹ · (y_LF - C · X_HF · β)
```

### Implementation Requirements

1. **Three disaggregation methods** — implement all:
   - **Chow-Lin:** AR(1) residual structure. Estimate ρ via GLS.
   - **Fernández (1981):** Random-walk residual (ρ = 1). No ρ estimation.
   - **Litterman (1983):** AR(1) on first-differenced residuals.

2. **Aggregation constraint options:**
   - **Additive:** Σ r_high = r_low (linear approximation)
   - **Multiplicative:** Π (1 + r_high) = (1 + r_low) (exact for returns)
   Default to multiplicative. Work in log-returns for linear algebra, then exponentiate.

3. **Flexible frequency conversion:**
   - Quarterly → Monthly (3:1 ratio) — standard case for PE, infrastructure, real estate, credit
   - Monthly → Weekly (~4.3:1) — possible for hedge funds
   - Monthly → Daily (~21:1) — for hedge funds going to daily
   - Quarterly → Daily — two-step: quarterly → monthly → daily

4. **Multi-strategy consistency:** When disaggregating multiple strategies simultaneously, preserve cross-sectional correlation structure. Use multivariate Chow-Lin or marginal disaggregation with copula adjustment.

5. **Indicator variable selection:** The high-frequency factor returns from the asset-class-specific factor model serve as indicators. The factor loadings from Stage 2 provide coefficient estimates.

### For Hedge Funds (Monthly → Daily disaggregation)
Since hedge fund data is typically monthly, Stage 3 disaggregates from monthly to daily directly. The indicators are daily factor returns. The aggregation ratio is ~21:1 (business days per month), which makes Chow-Lin's matrix inversion larger but still tractable for typical time series lengths.

---

## Stage 4: Daily Extension (Generic)

### Implementation Requirements

1. **Systematic component:**
   ```
   r_systematic,d = Σ_k β_k · F_k,d
   ```
   Deterministic given daily factor returns and Stage 2 betas.

2. **Idiosyncratic component — two modes:**

   **Mode A: Kalman Smoother (for point estimation / backtesting)**
   - State: daily idiosyncratic return ε_d
   - Transition: ε_d = φ · ε_{d-1} + η_d (AR(1) with daily persistence)
   - Observation: monthly (or quarterly) idiosyncratic return = sum of daily returns within period
   - Run Kalman smoother to produce E[ε_d | all observations]
   - Constraint: daily values sum to the higher-frequency idiosyncratic return from Stage 3

   **Mode B: Simulation (for Monte Carlo / VaR)**
   - Draw daily idiosyncratic shocks from fitted distribution (Stage 2), scaled to daily
   - Apply compounding constraint within each period
   - Generate N simulation paths
   - Multi-strategy: draw from joint distribution using estimated cross-strategy residual correlation

3. **Business day calendar:** NYSE calendar as default, configurable for other markets.

---

## Asset Class Preset Configurations

The library ships with sensible defaults for each asset class. Users can override any parameter.

```python
# ──────────────────────────────────────────────
# PRIVATE EQUITY
# ──────────────────────────────────────────────

PE_PRESETS = {
    'us_large_buyout': AssetClassConfig(
        asset_class='private_equity',
        smoothing_model='ar1_bayesian',
        preprocessing=None,                         # No preprocessing needed
        lambda_prior=BetaDist(2, 2),                 # λ ~ Beta(2,2), centered ~0.5
        beta_priors={
            'equity_market': NormalPrior(1.15, 0.50),  # Leveraged equity
        },
        alpha_prior=NormalPrior(0.0, 0.05),
        default_factors=['SP500', 'Russell2000'],
        public_proxy='Russell 2000 (leveraged)',
        native_frequency='quarterly',
        notes='Leveraged equity; beta prior reflects ~2x leverage declining to 1x at exit.',
    ),
    'us_early_venture': AssetClassConfig(
        asset_class='private_equity',
        smoothing_model='ar1_bayesian',
        preprocessing=None,
        lambda_prior=BetaDist(2, 2),
        beta_priors={
            'equity_market': NormalPrior(0.83, 0.25),  # Unleveraged small-cap-like, tighter prior
        },
        alpha_prior=NormalPrior(0.0, 0.05),
        default_factors=['Russell2000_Growth', 'NASDAQ'],
        public_proxy='Russell 2000 Growth',
        native_frequency='quarterly',
        notes='Unleveraged; beta prior based on small-cap factor with negligible size beta.',
    ),
    'us_late_venture': AssetClassConfig(
        asset_class='private_equity',
        smoothing_model='ar1_bayesian',
        preprocessing=None,
        lambda_prior=BetaDist(2, 2),
        beta_priors={
            'equity_market': NormalPrior(0.83, 0.25),
        },
        alpha_prior=NormalPrior(0.0, 0.05),
        default_factors=['Russell2000_Growth', 'NASDAQ'],
        public_proxy='Russell 2000 Growth',
        native_frequency='quarterly',
    ),
    'us_mezzanine': AssetClassConfig(
        asset_class='private_equity',
        smoothing_model='ar1_bayesian',
        preprocessing=None,
        lambda_prior=BetaDist(2, 2),
        beta_priors={
            'credit_proxy': NormalPrior(1.0, 0.50),    # Uncertain leverage/duration
        },
        alpha_prior=NormalPrior(0.0, 0.05),
        default_factors=['HY_Bond_Index', 'SP500'],
        public_proxy='ICE BofA High Yield',
        native_frequency='quarterly',
        notes='Uncertain duration and credit quality relative to HY proxy.',
    ),
    'us_distressed': AssetClassConfig(
        asset_class='private_equity',
        smoothing_model='ar1_bayesian',
        preprocessing=None,
        lambda_prior=BetaDist(2, 2),
        beta_priors={
            'credit_proxy': NormalPrior(1.0, 0.50),
        },
        alpha_prior=NormalPrior(0.0, 0.05),
        default_factors=['HY_Bond_Index', 'Distressed_Index'],
        public_proxy='ICE BofA Distressed',
        native_frequency='quarterly',
        notes='Higher credit spreads than mezzanine; duration uncertainty shifts beta.',
    ),
    # Replicate for Europe and Asia with region-specific priors:
    #   Europe Buyout β ~ N(0.95, 0.50)
    #   Asia Buyout β ~ N(0.91, 0.50)
    #   Europe Venture β ~ N(0.79, 0.25)
    #   Asia Venture β ~ N(0.84, 0.25)
}

# ──────────────────────────────────────────────
# PRIVATE INFRASTRUCTURE
# ──────────────────────────────────────────────

INFRA_PRESETS = {
    'core_infrastructure': AssetClassConfig(
        asset_class='private_infrastructure',
        smoothing_model='ar1_bayesian',
        preprocessing=None,
        lambda_prior=BetaDist(3, 2),                 # Slightly higher λ expected (more DCF-based)
        beta_priors={
            'equity_market': NormalPrior(0.5, 0.30),   # Lower beta to broad equity
            'utilities': NormalPrior(0.7, 0.30),
            'inflation': NormalPrior(0.3, 0.20),       # Inflation sensitivity
        },
        alpha_prior=NormalPrior(0.0, 0.03),            # Tighter alpha — more stable returns
        default_factors=['MSCI_World', 'FTSE_Infra', 'Breakeven_10Y', 'UST_10Y'],
        public_proxy='FTSE Global Core Infrastructure 50/50',
        native_frequency='quarterly',
        notes='Lower equity beta, meaningful duration & inflation sensitivity. λ often 0.5-0.8.',
    ),
    'opportunistic_infrastructure': AssetClassConfig(
        asset_class='private_infrastructure',
        smoothing_model='ar1_bayesian',
        preprocessing=None,
        lambda_prior=BetaDist(2, 2),
        beta_priors={
            'equity_market': NormalPrior(0.8, 0.40),   # Higher beta, more equity-like
            'utilities': NormalPrior(0.5, 0.30),
        },
        alpha_prior=NormalPrior(0.0, 0.05),
        default_factors=['MSCI_World', 'FTSE_Infra', 'SP_Global_Infra'],
        public_proxy='S&P Global Infrastructure',
        native_frequency='quarterly',
    ),
}

# ──────────────────────────────────────────────
# PRIVATE CREDIT
# ──────────────────────────────────────────────

CREDIT_PRESETS = {
    'direct_lending': AssetClassConfig(
        asset_class='private_credit',
        smoothing_model='threshold_ar1',
        preprocessing='carry_mtm_decomposition',      # Stage 0: split carry from MTM
        lambda_prior_normal=BetaDist(5, 2),            # λ_high ~ 0.6-0.9 in normal regime
        lambda_prior_stress=BetaDist(2, 5),            # λ_low ~ 0.1-0.3 in stress regime
        beta_priors={
            'credit_spread': NormalPrior(0.7, 0.30),   # Leveraged loan beta
            'rate_duration': NormalPrior(-0.2, 0.20),  # Short duration, negative rate sensitivity
        },
        alpha_prior=NormalPrior(0.0, 0.03),
        default_factors=['Lev_Loan_Index', 'HY_OAS', 'UST_2Y', 'Default_Rate'],
        public_proxy='Morningstar LSTA US Leveraged Loan 100',
        regime_indicator='HY_OAS',                     # Exogenous regime: HY OAS level
        regime_threshold=500,                          # bps; above = stress regime
        native_frequency='quarterly',
        notes='Carry component bypasses desmoothing. Only MTM component is desmoothed. '
              'Regime threshold should be calibrated to the specific vintage of the index.',
    ),
    'private_debt_senior': AssetClassConfig(
        asset_class='private_credit',
        smoothing_model='threshold_ar1',
        preprocessing='carry_mtm_decomposition',
        lambda_prior_normal=BetaDist(5, 2),
        lambda_prior_stress=BetaDist(2, 5),
        beta_priors={
            'credit_spread': NormalPrior(0.5, 0.25),
            'rate_duration': NormalPrior(-0.3, 0.20),
        },
        alpha_prior=NormalPrior(0.0, 0.02),
        default_factors=['IG_Corp_Index', 'Lev_Loan_Index', 'UST_5Y'],
        public_proxy='Bloomberg US Aggregate Credit',
        regime_indicator='HY_OAS',
        regime_threshold=500,
        native_frequency='quarterly',
    ),
}

# ──────────────────────────────────────────────
# HEDGE FUNDS
# ──────────────────────────────────────────────

HF_PRESETS = {
    'equity_long_short': AssetClassConfig(
        asset_class='hedge_fund',
        smoothing_model='ma_glm',                     # Getmansky-Lo-Makarov MA(q)
        preprocessing='reporting_lag_adjustment',
        reporting_lag_months=1,
        ma_lags=2,                                     # q=2 for relatively liquid strategy
        theta_prior='ordered_dirichlet',               # θ decreasing, θ_0 largest
        beta_priors={
            'equity_market': NormalPrior(0.4, 0.20),   # Hedged, net long
            'smb': NormalPrior(0.1, 0.15),             # Small-cap tilt
            'hml': NormalPrior(0.05, 0.15),            # Value tilt
        },
        alpha_prior=NormalPrior(0.0, 0.03),
        default_factors=['SP500', 'Russell2000', 'HML', 'MOM', 'VIX'],
        public_proxy='HFRI Equity Hedge Index',
        native_frequency='monthly',                    # HF data is monthly, not quarterly
        notes='Monthly native frequency. Stage 3 disaggregates monthly→daily if needed.',
    ),
    'global_macro': AssetClassConfig(
        asset_class='hedge_fund',
        smoothing_model='ma_glm',
        preprocessing='reporting_lag_adjustment',
        reporting_lag_months=1,
        ma_lags=2,
        theta_prior='ordered_dirichlet',
        beta_priors={
            'equity_market': NormalPrior(0.15, 0.20),
            'rates': NormalPrior(0.10, 0.15),
            'fx': NormalPrior(0.10, 0.15),
            'commodities': NormalPrior(0.10, 0.15),
        },
        alpha_prior=NormalPrior(0.0, 0.03),
        default_factors=['SP500', 'UST_10Y', 'DXY', 'GSCI'],
        public_proxy='HFRI Macro Index',
        native_frequency='monthly',
    ),
    'event_driven': AssetClassConfig(
        asset_class='hedge_fund',
        smoothing_model='ma_glm',
        preprocessing='reporting_lag_adjustment',
        reporting_lag_months=1,
        ma_lags=4,                                     # More illiquid holdings
        theta_prior='ordered_dirichlet',
        beta_priors={
            'equity_market': NormalPrior(0.35, 0.20),
            'credit_spread': NormalPrior(0.25, 0.20),
        },
        alpha_prior=NormalPrior(0.0, 0.03),
        default_factors=['SP500', 'HY_Bond_Index', 'M_and_A_Spread'],
        public_proxy='HFRI Event-Driven Index',
        native_frequency='monthly',
        notes='Higher illiquidity → more lags. Significant credit exposure.',
    ),
    'relative_value': AssetClassConfig(
        asset_class='hedge_fund',
        smoothing_model='ma_glm',
        preprocessing='reporting_lag_adjustment',
        reporting_lag_months=1,
        ma_lags=3,
        theta_prior='ordered_dirichlet',
        beta_priors={
            'equity_market': NormalPrior(0.10, 0.15),
            'credit_spread': NormalPrior(0.30, 0.20),
            'vol': NormalPrior(-0.15, 0.15),           # Short vol exposure
        },
        alpha_prior=NormalPrior(0.0, 0.03),
        default_factors=['SP500', 'HY_OAS', 'VIX', 'MOVE'],
        public_proxy='HFRI Relative Value Index',
        native_frequency='monthly',
    ),
    'managed_futures': AssetClassConfig(
        asset_class='hedge_fund',
        smoothing_model='no_smoothing',                # CTA/managed futures are exchange-traded
        preprocessing=None,
        beta_priors={
            'trend_equity': NormalPrior(0.10, 0.15),
            'trend_rates': NormalPrior(0.10, 0.15),
            'trend_fx': NormalPrior(0.10, 0.15),
            'trend_commodities': NormalPrior(0.10, 0.15),
        },
        alpha_prior=NormalPrior(0.0, 0.03),
        default_factors=['SG_Trend', 'SP500', 'UST_10Y', 'DXY', 'GSCI'],
        public_proxy='SG Trend Index',
        native_frequency='monthly',
        notes='Exchange-traded underlyings → no smoothing. Only factor decomposition + disaggregation.',
    ),
}

# ──────────────────────────────────────────────
# PRIVATE REAL ESTATE
# ──────────────────────────────────────────────

RE_PRESETS = {
    'us_core_real_estate': AssetClassConfig(
        asset_class='private_real_estate',
        smoothing_model='ar1_bayesian',
        preprocessing=None,
        lambda_prior=BetaDist(3, 1.5),                 # Higher λ expected (0.6-0.85)
        beta_priors={
            'reit_market': NormalPrior(0.6, 0.30),     # Lower beta to listed REITs
            'rate_duration': NormalPrior(-0.3, 0.20),  # Negative rate sensitivity
        },
        alpha_prior=NormalPrior(0.0, 0.03),
        default_factors=['FTSE_NAREIT', 'UST_10Y', 'Breakeven_5Y', 'GDP_Growth'],
        public_proxy='FTSE NAREIT All Equity REITs',
        native_frequency='quarterly',
        notes='Original Geltner (1993) domain. Very high smoothing. '
              'NCREIF NPI is the canonical index. Strong Q4 seasonality in appraisals.',
    ),
}
```

---

## Library Structure

```
private_assets_frequency/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── protocols.py              # SmoothingModel protocol, result types
│   ├── config.py                 # AssetClassConfig, NormalPrior, BetaDist, etc.
│   └── registry.py              # Preset registry: get_preset('us_large_buyout')
├── preprocessing/
│   ├── __init__.py
│   ├── credit_decomposition.py  # Carry / MTM decomposition (Stage 0)
│   ├── reporting_lag.py         # Hedge fund reporting lag adjustment
│   └── passthrough.py           # No-op preprocessor
├── desmoothing/
│   ├── __init__.py
│   ├── ar1_bayesian.py          # AR(1) Bayesian desmoothing (PE, Infra, RE)
│   ├── ma_glm.py               # MA(q) Getmansky-Lo-Makarov (Hedge Funds)
│   ├── threshold_ar1.py         # Threshold/regime-switching AR(1) (Credit)
│   ├── geltner_classic.py       # Classic Geltner (non-Bayesian, baseline)
│   ├── no_smoothing.py          # Pass-through (managed futures, pre-desmoothed)
│   └── induced_priors.py        # Cross-sectional shrinkage across peer strategies
├── decomposition/
│   ├── __init__.py
│   ├── factor_model.py          # Factor regression (Stage 2)
│   └── residual_analysis.py     # Residual diagnostics & distribution fitting
├── disaggregation/
│   ├── __init__.py
│   ├── chow_lin.py              # Chow-Lin, Fernández, Litterman (Stage 3)
│   ├── aggregation.py           # Temporal aggregation matrices & constraints
│   └── multivariate.py          # Multi-strategy joint disaggregation
├── daily/
│   ├── __init__.py
│   ├── kalman.py                # Kalman smoother (Stage 4A)
│   ├── simulation.py            # Monte Carlo daily paths (Stage 4B)
│   └── calendar.py              # Business day calendar utilities
├── pipeline/
│   ├── __init__.py
│   ├── runner.py                # End-to-end pipeline orchestration
│   ├── presets.py               # All asset class preset configurations
│   └── fallback.py             # FallbackPolicy enum & graceful degradation logic
├── validation/
│   ├── __init__.py
│   ├── diagnostics.py           # Statistical tests & diagnostics
│   ├── consistency.py           # Aggregation consistency checks
│   ├── sanity_anchors.py        # Public market cross-checks (vol, corr, drawdown, beta)
│   ├── stage_validation.py      # Inter-stage validation (output checks between stages)
│   └── visualization.py        # Diagnostic plots (matplotlib)
├── utils/
│   ├── __init__.py
│   ├── returns.py               # Return compounding, log transforms
│   └── time_series.py           # Rolling windows, alignment, resampling
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures: synthetic data generators per asset class
    ├── test_preprocessing.py    # Stage 0 tests
    ├── test_ar1_desmoothing.py  # AR(1) desmoothing tests
    ├── test_ma_desmoothing.py   # MA(q) desmoothing tests  
    ├── test_threshold_ar1.py    # Threshold AR(1) tests
    ├── test_decomposition.py    # Factor decomposition tests
    ├── test_disaggregation.py   # Chow-Lin / Fernández / Litterman tests
    ├── test_daily.py            # Kalman + simulation tests
    ├── test_pipeline_pe.py      # End-to-end PE pipeline
    ├── test_pipeline_hf.py      # End-to-end hedge fund pipeline
    ├── test_pipeline_credit.py  # End-to-end credit pipeline
    ├── test_pipeline_infra.py   # End-to-end infrastructure pipeline
    ├── test_pipeline_re.py      # End-to-end real estate pipeline
    └── test_consistency.py      # Round-trip aggregation & moment preservation
```

---

## API Design

### High-Level API

```python
from private_assets_frequency import FrequencyPipeline
from private_assets_frequency.pipeline.presets import PE_PRESETS, HF_PRESETS

# ── Example 1: Private Equity (quarterly → monthly → daily) ──

pipeline = FrequencyPipeline(
    returns=quarterly_pe_df,                    # DatetimeIndex, columns = strategy names
    factor_returns_monthly=monthly_factors_df,
    factor_returns_daily=daily_factors_df,       # Optional
    configs={
        'us_buyout': PE_PRESETS['us_large_buyout'],
        'us_vc': PE_PRESETS['us_early_venture'],
    },
    disaggregation_method='chow_lin',
    aggregation_type='multiplicative',
    fallback_policy='warn',                      # 'strict' | 'warn' | 'auto'
    uncertainty_mode='point',                    # 'point' | 'full' (propagates posterior samples)
)

result = pipeline.run()
monthly = result.monthly_returns
daily = result.daily_returns                     # None if daily factors not provided
warnings = result.warnings                       # List of identifiability/sanity warnings
sanity = result.sanity_anchors                   # Sanity anchor check results

# With uncertainty_mode='full':
# result.monthly_returns_bands  → Dict[str, DataFrame] with keys '5%', '25%', '50%', '75%', '95%'

# ── Example 2: Hedge Fund (monthly → daily) ──

hf_pipeline = FrequencyPipeline(
    returns=monthly_hf_df,
    factor_returns_daily=daily_factors_df,
    configs={
        'eq_ls': HF_PRESETS['equity_long_short'],
        'macro': HF_PRESETS['global_macro'],
    },
    disaggregation_method='chow_lin',
    aggregation_type='multiplicative',
)

hf_result = hf_pipeline.run()
# For HF: no monthly disaggregation needed (already monthly)
# Stage 3 disaggregates monthly → daily directly

# ── Example 3: Private Credit (quarterly → monthly) ──

from private_assets_frequency.pipeline.presets import CREDIT_PRESETS

credit_pipeline = FrequencyPipeline(
    returns=quarterly_credit_df,
    factor_returns_monthly=monthly_factors_df,
    yield_series=quarterly_yield_df,             # Required for carry/MTM decomposition
    regime_indicator=hy_oas_monthly,             # Required for threshold AR(1)
    configs={
        'direct_lending': CREDIT_PRESETS['direct_lending'],
    },
    disaggregation_method='fernandez',           # Random-walk residual often better for credit
    aggregation_type='multiplicative',
)

credit_result = credit_pipeline.run()

# ── Example 4: Custom configuration ──

from private_assets_frequency.core.config import AssetClassConfig, NormalPrior, BetaDist

custom = AssetClassConfig(
    asset_class='private_equity',
    smoothing_model='ar1_bayesian',
    preprocessing=None,
    lambda_prior=BetaDist(3, 2),
    beta_priors={
        'equity_market': NormalPrior(0.9, 0.40),
        'credit_spread': NormalPrior(0.3, 0.20),
    },
    alpha_prior=NormalPrior(0.02, 0.03),         # Expect some alpha
    default_factors=['MSCI_EM', 'EMBI'],
    public_proxy='MSCI Emerging Markets',
    native_frequency='quarterly',
)

# ── Simulation API ──

sim = pipeline.simulate(
    n_paths=10000,
    horizon_months=120,
    frequency='daily',
    seed=42,
    correlation_method='empirical',              # or 'shrinkage'
)

# sim.paths: (n_paths, n_days, n_strategies) array
# sim.statistics: VaR, CVaR, drawdown distributions, etc.

# ── Diagnostics ──

result.plot_desmoothing_diagnostics()            # Prior vs posterior, λ distribution
result.plot_aggregation_consistency()            # Monthly→quarterly round-trip
result.plot_factor_decomposition()               # Systematic vs idiosyncratic
result.plot_volatility_term_structure()          # Vol ratio at different horizons
result.summary()                                 # Text report of all parameters
```

---

## Testing Strategy

### Synthetic Data Tests (Per Smoothing Model)

Each smoothing model must have a synthetic data round-trip test:

**AR(1) synthetic test:**
1. Generate true monthly returns from a known factor model: r_t = β·F_t + α + ε_t
2. Aggregate to quarterly
3. Apply AR(1) smoothing with known λ
4. Feed smoothed quarterly data through Stage 0→1→2→3
5. Assert: recovered monthly returns close to true (within statistical tolerance)
6. Assert: recovered λ, β within 2σ of posterior of true values
7. Assert: recovered vol, correlation, beta close to true

**MA(q) synthetic test:**
1. Generate true monthly returns from a known factor model
2. Apply MA(q) smoothing with known θ weights
3. Feed through pipeline with `smoothing_model='ma_glm'`
4. Assert: recovered θ close to true (within tolerance)
5. Assert: desmoothed returns recover true vol and correlation

**Threshold AR(1) synthetic test:**
1. Generate true quarterly credit returns: carry + MTM with known regime-dependent λ
2. Apply threshold smoothing with known λ_high, λ_low, and regime indicator
3. Feed through pipeline with carry/MTM decomposition
4. Assert: carry component preserved exactly
5. Assert: MTM λ estimates close to true per regime

### Consistency Tests (Generic)
- **Round-trip aggregation:** High-frequency output compounded to low-frequency must exactly match input (within floating-point tolerance ~1e-10)
- **Moment preservation:** Annualized vol of disaggregated series matches desmoothed native-frequency vol (within sampling error)
- **Cross-strategy correlations:** High-frequency cross-correlations consistent with low-frequency post-desmoothing

### Edge Cases
- Very high smoothing (λ > 0.85, θ_0 < 0.3) — common for VC, real estate
- Near-zero smoothing (λ ≈ 0, θ_0 ≈ 1) — should recover input
- Short time series (< 10 years)
- Extreme returns (GFC, COVID)
- Missing data (some quarters with NaN)
- Mixed native frequencies in the same pipeline run (e.g., quarterly PE + monthly HF)

### Sanity Anchors (Public Market Cross-Checks)

Every pipeline run must include automated sanity checks comparing outputs against public market benchmarks. These are not pass/fail — they produce warnings when outputs look economically implausible:

1. **Volatility anchor:** Desmoothed annualized vol should be within [0.5×, 3.0×] the vol of the public proxy. If desmoothed US Buyout vol is 50% when levered Russell 2000 is at 20%, something is wrong.
2. **Correlation anchor:** Correlation between desmoothed PE and the public proxy should be in [0.4, 0.95]. Below 0.4 suggests factor model failure; above 0.95 suggests the output is just the public proxy.
3. **Drawdown anchor:** Maximum drawdown of the monthly series during known crises (GFC: 2008Q3–2009Q1, COVID: 2020Q1) should be at least 50% of the public proxy's drawdown in the same period. Smaller drawdowns suggest under-desmoothing.
4. **Beta anchor:** Estimated beta to the public proxy should be consistent with economic leverage. Unleveraged PE (VC) beta > 1.5 to unlevered equity, or leveraged buyout beta < 0.5, should trigger warnings.

Implement as `validation/sanity_anchors.py` and integrate into the pipeline result object.

### Fallback Mechanisms

If any stage fails or produces diagnostically suspect output, the pipeline should degrade gracefully rather than crash or silently produce garbage:

1. **Bayesian AR(1) fails to identify λ** (flat posterior) → fall back to classic Geltner with a conservative λ = 0.5 and warn.
2. **MA(q) inversion has condition number > 100** → reduce q by 1 and re-estimate. If q=1 still fails → fall back to AR(1).
3. **Factor regression has R² < 0.05** → warn that systematic component is negligible and the disaggregation will be dominated by the residual distribution.
4. **Chow-Lin residual ρ outside [−0.99, 0.99]** → clip and warn; consider switching to Fernández (ρ=1).
5. **Induced prior iteration does not converge in 10 iterations** → stop and use the last iterate with a warning.

Implement as a `FallbackPolicy` enum (`strict` = raise on any warning, `warn` = continue with warnings, `auto` = apply fallbacks silently) configurable per pipeline run.

---

## Dependencies

**Core (required):**
- numpy >= 1.24
- scipy >= 1.10 (optimize, integrate, linalg, stats, signal)
- pandas >= 2.0
- statsmodels >= 0.14

**Optional:**
- matplotlib >= 3.7 (for validation/visualization module)
- pandas_market_calendars (for daily business day calendar)

**Do NOT use:** PyMC, Stan, TensorFlow Probability, or any MCMC library. All Bayesian computation must use scipy numerical integration, quadrature, or Laplace approximation.

---

## Documentation Requirements

- **Module-level docstrings** explaining the econometric methodology for each smoothing model
- **Function-level docstrings** in NumPy style with Parameters, Returns, Notes, References
- **References section** citing:
  - Geltner (1993) — AR(1) desmoothing for real estate
  - Getmansky, Lo, Makarov (2004) — MA(q) smoothing in hedge fund returns
  - Chow & Lin (1971) — temporal disaggregation
  - Fernández (1981) — random-walk variant
  - Litterman (1983) — AR(1) on differences variant
  - Shepard (2014) — MSCI Bayesian desmoothing methodology
  - Mariano & Murasawa (2003) — state-space mixed frequency models
  - Hansen (1994) — skewed-t distribution
  - MSCI (2025) — "The MSCI Private Equity Factor Model"
- **A README.md** with installation, quickstart per asset class, methodology overview, and caveats
- **Type hints** throughout (Python 3.10+ style using `X | None` syntax)

---

## Critical Implementation Notes

1. **Numerical stability:** Desmoothing `r = (s - λ*s_lag) / (1-λ)` blows up as λ → 1. Clip λ posterior to [0.01, 0.95]. MA inversion condition number grows with q — monitor and warn. Log-space for return compounding.

2. **Temporal alignment:** Quarterly PE returns are end-of-quarter. Monthly factor returns are end-of-month. Monthly HF returns may use different month-end conventions. Use explicit `pandas.tseries.offsets` and document the convention. Never rely on implicit date alignment.

3. **Return compounding:** Always use `(1 + r_Q) = Π(1 + r_m)` for aggregation checks. The additive approximation can error by 50+ bps for PE-magnitude returns.

4. **Native frequency detection:** The pipeline should auto-detect whether input data is quarterly or monthly from the DatetimeIndex frequency. If ambiguous, require the user to specify.

5. **Reproducibility:** All random operations accept a `seed` or `rng` (numpy.random.Generator) parameter. Default to deterministic behavior.

6. **Performance:** AR(1) grid integration (50 points × strategies): fast. MA(q) grid on simplex for q≤3: feasible. For q>3: use Laplace approximation (MAP + Hessian). Chow-Lin matrix inversion for 30 years monthly (~360×360): fine. Daily Kalman for 30 years (~7500 states): use scipy's efficient banded implementations.

7. **Inter-stage validation.** Each stage must validate its outputs before passing downstream. Specifically:
   - Stage 1 → Stage 2: desmoothed returns must have higher volatility than smoothed (if not, λ estimate is suspect).
   - Stage 2 → Stage 3: residuals must be approximately uncorrelated with factors (if not, the factor model is misspecified or factors are missing).
   - Stage 3 → Stage 4: monthly returns must compound to quarterly within tolerance 1e-10.
   - Each validation emits a `StageValidationResult` with pass/warn/fail status.

8. **Epistemic honesty in outputs.** The pipeline generates *model-implied* high-frequency returns, not *recovered true* returns. The README, docstrings, and result objects must use language like "synthetic monthly returns consistent with observed quarterly data under [model] assumptions" rather than "true monthly returns." The daily path in Mode A (Kalman smoother) is a conditional expectation, not a realization — document this clearly.

---

## Build Order

Implement in this order, testing each before proceeding:

0. `DESIGN_NOTES.md` — Critical review: top 3 identifiability/fragility risks and mitigations (must be written before any code)
1. `core/protocols.py` + `core/config.py` — type definitions, protocol, config classes
2. `utils/returns.py` — return math (compounding, log, aggregation)
3. `preprocessing/` — all three preprocessors (passthrough, credit decomp, HF lag)
4. `desmoothing/geltner_classic.py` — non-Bayesian baseline
5. `desmoothing/ar1_bayesian.py` — full Bayesian AR(1) with grid integration
6. `desmoothing/ma_glm.py` — MA(q) Getmansky-Lo-Makarov
7. `desmoothing/threshold_ar1.py` — regime-switching AR(1) for credit
8. `desmoothing/no_smoothing.py` — pass-through
9. `desmoothing/induced_priors.py` — cross-sectional shrinkage
10. `decomposition/factor_model.py` + `residual_analysis.py`
11. `disaggregation/chow_lin.py` + `aggregation.py`
12. `disaggregation/multivariate.py`
13. `pipeline/runner.py` + `pipeline/presets.py` — wire everything together
14. `daily/kalman.py`
15. `daily/simulation.py`
16. `validation/` — all diagnostics, consistency, visualization
17. `tests/` — comprehensive test suite per asset class
18. `README.md` — documentation

At each step, write and run the corresponding unit tests before moving on.
