# Claude Code Prompt — Tutorial Notebook for private_assets_frequency

## Task

Create a Jupyter notebook (`docs/tutorial.ipynb`) that serves as the primary tutorial and usage guide for the `private_assets_frequency` library. The notebook must be **fully self-contained and runnable** — it generates its own synthetic data so that anyone can execute it without access to proprietary indices.

Read the library specification at `docs/claude_code_prompt_private_assets_frequency_v2.md` for the full API and architecture reference. The notebook's code must be consistent with the API defined there.

## Audience

Quantitative analysts and portfolio managers who will use the library in production. They:
- Know Python and pandas well
- Understand the basics of factor models and time series
- May NOT know the details of desmoothing, Chow-Lin, or Kalman filtering
- Want to see **what the library does**, **how to call it**, and **how to interpret the outputs**
- Will copy-paste cells into their own workflows

## Structure & Pedagogical Approach

The notebook follows a **progressive disclosure** pattern:
1. Start with the simplest possible end-to-end example (5 lines of code, one strategy, default settings)
2. Peel back each stage, explaining what happens and showing intermediate outputs
3. Build up to multi-asset-class, multi-strategy, simulation, and advanced configuration
4. End with production tips and common pitfalls

Every section follows the rhythm: **motivation → code → output → interpretation → gotcha**.

Use matplotlib for all plots. Style them with a dark background (`#22272E`) and teal/gold accents to match the library's identity. Set this as the default style at the top of the notebook.

---

## Notebook Outline

### Cell 1 — Title & Introduction (Markdown)

```markdown
# private_assets_frequency — Tutorial & Usage Guide

This notebook demonstrates how to use the `private_assets_frequency` library to convert 
low-frequency private asset returns into higher-frequency synthetic series that preserve 
true risk characteristics.

**What this library does:**
1. **Desmooths** quarterly returns to recover true volatility and correlations
2. **Decomposes** returns into systematic (factor-driven) and idiosyncratic components
3. **Disaggregates** quarterly data to monthly (or monthly to daily)
4. **Simulates** high-frequency paths for Monte Carlo / VaR

**What you'll learn in this notebook:**
- Quickstart: 5 lines to go from quarterly PE returns to monthly
- How each pipeline stage works, with visual diagnostics
- How to configure the library for PE, hedge funds, credit, infrastructure, and real estate
- How to run simulations and interpret uncertainty bands
- Production tips and common pitfalls

All examples use synthetic data generated in this notebook — no proprietary data required.
```

### Cell 2 — Setup & Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Library imports
from private_assets_frequency import FrequencyPipeline
from private_assets_frequency.pipeline.presets import (
    PE_PRESETS, HF_PRESETS, CREDIT_PRESETS, INFRA_PRESETS, RE_PRESETS
)
from private_assets_frequency.core.config import AssetClassConfig, NormalPrior, BetaDist

# Plot styling — dark theme matching library identity
plt.rcParams.update({
    'figure.facecolor': '#22272E',
    'axes.facecolor': '#22272E',
    'axes.edgecolor': '#555555',
    'axes.labelcolor': '#E8E8E8',
    'text.color': '#E8E8E8',
    'xtick.color': '#AAAAAA',
    'ytick.color': '#AAAAAA',
    'grid.color': '#333333',
    'figure.figsize': (12, 5),
    'font.size': 11,
})
TEAL = '#00B4D8'
GOLD = '#D4A843'
RED = '#E07A5F'
GREEN = '#81B29A'

print(f"private_assets_frequency loaded successfully")
```

### Cell 3 — Synthetic Data Generator (Markdown + Code)

**Markdown:**
```markdown
## Generating Synthetic Data

Since private asset indices are proprietary, we generate synthetic data with **known ground truth**. 
This lets us verify the library recovers the correct parameters.

The data generating process:
1. Create monthly factor returns (equity market, credit spread, rates)
2. Generate **true** monthly PE returns using a known factor model (β=1.15, α=0.5%/qtr, σ_ε=4%/qtr)
3. Aggregate true monthly returns to quarterly
4. Apply AR(1) smoothing with known λ=0.55 to simulate appraisal-based reporting
5. The smoothed quarterly series is what we'd observe in practice — the library must recover the rest
```

**Code:**
```python
def generate_synthetic_pe_data(
    n_years=25,
    true_beta=1.15,
    true_alpha_quarterly=0.005,
    true_sigma_eps_quarterly=0.04,
    true_lambda=0.55,
    seed=42,
):
    """
    Generate synthetic PE data with known ground truth for validation.
    
    Returns quarterly smoothed returns (what we observe),
    true monthly returns (what we want to recover),
    and monthly factor returns (high-frequency indicators).
    """
    rng = np.random.default_rng(seed)
    n_months = n_years * 12
    dates_monthly = pd.date_range('2000-01-31', periods=n_months, freq='ME')
    
    # Monthly factor returns (equity market proxy)
    equity_monthly = rng.normal(0.008, 0.045, n_months)  # ~9.6% ann return, ~15.6% ann vol
    credit_monthly = rng.normal(0.003, 0.015, n_months)  # credit spread factor
    
    factor_df = pd.DataFrame({
        'equity_market': equity_monthly,
        'credit_spread': credit_monthly,
    }, index=dates_monthly)
    
    # True monthly PE returns (unobservable in practice)
    monthly_alpha = true_alpha_quarterly / 3
    monthly_sigma = true_sigma_eps_quarterly / np.sqrt(3)
    idiosyncratic = rng.normal(0, monthly_sigma, n_months)
    
    true_monthly = true_beta * equity_monthly + 0.3 * credit_monthly + monthly_alpha + idiosyncratic
    true_monthly_series = pd.Series(true_monthly, index=dates_monthly, name='us_buyout')
    
    # Aggregate to quarterly (compounding)
    true_quarterly_index = (1 + true_monthly_series).groupby(
        true_monthly_series.index.to_period('Q')
    ).prod() - 1
    true_quarterly_index.index = true_quarterly_index.index.to_timestamp('Q')
    
    # Apply AR(1) smoothing to simulate appraisal-based reporting
    smoothed_quarterly = pd.Series(index=true_quarterly_index.index, dtype=float)
    smoothed_quarterly.iloc[0] = true_quarterly_index.iloc[0]
    for i in range(1, len(true_quarterly_index)):
        smoothed_quarterly.iloc[i] = (
            (1 - true_lambda) * true_quarterly_index.iloc[i] 
            + true_lambda * smoothed_quarterly.iloc[i-1]
        )
    
    return {
        'smoothed_quarterly': smoothed_quarterly.to_frame('us_buyout'),
        'true_monthly': true_monthly_series.to_frame('us_buyout'),
        'true_quarterly': true_quarterly_index.to_frame('us_buyout'),
        'factor_returns_monthly': factor_df,
        'true_params': {
            'beta': true_beta, 'alpha_q': true_alpha_quarterly,
            'sigma_eps_q': true_sigma_eps_quarterly, 'lambda': true_lambda,
        },
    }

data = generate_synthetic_pe_data()
print(f"Generated {len(data['smoothed_quarterly'])} quarters of synthetic PE data")
print(f"True parameters: λ={data['true_params']['lambda']}, β={data['true_params']['beta']}")
print(f"\nSmoothed quarterly vol:  {data['smoothed_quarterly']['us_buyout'].std() * 2:.1%} (annualized)")
print(f"True quarterly vol:      {data['true_quarterly']['us_buyout'].std() * 2:.1%} (annualized)")
```

### Cell 4 — Visualize the Smoothing Problem (Code)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: cumulative returns — smoothed vs true
cum_smooth = (1 + data['smoothed_quarterly']['us_buyout']).cumprod()
cum_true = (1 + data['true_quarterly']['us_buyout']).cumprod()

axes[0].plot(cum_smooth.index, cum_smooth, color=RED, linewidth=2, label='Smoothed (observed)')
axes[0].plot(cum_true.index, cum_true, color=TEAL, linewidth=1.5, alpha=0.7, label='True (unobservable)')
axes[0].set_title('Cumulative Returns: Smoothed vs True', fontsize=13)
axes[0].legend(frameon=False)
axes[0].set_ylabel('Growth of $1')

# Right: rolling vol comparison
roll_smooth = data['smoothed_quarterly']['us_buyout'].rolling(8).std() * 2
roll_true = data['true_quarterly']['us_buyout'].rolling(8).std() * 2

axes[1].plot(roll_smooth.index, roll_smooth, color=RED, linewidth=2, label='Smoothed vol')
axes[1].plot(roll_true.index, roll_true, color=TEAL, linewidth=1.5, alpha=0.7, label='True vol')
axes[1].set_title('Rolling 2-Year Annualized Volatility', fontsize=13)
axes[1].legend(frameon=False)
axes[1].set_ylabel('Annualized Vol')

plt.tight_layout()
plt.show()
```

---

### Cell 5 — Quickstart: End-to-End in 5 Lines (Markdown + Code)

**Markdown:**
```markdown
## Quickstart: Quarterly PE → Monthly in 5 Lines

This is the minimum viable usage. The library handles desmoothing, factor decomposition, 
and temporal disaggregation automatically using preset configurations.
```

**Code:**
```python
pipeline = FrequencyPipeline(
    returns=data['smoothed_quarterly'],
    factor_returns_monthly=data['factor_returns_monthly'],
    configs={'us_buyout': PE_PRESETS['us_large_buyout']},
)

result = pipeline.run()
print(result.monthly_returns.head(12))
print(f"\nRecovered monthly vol: {result.monthly_returns['us_buyout'].std() * np.sqrt(12):.1%} (annualized)")
```

### Cell 6 — Validate Recovery Against Ground Truth (Code)

```python
# Compare recovered monthly returns to true monthly returns
recovered = result.monthly_returns['us_buyout']
true = data['true_monthly']['us_buyout']

# Align indices
common_idx = recovered.index.intersection(true.index)
recovered_aligned = recovered.loc[common_idx]
true_aligned = true.loc[common_idx]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Correlation
axes[0].scatter(true_aligned, recovered_aligned, alpha=0.3, s=10, color=TEAL)
axes[0].set_xlabel('True Monthly Return')
axes[0].set_ylabel('Recovered Monthly Return')
axes[0].set_title(f'Correlation: {true_aligned.corr(recovered_aligned):.3f}', fontsize=13)

# Volatility comparison
axes[1].bar(['True', 'Smoothed\n(raw quarterly)', 'Recovered\n(monthly)'],
            [true_aligned.std() * np.sqrt(12),
             data['smoothed_quarterly']['us_buyout'].std() * 2,
             recovered_aligned.std() * np.sqrt(12)],
            color=[TEAL, RED, GREEN])
axes[1].set_ylabel('Annualized Volatility')
axes[1].set_title('Volatility Recovery', fontsize=13)

# Rolling difference
rolling_err = (recovered_aligned - true_aligned).rolling(12).mean()
axes[2].plot(rolling_err.index, rolling_err, color=GOLD, linewidth=1.5)
axes[2].axhline(0, color='#555555', linewidth=0.5)
axes[2].set_title('Rolling 12-Month Mean Error', fontsize=13)
axes[2].set_ylabel('Mean Error (recovered − true)')

plt.tight_layout()
plt.show()
```

---

### Cell 7 — Stage-by-Stage Walkthrough: Desmoothing (Markdown + Code)

**Markdown:**
```markdown
## Deep Dive: Stage 1 — Bayesian Desmoothing

Let's look at what the desmoothing stage actually estimates. The library jointly estimates 
the smoothing parameter λ and factor betas β using a Bayesian framework with 
Normal-Inverse-Gamma conjugate priors.
```

**Code:**
```python
# Access desmoothing diagnostics
desmoothing = result.desmoothing_summary['us_buyout']

print("=== Desmoothing Results ===")
print(f"Estimated λ (posterior mean):  {desmoothing['lambda_posterior_mean']:.3f}")
print(f"True λ:                        {data['true_params']['lambda']:.3f}")
print(f"λ 90% credible interval:       [{desmoothing['lambda_ci_5']:.3f}, {desmoothing['lambda_ci_95']:.3f}]")
print(f"\nEstimated β (posterior mean):  {desmoothing['beta_posterior_mean']:.3f}")
print(f"True β:                        {data['true_params']['beta']:.3f}")
print()

# Check for warnings
if result.warnings:
    print("⚠️  Warnings:")
    for w in result.warnings:
        print(f"   - {w}")
else:
    print("✓ No identifiability warnings")
```

### Cell 8 — Desmoothing Diagnostic Plots (Code)

```python
result.plot_desmoothing_diagnostics()
```

**Follow with markdown:**
```markdown
**How to read these plots:**

- **Top-left (λ posterior):** The posterior distribution of the smoothing parameter. 
  A narrow peak means the data is informative; a flat distribution means the output 
  is driven by the prior. The vertical line shows the prior mean.
  
- **Top-right (β posterior):** Same for the factor beta. Compare with the prior 
  (shaded region) — if they overlap heavily, the data isn't moving the estimate much.

- **Bottom-left (prior vs posterior):** Overlay showing how much the data updated our beliefs.

- **Bottom-right (desmoothed vs smoothed returns):** Time series comparison. 
  Desmoothed returns should be more volatile and show larger drawdowns during crises.
```

---

### Cell 9 — Stage-by-Stage: Factor Decomposition (Markdown + Code)

**Markdown:**
```markdown
## Deep Dive: Stage 2 — Factor Decomposition

After desmoothing, the library decomposes the return into systematic (factor-driven) 
and idiosyncratic components. The systematic piece drives the temporal disaggregation; 
the idiosyncratic piece is distributed via Chow-Lin GLS.
```

**Code:**
```python
result.plot_factor_decomposition()

# Access decomposition details
decomp = result.decomposition_summary['us_buyout']
print(f"Factor R²:            {decomp['r_squared']:.3f}")
print(f"Residual autocorr:    {decomp['ljung_box_pvalue']:.3f} (Ljung-Box p-value)")
print(f"Residual normality:   {decomp['jarque_bera_pvalue']:.3f} (Jarque-Bera p-value)")
print(f"Fitted distribution:  {decomp['residual_distribution']}")
```

---

### Cell 10 — Stage-by-Stage: Temporal Disaggregation (Markdown + Code)

**Markdown:**
```markdown
## Deep Dive: Stage 3 — Temporal Disaggregation (Chow-Lin)

This is where quarterly returns become monthly. The systematic component is straightforward: 
monthly factor returns × β. The challenge is distributing the quarterly idiosyncratic 
residual across three months.

The library offers three methods. Let's compare them.
```

**Code:**
```python
methods = ['chow_lin', 'fernandez', 'litterman']
results_by_method = {}

for method in methods:
    p = FrequencyPipeline(
        returns=data['smoothed_quarterly'],
        factor_returns_monthly=data['factor_returns_monthly'],
        configs={'us_buyout': PE_PRESETS['us_large_buyout']},
        disaggregation_method=method,
    )
    results_by_method[method] = p.run()

# Compare
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {'chow_lin': TEAL, 'fernandez': GOLD, 'litterman': GREEN}

for method, res in results_by_method.items():
    monthly = res.monthly_returns['us_buyout']
    cum = (1 + monthly).cumprod()
    axes[0].plot(cum.index, cum, color=colors[method], linewidth=1.5, label=method, alpha=0.8)

axes[0].set_title('Cumulative Returns by Disaggregation Method', fontsize=13)
axes[0].legend(frameon=False)

# Rolling vol comparison
for method, res in results_by_method.items():
    monthly = res.monthly_returns['us_buyout']
    roll_vol = monthly.rolling(24).std() * np.sqrt(12)
    axes[1].plot(roll_vol.index, roll_vol, color=colors[method], linewidth=1.5, label=method, alpha=0.8)

axes[1].set_title('Rolling 2-Year Annualized Volatility', fontsize=13)
axes[1].legend(frameon=False)

plt.tight_layout()
plt.show()
```

**Follow with markdown:**
```markdown
**How to choose:**

- **Chow-Lin** (default): AR(1) residual structure. Best general-purpose choice. 
  Estimates the residual autocorrelation ρ from the data.
- **Fernández**: Random walk residual (ρ=1). Smoother monthly series. 
  Often preferred for credit where MTM shocks persist.
- **Litterman**: AR(1) on first differences. Intermediate behavior. 
  Good when you're unsure about the residual structure.

In practice, the differences are usually small. If they diverge significantly, 
your factor model may be missing something.
```

---

### Cell 11 — Aggregation Consistency Check (Code)

**Markdown:**
```markdown
## Sanity Check: Round-Trip Aggregation

The most important consistency check: monthly returns compounded back to quarterly 
must match the original input **exactly** (within floating-point tolerance).
```

**Code:**
```python
result.plot_aggregation_consistency()

# Manual verification
monthly = result.monthly_returns['us_buyout']
reconstructed_quarterly = (1 + monthly).groupby(
    monthly.index.to_period('Q')
).prod() - 1
reconstructed_quarterly.index = reconstructed_quarterly.index.to_timestamp('Q')

original = data['smoothed_quarterly']['us_buyout']
common = reconstructed_quarterly.index.intersection(original.index)

max_error = (reconstructed_quarterly.loc[common] - original.loc[common]).abs().max()
print(f"Max aggregation error: {max_error:.2e}")
print(f"{'✓ PASS' if max_error < 1e-10 else '✗ FAIL'}: round-trip aggregation consistency")
```

---

### Cell 12 — Multi-Strategy PE Example (Markdown + Code)

**Markdown:**
```markdown
## Multi-Strategy: Buyout + Venture Capital

The library handles multiple strategies simultaneously, preserving cross-strategy 
correlation structure in the disaggregation.
```

**Code — generate multi-strategy data:**
```python
def generate_multi_strategy_data(seed=42):
    """Generate synthetic data for buyout and venture capital."""
    rng = np.random.default_rng(seed)
    n_months = 300  # 25 years
    dates = pd.date_range('2000-01-31', periods=n_months, freq='ME')
    
    # Shared factors
    equity = rng.normal(0.008, 0.045, n_months)
    small_cap = rng.normal(0.003, 0.055, n_months)
    
    factors = pd.DataFrame({
        'equity_market': equity,
        'small_cap_growth': small_cap,
    }, index=dates)
    
    # True monthly returns
    eps_buyout = rng.normal(0, 0.025, n_months)
    eps_vc = rng.normal(0, 0.04, n_months)
    # Correlated idiosyncratic shocks
    eps_vc = 0.3 * eps_buyout + np.sqrt(1 - 0.3**2) * eps_vc
    
    true_buyout = 1.15 * equity + 0.002 + eps_buyout
    true_vc = 0.5 * equity + 0.7 * small_cap + 0.001 + eps_vc
    
    true_monthly = pd.DataFrame({
        'us_buyout': true_buyout,
        'us_early_vc': true_vc,
    }, index=dates)
    
    # Aggregate and smooth
    quarterly_frames = []
    for col in true_monthly.columns:
        lam = 0.55 if 'buyout' in col else 0.65  # VC has higher smoothing
        q = (1 + true_monthly[col]).groupby(true_monthly.index.to_period('Q')).prod() - 1
        q.index = q.index.to_timestamp('Q')
        smoothed = q.copy()
        for i in range(1, len(q)):
            smoothed.iloc[i] = (1 - lam) * q.iloc[i] + lam * smoothed.iloc[i-1]
        quarterly_frames.append(smoothed.rename(col))
    
    smoothed_q = pd.concat(quarterly_frames, axis=1)
    return smoothed_q, factors, true_monthly

multi_quarterly, multi_factors, multi_true = generate_multi_strategy_data()
```

**Code — run pipeline:**
```python
multi_pipeline = FrequencyPipeline(
    returns=multi_quarterly,
    factor_returns_monthly=multi_factors,
    configs={
        'us_buyout': PE_PRESETS['us_large_buyout'],
        'us_early_vc': PE_PRESETS['us_early_venture'],
    },
    fallback_policy='warn',
)

multi_result = multi_pipeline.run()

# Compare correlations
raw_corr = multi_quarterly.corr().iloc[0, 1]
recovered_corr = multi_result.monthly_returns.corr().iloc[0, 1]
true_corr = multi_true.corr().iloc[0, 1]

print(f"Cross-strategy correlation:")
print(f"  Raw (smoothed quarterly):  {raw_corr:.3f}")
print(f"  Recovered (monthly):       {recovered_corr:.3f}")
print(f"  True:                      {true_corr:.3f}")
```

---

### Cell 13 — Hedge Fund Example: MA(q) Desmoothing (Markdown + Code)

**Markdown:**
```markdown
## Hedge Fund Example: MA(q) Desmoothing

Hedge fund smoothing is fundamentally different from PE — it's caused by illiquid holdings 
and stale prices (a moving average process), not by appraisal lag (an autoregressive process). 
The library uses the Getmansky-Lo-Makarov (2004) MA(q) model for hedge funds.
```

**Code — generate HF data:**
```python
def generate_synthetic_hf_data(seed=123):
    """Generate synthetic hedge fund data with MA(q) smoothing."""
    rng = np.random.default_rng(seed)
    n_months = 240  # 20 years
    dates = pd.date_range('2004-01-31', periods=n_months, freq='ME')
    
    # Factors
    equity = rng.normal(0.007, 0.042, n_months)
    credit = rng.normal(0.003, 0.018, n_months)
    
    factors_daily = None  # Could generate daily factors here for daily extension
    factors_monthly = pd.DataFrame({
        'equity_market': equity,
        'credit_spread': credit,
    }, index=dates)
    
    # True monthly HF returns (equity long-short)
    eps = rng.normal(0, 0.012, n_months)
    true_monthly = 0.4 * equity + 0.15 * credit + 0.002 + eps
    
    # Apply MA(2) smoothing: s_t = 0.6*r_t + 0.25*r_{t-1} + 0.15*r_{t-2}
    theta = [0.6, 0.25, 0.15]
    smoothed = np.zeros(n_months)
    for t in range(n_months):
        for j, th in enumerate(theta):
            if t - j >= 0:
                smoothed[t] += th * true_monthly[t - j]
    
    hf_returns = pd.DataFrame({
        'equity_ls': smoothed,
    }, index=dates)
    
    return hf_returns, factors_monthly, theta

hf_data, hf_factors, true_theta = generate_synthetic_hf_data()
print(f"True MA weights: θ = {true_theta}")
print(f"HF data: {len(hf_data)} monthly observations")
```

**Code — run HF pipeline:**
```python
hf_pipeline = FrequencyPipeline(
    returns=hf_data,
    factor_returns_monthly=hf_factors,
    configs={'equity_ls': HF_PRESETS['equity_long_short']},
    fallback_policy='warn',
)

hf_result = hf_pipeline.run()

# Check recovered theta
hf_desmoothing = hf_result.desmoothing_summary['equity_ls']
print(f"\nEstimated θ: {hf_desmoothing['theta_posterior_mean']}")
print(f"True θ:      {true_theta}")
print(f"\nSmoothed vol:    {hf_data['equity_ls'].std() * np.sqrt(12):.1%}")
print(f"Desmoothed vol:  {hf_result.desmoothed_returns['equity_ls'].std() * np.sqrt(12):.1%}")
```

---

### Cell 14 — Private Credit Example: Carry/MTM + Threshold AR(1) (Markdown + Code)

**Markdown:**
```markdown
## Private Credit: Carry/MTM Decomposition + Threshold AR(1)

Private credit requires two special steps:
1. **Stage 0:** Decompose total return into carry (coupon income) and mark-to-market
2. **Stage 1:** Apply regime-dependent desmoothing only to the MTM component

The carry component is not smoothed — it genuinely accrues at a steady rate. 
Desmoothing it would create fake volatility.
```

**Code — generate credit data:**
```python
def generate_synthetic_credit_data(seed=456):
    """Generate synthetic private credit data with carry + MTM and regime-dependent smoothing."""
    rng = np.random.default_rng(seed)
    n_quarters = 80  # 20 years
    dates_q = pd.date_range('2004-03-31', periods=n_quarters, freq='QE')
    
    # Carry component: ~8% annual yield, accruing steadily
    quarterly_carry = np.full(n_quarters, 0.02)  # 2% per quarter
    
    # MTM component: spread-driven, with regime-dependent smoothing
    hy_oas = 400 + rng.normal(0, 80, n_quarters).cumsum()  # mean-reverting-ish OAS
    hy_oas = np.clip(hy_oas, 200, 1200)  # bound it
    
    # True MTM returns
    spread_change = np.diff(np.insert(hy_oas, 0, 400))
    true_mtm = -0.04 * spread_change / 100 + rng.normal(0, 0.01, n_quarters)
    
    # Regime: stress when OAS > 600
    regime_stress = hy_oas > 600
    lambda_vals = np.where(regime_stress, 0.15, 0.7)  # low smoothing in stress
    
    # Apply threshold smoothing to MTM only
    smoothed_mtm = np.zeros(n_quarters)
    smoothed_mtm[0] = true_mtm[0]
    for i in range(1, n_quarters):
        smoothed_mtm[i] = (1 - lambda_vals[i]) * true_mtm[i] + lambda_vals[i] * smoothed_mtm[i-1]
    
    total_return = quarterly_carry + smoothed_mtm
    
    credit_returns = pd.DataFrame({'direct_lending': total_return}, index=dates_q)
    yield_series = pd.DataFrame({'direct_lending': quarterly_carry}, index=dates_q)
    regime = pd.Series(hy_oas, index=dates_q, name='HY_OAS')
    
    # Factors (monthly)
    n_months = n_quarters * 3
    dates_m = pd.date_range('2004-01-31', periods=n_months, freq='ME')
    factors = pd.DataFrame({
        'lev_loan_index': rng.normal(0.004, 0.012, n_months),
        'hy_oas_change': rng.normal(0, 0.008, n_months),
    }, index=dates_m)
    
    return credit_returns, yield_series, regime, factors

credit_returns, yield_series, regime_indicator, credit_factors = generate_synthetic_credit_data()
```

**Code — run credit pipeline:**
```python
credit_pipeline = FrequencyPipeline(
    returns=credit_returns,
    factor_returns_monthly=credit_factors,
    yield_series=yield_series,
    regime_indicator=regime_indicator,
    configs={'direct_lending': CREDIT_PRESETS['direct_lending']},
    disaggregation_method='fernandez',  # random walk often better for credit
    fallback_policy='warn',
)

credit_result = credit_pipeline.run()

print(f"Smoothed quarterly vol:  {credit_returns['direct_lending'].std() * 2:.1%}")
print(f"Recovered monthly vol:   {credit_result.monthly_returns['direct_lending'].std() * np.sqrt(12):.1%}")
```

---

### Cell 15 — Uncertainty Propagation (Markdown + Code)

**Markdown:**
```markdown
## Uncertainty Propagation: How Confident Are We?

Point estimates hide how uncertain the recovery is. With `uncertainty_mode='full'`, the library 
samples from the joint posterior of (λ, β, σ_ε) and runs the full pipeline for each draw, 
producing confidence bands on the monthly returns.
```

**Code:**
```python
uncertain_pipeline = FrequencyPipeline(
    returns=data['smoothed_quarterly'],
    factor_returns_monthly=data['factor_returns_monthly'],
    configs={'us_buyout': PE_PRESETS['us_large_buyout']},
    uncertainty_mode='full',  # Sample from posterior
)

uncertain_result = uncertain_pipeline.run()

# Plot uncertainty bands
bands = uncertain_result.monthly_returns_bands
median = bands['50%']['us_buyout']
p5 = bands['5%']['us_buyout']
p95 = bands['95%']['us_buyout']

cum_median = (1 + median).cumprod()
cum_p5 = (1 + p5).cumprod()
cum_p95 = (1 + p95).cumprod()

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(cum_p5.index, cum_p5, cum_p95, alpha=0.25, color=TEAL, label='90% CI')
ax.plot(cum_median.index, cum_median, color=TEAL, linewidth=2, label='Median')

# Overlay true
cum_true = (1 + data['true_monthly']['us_buyout']).cumprod()
ax.plot(cum_true.index, cum_true, color=GOLD, linewidth=1, alpha=0.6, label='True (unobservable)')

ax.set_title('Recovered Monthly Returns with Uncertainty Bands', fontsize=13)
ax.legend(frameon=False)
ax.set_ylabel('Growth of $1')
plt.tight_layout()
plt.show()
```

---

### Cell 16 — Monte Carlo Simulation (Markdown + Code)

**Markdown:**
```markdown
## Monte Carlo Simulation

For VaR, CVaR, and scenario analysis, the library generates N simulation paths 
at daily or monthly frequency, drawing from the estimated factor model and 
idiosyncratic distribution.
```

**Code:**
```python
sim = result.simulate(
    n_paths=5000,
    horizon_months=60,  # 5-year forward simulation
    frequency='monthly',
    seed=42,
    correlation_method='empirical',
)

# sim.paths shape: (n_paths, horizon_months, n_strategies)
terminal_wealth = (1 + sim.paths[:, :, 0]).prod(axis=1)  # buyout terminal value

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fan chart: percentile paths
percentiles = [5, 25, 50, 75, 95]
cum_paths = (1 + sim.paths[:, :, 0]).cumprod(axis=1)
months = np.arange(1, 61)

for p in percentiles:
    val = np.percentile(cum_paths, p, axis=0)
    axes[0].plot(months, val, color=TEAL if p == 50 else GOLD, 
                 linewidth=2 if p == 50 else 1, alpha=0.5 if p != 50 else 1,
                 label=f'{p}th percentile')

axes[0].set_title('Simulated Wealth Paths (5-Year Horizon)', fontsize=13)
axes[0].set_xlabel('Months')
axes[0].set_ylabel('Growth of $1')
axes[0].legend(frameon=False, fontsize=9)

# Terminal wealth distribution
axes[1].hist(terminal_wealth, bins=80, color=TEAL, alpha=0.7, edgecolor='none')
var_95 = np.percentile(terminal_wealth, 5)
axes[1].axvline(var_95, color=RED, linewidth=2, linestyle='--', label=f'5% VaR: {var_95:.2f}x')
axes[1].set_title('Terminal Wealth Distribution (5-Year)', fontsize=13)
axes[1].set_xlabel('Terminal Wealth (multiple of initial)')
axes[1].legend(frameon=False)

plt.tight_layout()
plt.show()

print(f"Simulation summary (5-year horizon, {sim.paths.shape[0]} paths):")
print(f"  Median terminal wealth:  {np.median(terminal_wealth):.2f}x")
print(f"  5th percentile (VaR):    {var_95:.2f}x")
print(f"  95th percentile:         {np.percentile(terminal_wealth, 95):.2f}x")
```

---

### Cell 17 — Custom Configuration (Markdown + Code)

**Markdown:**
```markdown
## Custom Configuration: Building Your Own Preset

The presets are starting points. For a specific fund or index with known characteristics, 
you'll want to customize the priors and factor model.
```

**Code:**
```python
# Example: EM Private Equity fund with known characteristics
em_pe_config = AssetClassConfig(
    asset_class='private_equity',
    smoothing_model='ar1_bayesian',
    preprocessing=None,
    lambda_prior=BetaDist(3, 2),               # Expect higher smoothing
    beta_priors={
        'equity_market': NormalPrior(0.9, 0.40),   # EM beta to global equity
        'credit_spread': NormalPrior(0.3, 0.20),   # EM credit sensitivity
    },
    alpha_prior=NormalPrior(0.02, 0.03),           # Expect some alpha
    default_factors=['MSCI_EM', 'EMBI'],
    public_proxy='MSCI Emerging Markets',
    native_frequency='quarterly',
)

# Show available presets
print("Available presets by asset class:\n")
for name, preset in {**PE_PRESETS, **HF_PRESETS, **CREDIT_PRESETS, **INFRA_PRESETS, **RE_PRESETS}.items():
    print(f"  {name:30s}  model={preset.smoothing_model:20s}  freq={preset.native_frequency}")
```

---

### Cell 18 — Sanity Anchors & Warnings (Markdown + Code)

**Markdown:**
```markdown
## Interpreting Warnings & Sanity Anchors

The library performs automated sanity checks comparing outputs against public market 
benchmarks. These aren't pass/fail — they flag when results look economically implausible.
```

**Code:**
```python
print("=== Sanity Anchor Report ===\n")
for check in result.sanity_anchors:
    status = '✓' if check.passed else '⚠️'
    print(f"  {status} {check.name}: {check.message}")
    if not check.passed:
        print(f"     → {check.recommendation}")

print(f"\n=== Pipeline Warnings ===\n")
if result.warnings:
    for w in result.warnings:
        print(f"  ⚠️  {w}")
else:
    print("  ✓ No warnings")
```

---

### Cell 19 — Comparing Disaggregation Methods: Additive vs Multiplicative (Markdown + Code)

**Markdown:**
```markdown
## Additive vs Multiplicative Aggregation

For small returns the difference is negligible. For PE-magnitude returns (±20% in crises), 
it matters. Always use multiplicative (the default) unless you have a specific reason not to.
```

**Code:**
```python
# Run both
add_pipeline = FrequencyPipeline(
    returns=data['smoothed_quarterly'],
    factor_returns_monthly=data['factor_returns_monthly'],
    configs={'us_buyout': PE_PRESETS['us_large_buyout']},
    aggregation_type='additive',
)
add_result = add_pipeline.run()

mult_result = result  # Already computed with multiplicative (default)

# Compare during crisis periods
crisis_start, crisis_end = '2008-07', '2009-03'
add_crisis = add_result.monthly_returns.loc[crisis_start:crisis_end, 'us_buyout']
mult_crisis = mult_result.monthly_returns.loc[crisis_start:crisis_end, 'us_buyout']

print(f"GFC period ({crisis_start} to {crisis_end}):")
print(f"  Additive cumulative:       {(1 + add_crisis).prod() - 1:.2%}")
print(f"  Multiplicative cumulative: {(1 + mult_crisis).prod() - 1:.2%}")
print(f"  Difference:                {((1+mult_crisis).prod() - (1+add_crisis).prod()):.4%}")
print(f"\n  → Difference is small here but can reach 50+ bps in extreme quarters")
```

---

### Cell 20 — Production Tips & Common Pitfalls (Markdown)

```markdown
## Production Tips & Common Pitfalls

### Do ✓

- **Always check `result.warnings` before using outputs.** If λ is poorly identified, 
  your monthly returns are prior-driven, not data-driven.
  
- **Use `uncertainty_mode='full'` for any risk calculation.** Point estimates create 
  false precision. The 90% CI on monthly vol can be ±30% of the point estimate.

- **Validate with round-trip aggregation.** Monthly compounded to quarterly must match 
  the input exactly. If it doesn't, there's a bug.
  
- **Use multiplicative aggregation.** Always. The additive approximation is only acceptable 
  for very small returns.

- **Choose priors carefully.** The beta prior should reflect economic reality — leverage 
  for buyouts, duration for infrastructure, credit quality for mezzanine. Wrong priors 
  with short history = wrong outputs.

### Don't ✗

- **Don't use raw quarterly PE returns for risk models.** The whole point of this library 
  is that they're dangerously misleading. Raw vol understates by ~60%, raw correlation by ~30%.

- **Don't skip Stage 1 (desmoothing).** If you feed smoothed data directly to Chow-Lin, 
  you get smoothed monthly data. Garbage in, garbage out.

- **Don't over-interpret the daily path (Kalman Mode A).** It's a conditional expectation, 
  not a realization. The daily path within a month is heavily interpolated.

- **Don't assume stationarity.** If your PE strategy's leverage or style has changed 
  meaningfully over time, the full-sample beta estimate is an average, not the current exposure.

- **Don't ignore the carry/MTM decomposition for credit.** Desmoothing the carry component 
  creates spurious volatility. Always decompose first.

### When Results Look Wrong

1. Check if λ posterior is flat → data is uninformative, output is prior-driven
2. Check if factor R² is very low → missing systematic factors
3. Check if desmoothed vol is > 3× public proxy vol → over-desmoothing (λ too high)
4. Check if cross-strategy correlation is > 0.95 → everything is just the equity factor
5. Run with `fallback_policy='strict'` to see all warnings as errors
```

---

### Cell 21 — Next Steps (Markdown)

```markdown
## Next Steps

- **Bring your own data:** Replace the synthetic generators with your actual PE/HF/credit indices 
  and Barra factor returns. The API is identical.

- **Custom factor models:** Add factors beyond equity/credit — sector, size, value, momentum, 
  geography. More factors = better decomposition = better disaggregation.

- **Daily extension:** Provide daily factor returns to get daily synthetic returns via Kalman 
  smoother or simulation.

- **Full methodology details:** See the MSCI Private Equity Factor Model (2025) paper and 
  Getmansky, Lo, Makarov (2004) for the theoretical foundations.
```

---

## Implementation Notes for Claude Code

1. **The notebook must be fully runnable.** Every cell must execute without error when the library is installed. Use `try/except` blocks around API calls that depend on library implementation details — if a method doesn't exist yet, print a placeholder message rather than crashing.

2. **Synthetic data generators are critical.** They must produce data that the library can actually process. The generators should match the data format the pipeline expects: `pd.DataFrame` with `DatetimeIndex` at the correct frequency, column names matching the config strategy names.

3. **Plot styling is important.** All plots should use the dark theme defined in Cell 2. Use the TEAL/GOLD/RED/GREEN color constants consistently. Legends should have `frameon=False`. Titles at 13pt.

4. **Cell output matters.** After each code cell, include a markdown cell interpreting what the user should see and what it means. The notebook is a teaching tool, not just a code dump.

5. **Save the notebook to:** `docs/tutorial.ipynb`

6. **Test the notebook:** After creating it, run all cells using `jupyter nbconvert --execute` to verify everything runs. Fix any errors.
