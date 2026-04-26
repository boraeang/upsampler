# private_assets_frequency

Production-grade Python library for converting low-frequency private and
alternative asset returns into higher-frequency synthetic series that
preserve the true risk characteristics — correcting for the well-documented
appraisal-smoothing biases before disaggregation.

Supports five asset classes out of the box:

| Asset class | Smoothing model | Native frequency | Stage 0 preprocessing |
|---|---|---|---|
| Private equity (buyout, venture, mezzanine, distressed) | AR(1) Bayesian | Quarterly | None |
| Private infrastructure (core, opportunistic) | AR(1) Bayesian | Quarterly | None |
| Private credit (direct lending, senior debt) | Threshold AR(1) | Quarterly | Carry / MTM split |
| Hedge funds (eq L/S, macro, event-driven, RV, managed futures) | MA(q) Getmansky-Lo-Makarov / NoSmoothing | Monthly | Reporting-lag adjustment |
| Private real estate (core) | AR(1) Bayesian | Quarterly | None |

The pipeline architecture is asset-class generic — Stages 0–4 are pluggable
at every dimension that matters (preprocessor, smoothing model, factor
priors, disaggregation method, daily extension).

---

## Why this exists

Private and alternative asset indices report NAV-based or appraisal-based
returns that dramatically understate true volatility — typically by 50 % or
more — and dramatically overstate diversification benefits relative to
public markets. **Any backtest, VaR model, or simulation built directly on
the raw indices produces dangerously misleading risk numbers.** Geltner
(1993), Getmansky-Lo-Makarov (2004), and a long line of subsequent
literature document the smoothing mechanisms; this library implements those
desmoothing methodologies on a multi-stage pipeline that also handles the
downstream temporal disaggregation needed to actually use the results in
daily-frequency models.

---

## Installation

Requires Python ≥ 3.10. Core dependencies are NumPy / SciPy / pandas /
statsmodels:

```bash
pip install private_assets_frequency
```

Optional extras:

```bash
pip install "private_assets_frequency[viz]"       # matplotlib for plotting
pip install "private_assets_frequency[calendar]"  # pandas_market_calendars (NYSE holidays)
pip install "private_assets_frequency[dev]"       # pytest, mypy, ruff
```

The library will fall back to `pandas.bdate_range` (Mon–Fri) if
`pandas_market_calendars` is not installed; that is fine for testing /
non-US use cases.

---

## Pipeline architecture

```
[Raw low-frequency returns]
    │
    ▼
Stage 0: Return Preprocessing
    – credit: carry / mark-to-market decomposition
    – HF: reporting-lag adjustment
    – PE / RE / infra: pass-through
    │
    ▼
Stage 1: Desmoothing  (pluggable model)
    – AR(1) Bayesian (PE / infra / RE)
    – MA(q) Getmansky-Lo-Makarov (hedge funds)
    – Threshold AR(1) (credit)
    – NoSmoothing / Geltner classic (baselines)
    │
    ▼
Stage 2: Factor Decomposition
    – OLS / WLS factor regression at native frequency
    – residual diagnostics: Ljung-Box, ARCH-LM, JB
    – best-fit residual distribution: normal / Student-t / Hansen skewed-t
    │
    ▼
Stage 3: Temporal Disaggregation
    – Chow-Lin (1971) AR(1) residuals
    – Fernández (1981) RW residuals
    – Litterman (1983) AR(1) on differences
    – multiplicative or additive aggregation constraint
    │
    ▼
Stage 4: Daily Extension
    – Mode A: Kalman smoother / GLS BLUE for point estimation
    – Mode B: Monte Carlo simulation for VaR / CVaR
    – business-day calendar (NYSE default, configurable)
```

Every stage emits a `StageValidationResult` so the pipeline runner knows
when a stage has produced suspect output. The `FallbackPolicy` controls
whether warnings are silently auto-fixed (`auto`), surfaced on the result
(`warn`, default), or escalated to exceptions (`strict`).

---

## Quickstart

### Private equity — quarterly → monthly → daily

```python
import pandas as pd
from private_assets_frequency.pipeline.runner import FrequencyPipeline
from private_assets_frequency.pipeline.presets import PE_PRESETS

pipeline = FrequencyPipeline(
    returns=quarterly_pe_returns_df,            # DatetimeIndex, columns = strategies
    factor_returns_monthly=monthly_factors_df,  # for Stage 3 quarterly→monthly
    factor_returns_daily=daily_factors_df,      # for Stage 4 monthly→daily
    configs={
        "us_buyout": PE_PRESETS["us_large_buyout"],
        "us_vc":     PE_PRESETS["us_early_venture"],
    },
    disaggregation_method="chow_lin",
    aggregation_type="multiplicative",
    fallback_policy="warn",
    uncertainty_mode="point",
)

result = pipeline.run()
monthly = result.monthly_returns        # DataFrame
daily   = result.daily_returns          # DataFrame or None
warnings_list = result.warnings         # list[str]
sanity_per_strategy = result.sanity_anchors
```

### Hedge funds — monthly → daily

```python
from private_assets_frequency.pipeline.presets import HF_PRESETS

pipeline = FrequencyPipeline(
    returns=monthly_hf_returns_df,
    factor_returns_monthly=monthly_factors_df,
    factor_returns_daily=daily_factors_df,
    configs={
        "eq_ls": HF_PRESETS["equity_long_short"],
        "macro": HF_PRESETS["global_macro"],
    },
)
result = pipeline.run()
```

### Private credit — quarterly → monthly with carry / MTM split

```python
from private_assets_frequency.pipeline.presets import CREDIT_PRESETS

pipeline = FrequencyPipeline(
    returns=quarterly_credit_df,
    factor_returns_monthly=monthly_factors_df,
    yield_series=quarterly_yield_df,         # required for carry/MTM
    regime_indicator=hy_oas_regime_df,       # required for threshold AR(1)
    configs={"direct_lending": CREDIT_PRESETS["direct_lending"]},
    disaggregation_method="fernandez",       # RW residuals fit credit better
)
result = pipeline.run()
```

### Posterior uncertainty bands

```python
pipeline = FrequencyPipeline(
    returns=quarterly_pe_returns_df,
    factor_returns_monthly=monthly_factors_df,
    configs={"us_buyout": PE_PRESETS["us_large_buyout"]},
    uncertainty_mode="full",                 # propagate λ posterior
    uncertainty_n_samples=100,
)
result = pipeline.run()
bands = result.uncertainty_bands
bands.monthly[5]    # 5th percentile  monthly DataFrame
bands.monthly[50]   # 50th percentile (median)
bands.monthly[95]   # 95th percentile
```

`uncertainty_mode='full'` draws posterior samples of the smoothing
parameters from the desmoother (currently AR(1) Bayesian; MA(q) is wired
through too), re-runs Stage 3 for each draw, and stacks the resulting
monthly paths into pointwise percentile bands. With 100 samples the
runtime is roughly 100× the point-estimate pipeline — still tractable for
typical horizons.

### Custom configuration

```python
from private_assets_frequency.core.config import (
    AssetClassConfig, NormalPrior, BetaDist, InverseGammaPrior,
)

custom = AssetClassConfig(
    asset_class="private_equity",
    smoothing_model="ar1_bayesian",
    native_frequency="quarterly",
    lambda_prior=BetaDist(3, 2),
    beta_priors={
        "equity_market": NormalPrior(0.9, 0.40),
        "credit_spread": NormalPrior(0.3, 0.20),
    },
    alpha_prior=NormalPrior(0.02, 0.03),
    sigma_eps_prior=InverseGammaPrior(3.0, 0.02),
    default_factors=("MSCI_EM", "EMBI"),
    public_proxy="MSCI Emerging Markets",
)
```

---

## Methodology overview

### Stage 1 — desmoothing

#### AR(1) Bayesian (PE / infrastructure / real estate)

The appraisal-smoothing process is

$$
s_t = (1 - \lambda) r_t + \lambda s_{t-1}
\quad\Longrightarrow\quad
r_t = \frac{s_t - \lambda s_{t-1}}{1 - \lambda}.
$$

Joint posterior of `(λ, β, α, σ_ε)` via grid integration over λ ∈ [0.01, 0.95]
(50 points) with a closed-form Normal-Inverse-Gamma conjugate posterior on
`(β, α, σ²)` at each λ. The marginal log-likelihood includes the
`-n · log(1 − λ)` Jacobian of the desmoothing transform — without that
correction the fit is biased toward small λ. Integration over λ uses
`scipy.integrate.trapezoid`.

Identifiability diagnostics fire when:

* the 90 % credible interval on λ is wider than 0.4
* `KL(posterior‖prior)` on λ is below 0.1 nats
* β changes sign or magnitude > 50 % across the λ grid

Empirically the posterior mean recovers λ within the 90 % CI on ≥ 93 % of
seeds at λ ∈ {0.3, 0.5, 0.6, 0.75, 0.85} with 30 years of quarterly data;
the credible interval narrows monotonically as λ → 1 (high-λ AR(1) is
more identifiable).

#### MA(q) Getmansky-Lo-Makarov (hedge funds)

Smoothing process: `s_t = Σ θ_j r_{t-j}` with `Σ θ_j = 1`, `θ_j ≥ 0`.
Inversion via `scipy.signal.lfilter([1.0], θ, s)`. Bayesian estimation
over an ordered-Dirichlet simplex grid for q ≤ 3, with a `θ_0 ≥ 0.2`
floor and a Dirichlet(5, 3, 2, 1) prior over the ordered region. q ≥ 4
needs a Laplace-approximation extension (not yet shipped).

Diagnostic warnings fire when posterior `θ_0 < 0.3` or `cond(Θ) > 100` —
both are signals that the inversion is amplifying noise.

#### Threshold AR(1) (private credit)

Two regime-specific smoothing parameters `λ_normal` and `λ_stress`,
selected by an exogenous binary regime indicator (typically HY OAS above
a configurable threshold). 2-D grid over `(λ_normal, λ_stress)`,
regime-aware Jacobian `−n_n log(1 − λ_n) − n_s log(1 − λ_s)`. Posterior
flags when stress-regime mass exceeds normal-regime mass (contradicts
the model premise — usually means the regime threshold is mis-calibrated
for the index vintage).

### Stage 2 — factor decomposition

OLS or exponential-decay-weighted LS factor regression, residual
diagnostics (Ljung-Box, ARCH-LM, Jarque-Bera) and best-fit distribution
out of normal / Student-t / Hansen (1994) skewed-t. Residual covariance
across strategies is also computed for the multivariate Stage 3 / Stage 4
correlation rotation.

### Stage 3 — temporal disaggregation

Chow-Lin (AR(1) residuals, `ρ` profile-likelihood grid), Fernández
(random-walk residuals), or Litterman (AR(1) on differences). All three
satisfy the multiplicative aggregation constraint `(1 + r_LF) = ∏(1 +
r_HF)` to machine precision (round-trip < 1e-10) by working in
log-return space. Multivariate variant rotates within-block residual
deviations to match a target cross-strategy correlation while leaving
the per-strategy aggregation constraint exact.

### Stage 4 — daily extension

Mode A (Kalman / GLS BLUE) produces a single conditional-expectation
daily path; Mode B (Monte Carlo) produces N constraint-respecting paths
sampled from the per-strategy fitted residual distribution with optional
Gaussian-copula cross-strategy correlation. Both modes use the irregular
business-day-block aggregation matrix to handle the month-to-month
variation in business-day counts.

---

## Validation, sanity anchors, and fallbacks

Every pipeline run produces:

* **Inter-stage validation** — Stage 1→2 (desmoothed σ > observed σ),
  Stage 2→3 (residuals approximately orthogonal to factors), Stage 3→4
  (round-trip aggregation < 1e-10). Each emits a `StageValidationResult`.
* **Sanity anchors** vs. a public proxy:
  - Volatility ratio in `[0.5×, 3.0×]`
  - Correlation in `[0.4, 0.95]`
  - Crisis drawdown ≥ 50 % × proxy's
  - Beta in a configurable economic-leverage band
* **`FallbackPolicy`**: `'strict'` raises on any warning; `'warn'`
  (default) accumulates warnings on the result; `'auto'` applies stage-
  specific mitigations silently (e.g. fall back to `GeltnerClassicSmoother`
  with `λ = 0.5` if the Bayesian fit is un-identified).

---

## Caveats

> The pipeline produces **synthetic high-frequency returns consistent with
> the observed low-frequency data under the chosen model assumptions** —
> not "the true monthly returns." Mode A (Kalman smoother) outputs are
> conditional expectations; Mode B (Monte Carlo) outputs are sampled
> realisations. The library deliberately uses the language "synthetic" /
> "model-implied" in result objects and docstrings so callers don't
> mistake a model output for ground truth.

Specific known limitations:

* **MA(q) for q ≥ 4** is not yet supported — the spec recommends Laplace
  approximation for q > 3 and this build raises `NotImplementedError` for
  those cases. The event-driven HF preset is therefore pinned to q=3
  rather than q=4.
* **Long-span Kalman smoothing.** The current implementation forms an
  `n × n` AR(1) covariance matrix densely. For 30-year daily horizons
  (~7500 states) this is ~450 MB and starts to bite; the spec recommends
  switching to scipy's banded primitives.
* **Stage 4 HF disaggregation** uses a uniform `ratio = 21` for
  monthly→daily Chow-Lin. Real business-day months span 19–23 days; the
  irregular-block path lives in `daily/calendar.py` + `daily/kalman.py`
  and should be wired through Stage 3 for production VaR.
* **Cross-strategy multivariate Chow-Lin** is currently a marginal-per-
  strategy fit followed by a within-block rotation — not a true joint
  multivariate GLS. For most use cases this is what's wanted (the LF
  correlation is structural; the HF deviation correlation is what risk
  numbers care about); rare cases where this distinction matters will
  need the joint formulation.
* **`uncertainty_mode='full'`** currently propagates the λ / θ posterior
  (the dominant source of identifiability uncertainty) but holds the
  factor model's β at the posterior mean. Full joint propagation through
  Stage 4 is a follow-up.

---

## Testing & validation

The library ships ~475 tests covering:

* Synthetic round-trip recovery for every desmoothing model (the AR(1)
  Bayesian estimator recovers λ within the 90 % CI on at least 90 % of
  seeds across λ ∈ {0.3, 0.5, 0.6, 0.75, 0.85} with 30 yr of data).
* Stage 3 round-trip aggregation < 1e-10 across all three methods × both
  aggregation types × 10 random seeds.
* Stage 4 round-trip < 1e-10 on the multiplicative path for both Kalman
  Mode A and the Monte Carlo simulator.
* End-to-end pipeline tests for each asset class (PE, HF, credit) on the
  conftest synthetic datasets.
* Validation diagnostics: Ljung-Box / ARCH-LM / JB / Hansen-skewed-t fit.

Run with:

```bash
pytest private_assets_frequency/tests
```

---

## References

* Geltner (1993) — "Estimating Market Values from Appraised Values without
  Assuming an Efficient Market," *Journal of Real Estate Finance and
  Economics* 8: 325–345.
* Getmansky, Lo, Makarov (2004) — "An econometric model of serial
  correlation and illiquidity in hedge fund returns," *Journal of Financial
  Economics* 74: 529–609.
* Chow & Lin (1971) — "Best linear unbiased interpolation, distribution,
  and extrapolation of time series by related series," *Review of
  Economics and Statistics* 53(4): 372–375.
* Fernández (1981) — "A methodological note on the estimation of time
  series," *Review of Economics and Statistics* 63: 471–476.
* Litterman (1983) — "A random walk, Markov model for the distribution of
  time series."
* Hansen (1994) — "Autoregressive conditional density estimation,"
  *International Economic Review* 35: 705–730.
* Mariano & Murasawa (2003) — "A new coincident index of business cycles
  based on monthly and quarterly series."
* Murphy (2007) — "Conjugate Bayesian analysis of the Gaussian
  distribution," technical note.
* Shepard (2014) — MSCI Bayesian desmoothing methodology.
* MSCI (2025) — "The MSCI Private Equity Factor Model."

---

## Design notes

A more detailed discussion of the top three identifiability / fragility
risks in the pipeline and how the implementation mitigates each lives in
[`DESIGN_NOTES.md`](./DESIGN_NOTES.md). Read that before running this on
production data — the model risk is real and the diagnostics are the
primary defence.

---

## Decision priorities

When the implementation has to make a call between competing concerns, it
applies the following priority order (in descending order):

1. Statistical validity > architectural purity
2. Numerical stability > theoretical elegance
3. Robustness to small samples > asymptotic correctness
4. Interpretability > marginal accuracy gains
