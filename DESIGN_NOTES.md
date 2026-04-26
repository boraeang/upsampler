# DESIGN_NOTES.md — `private_assets_frequency`

## Purpose of this document

The spec mandates that before any code is written, a senior-quant review surface
the **top 3 identifiability / fragility risks** in the pipeline along with the
concrete design choices that mitigate each. The point is to commit the model-risk
posture in writing so it can be checked against the eventual implementation,
rather than discovered during debugging.

Pipeline under review: Stage 0 (preprocessing) → Stage 1 (desmoothing) →
Stage 2 (factor decomposition) → Stage 3 (temporal disaggregation) →
Stage 4 (daily extension), per
`docs/claude_code_prompt_private_assets_frequency_v2.md`.

The decision priorities apply throughout: statistical validity > architectural
purity; numerical stability > theoretical elegance; small-sample robustness >
asymptotic correctness; interpretability > marginal accuracy.

---

## Risk 1 — λ ↔ β confounding in the Bayesian AR(1) desmoother (PE, Infra, RE)

### The problem

The AR(1) desmoothing model
`r_t = (s_t − λ·s_{t−1}) / (1 − λ)`
is jointly fit with a factor regression
`r_t = β·F_t + α + ε_t`.
With ~30 years of quarterly data (≈120 obs, fewer for newer indices) the
likelihood ridge in (λ, β) space is *long and shallow*: a high-λ /
high-β pair produces a smoothed series that is nearly indistinguishable from
a moderate-λ / moderate-β pair. The information that separates them lives in
the higher-order autocovariance of the residuals — exactly where small-sample
noise dominates. As λ → 1 the desmoothing transform `1/(1−λ)` also blows up
numerically, so any prior weight near λ=1 distorts posterior moments. Net
effect: a naïve MAP or MLE on its own will quietly produce a desmoothed series
whose volatility / beta is driven by the prior, not the data, and the user
will not see it.

### Mitigations baked into the architecture

1. **Rolling-annual returns rather than raw quarterly** (Stage 1 spec point 1).
   Q4 carries disproportionate mark-to-market activity in PE and RE
   appraisals — a model fit on raw quarters confounds seasonal accounting
   conventions with the smoothing parameter. Four overlapping annual series
   (one per quarter offset) average out the seasonality.
2. **Joint single-step Bayesian estimation with conjugate β-given-λ posterior**
   plus **grid integration over λ on [0.01, 0.95]** (50 points). This avoids
   point-MAP collapse onto the ridge and gives an honest marginal posterior on
   λ. λ is clipped strictly below 1 so the `1/(1−λ)` transform never blows up.
3. **Strategy-specific informative priors on β** (PE_PRESETS, INFRA_PRESETS,
   RE_PRESETS). These are the lever that breaks the ridge: an N(1.15, 0.50)
   prior for US large-cap buyout encodes the leverage-decline economics and
   anchors β when the data alone cannot.
4. **Cross-sectional shrinkage via induced priors** (`induced_priors.py`).
   Estimating peer strategies independently, then using the empirical
   cross-sectional distribution as a hierarchical prior, pools strength across
   strategies when the time-series alone is uninformative.
5. **Explicit, automated identifiability diagnostics** — these turn an invisible
   failure into a visible warning. Per Stage 1 spec point 6:
   - 90 % credible interval on λ wider than 0.4 → "smoothing poorly identified,
     output is prior-driven."
   - KL(posterior‖prior) < 0.1 nats → "data is not informative for λ."
   - β changes sign or magnitude > 50 % across the λ grid → "λ and β are
     confounded; use uncertainty_mode='full'."
   - Effective sample size reported (overlapping annual returns are correlated;
     naïve N overstates information).
6. **`uncertainty_mode='full'`** runs the full Stage 2→3→4 pipeline over 100
   posterior draws of (λ, β, σ_ε) and returns 5/25/50/75/95 percentile bands
   on the disaggregated series. This is the structural defence against false
   precision in downstream VaR / backtest numbers — the user sees the band
   width and cannot mistake a prior-driven point estimate for a data-driven one.
7. **Graceful degradation via `FallbackPolicy`**: if the Bayesian fit produces
   a flat λ posterior, fall back to classic Geltner with λ = 0.5 and emit a
   warning, instead of returning a confidently wrong number.

---

## Risk 2 — MA(q) inversion noise amplification & weight identifiability (Hedge Funds)

### The problem

Getmansky–Lo–Makarov decomposes observed returns as `s_t = Σ θ_j r_{t−j}`,
sum-to-one, non-negative. Recovering `r` requires inverting the banded
Toeplitz operator Θ. Two coupled fragilities:

- **Inversion blow-up.** When θ_0 is small (concentrated illiquid strategies
  like distressed) the matrix Θ is poorly conditioned and the inverse
  amplifies measurement noise in `s`. The recovered `r` series can show
  artefactual oscillations whose magnitude is unrelated to true return
  variability.
- **Weight identifiability.** With ~240 monthly obs (20 yr) and q=4, five θ
  weights must be pinned down by sample autocovariances at lags 0…4. Lag-4
  autocovariance of monthly returns is small in absolute terms and dominated
  by sampling noise; the resulting θ estimates can violate monotonic decay,
  hit the boundary θ_q = 0, or place suspicious mass on lag q. The number of
  lags q is itself a structural choice (BIC vs. fixed) that interacts with
  the prior.

### Mitigations baked into the architecture

1. **Hard floor θ_0 ≥ 0.2** (Stage 1 spec point 4). Caps the inverse's
   condition number by guaranteeing a minimum diagonal weight.
2. **Ordered Dirichlet / stick-breaking prior** enforcing θ_0 ≥ θ_1 ≥ … ≥ θ_q.
   The economic prior is unambiguous (more recent NAVs reflect more
   contemporaneous returns); encoding it as a hard order constraint removes
   a large slice of the parameter space where likelihood is flat.
3. **Exponential-decay prior on higher-lag coefficients** (Stage 1 spec
   point 4). Tames sampling noise that would otherwise place spurious mass
   on lag q.
4. **q chosen per-strategy, not estimated freely**: q=2 for liquid (equity
   L/S, macro), q=3 for relative value, q=4 for event-driven, q=0 for
   managed futures (NoSmoothing). Selecting q from data is itself fragile;
   strategy-fixed q is more robust and more interpretable.
5. **Estimation method selected by dimension**: grid integration over the
   simplex for q ≤ 3 (tractable, gives true marginals); Laplace approximation
   (MAP via `scipy.optimize.minimize` + Hessian) for q > 3 to avoid
   exponential grid blow-up.
6. **Condition-number monitoring + automatic q-reduction fallback**: if
   `cond(Θ) > 100`, drop q by 1 and re-estimate; if q=1 still fails, fall
   back to AR(1). User sees a warning either way.
7. **Stage 0 reporting-lag adjustment is upstream of the MA fit**, separating
   mechanical reporting delay from genuine economic smoothing. Otherwise the
   MA filter would absorb both, biasing θ away from the smoothing-only
   interpretation that downstream sanity anchors expect.

---

## Risk 3 — Temporal disaggregation is structurally under-determined (Stage 3, generic)

### The problem

Chow-Lin and its variants (Fernández, Litterman) recover ~3× as many monthly
values as quarterly observations (or ~21× for monthly→daily). The temporal
aggregation constraint pins down a small share of that information; everything
else comes from (a) the high-frequency indicator path (factor returns) and
(b) the assumed AR(1) / RW residual structure. Three failure modes follow:

- **Residual-dominated disaggregation.** If the factor model's R² is low
  (common for distressed, RV, niche strategies), the indicator does almost no
  work and the disaggregated path is effectively the residual smoothing model
  spread evenly across the period. This produces high-frequency series that
  look plausible per-period but have wrong distributional properties — wrong
  vol term structure, wrong drawdown shapes, wrong cross-strategy correlation.
- **Aggregation-constraint accuracy.** Returns compound, they don't add. The
  additive constraint `Σr_high = r_low` errors by 50+ bps for PE-magnitude
  quarterly returns. Naïve linear-algebra implementations get this wrong by
  default.
- **Cross-strategy correlation collapse.** Running Chow-Lin independently
  per strategy — the obvious thing to do — destroys the cross-sectional
  correlation structure because each strategy's residual draw is independent
  by construction. The downstream VaR / drawdown distribution loses
  diversification properties (or, worse, fakes them).

### Mitigations baked into the architecture

1. **Multiplicative aggregation constraint by default** (Stage 3 spec
   point 2). Work in log-returns inside the linear algebra, exponentiate at
   the boundary. Guarantees `Π(1 + r_high) = (1 + r_low)` to machine
   precision.
2. **Three disaggregation methods**, not one. Chow-Lin (AR(1) residual) is
   the default; Fernández (RW residual, ρ=1) is the explicit fallback when
   Chow-Lin's ρ estimate clips at ±0.99 (a sign that AR(1) is mis-specified);
   Litterman is available for residuals with strong I(1) structure. The
   `FallbackPolicy` automates the switch.
3. **Multivariate / copula-corrected disaggregation** (`disaggregation/
   multivariate.py`) is a first-class component, not an afterthought. The
   default cross-strategy disaggregation preserves the residual covariance
   estimated in Stage 2.
4. **Inter-stage validation gates** (`validation/stage_validation.py`) that
   are not optional:
   - Stage 1→2: desmoothed σ must exceed observed σ; otherwise the λ
     estimate is suspect.
   - Stage 2→3: residuals must be approximately uncorrelated with factors;
     otherwise the factor model is misspecified and disaggregation will lean
     on a broken indicator.
   - Stage 3→4: monthly compounds to quarterly within 1e-10. Round-trip
     consistency is enforced, not assumed.
5. **Sanity anchors against public proxies** (`validation/sanity_anchors.py`)
   on the *output*, not just the intermediates. The four anchors — vol ratio
   ∈ [0.5, 3.0], correlation ∈ [0.4, 0.95], crisis drawdown ≥ 50 % of proxy,
   beta consistent with leverage — exist precisely to catch a residual-
   dominated disaggregation that produces statistically reasonable but
   economically nonsensical paths.
6. **Explicit factor-R² warning at R² < 0.05** (FallbackPolicy item 3): tells
   the user the disaggregated series is residual-dominated *before* they
   build a backtest on top of it.
7. **Epistemic-honesty language in outputs and docstrings** (Critical
   Implementation Note 8). The result object describes itself as "synthetic
   high-frequency returns consistent with observed low-frequency data under
   [model] assumptions," not as "true monthly returns." Mode A (Kalman
   smoother) outputs are conditional expectations, not realisations — the
   API names and docstrings make this distinction visible to the caller.

---

## Honourable mentions (tracked, not in the top 3)

- **Threshold-AR(1) regime mis-specification (credit).** An exogenous
  HY-OAS threshold that is mis-calibrated for the index vintage will bias
  both λ values; the spec's choice to default to exogenous regimes is
  conservative but the threshold is a hyperparameter that needs sanity
  checking. Mitigation: keep the threshold a config parameter, document it,
  and surface regime counts in diagnostics so the user can see when the
  stress regime never (or always) fires.
- **Carry/MTM decomposition's reliance on a clean yield series.** If the
  user supplies a stale or model-implied yield, the carry/MTM split is
  arbitrary and Stage 1 desmooths the wrong signal. Mitigation: validate
  yield-series coverage against return coverage and warn when carry exceeds
  total return for any quarter.
- **Reporting-lag double-counting (HF).** If the data vendor has already
  applied a lag adjustment, applying another in Stage 0 will undo true
  contemporaneous information. Mitigation: keep `reporting_lag_months=0` as
  a documented option and surface autocorrelation diagnostics pre/post
  adjustment so the user can detect double-shifting.
