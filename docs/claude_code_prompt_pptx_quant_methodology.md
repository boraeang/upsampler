# Claude Code Prompt — Quant Methodology Presentation for private_assets_frequency

## Task

Create a PowerPoint presentation (.pptx) explaining the mathematical models implemented in the `private_assets_frequency` library. Read the full library specification at `docs/claude_code_prompt_private_assets_frequency_v2.md` for content.

## Audience

**Quantitative researchers and portfolio analysts** at a hedge fund. Assume the audience:
- Has graduate-level training in econometrics and statistics
- Is comfortable with matrix notation, Bayesian inference, and time series models
- Cares about identifiability, estimation robustness, and where models break
- Will challenge hand-wavy claims — every equation needs economic or statistical motivation
- Does NOT want marketing language or oversimplification

## Tone & Style

- **Technical but clean** — write like a well-structured research seminar, not a textbook
- Equations are first-class citizens: display them prominently, not buried in footnotes
- Use notation consistently throughout (define once, reuse everywhere)
- Every assumption should be stated explicitly and labeled (A1, A2, ...) so they can be referenced and challenged
- Include "Where this breaks" commentary — quants respect honesty about limitations

## Color Palette & Design

Use a dark, professional theme suitable for a quant research desk:

| Role | Color | Hex |
|------|-------|-----|
| Background (title/section slides) | Near-black | `1B1F2A` |
| Background (content slides) | Dark charcoal | `22272E` |
| Primary text | Off-white | `E8E8E8` |
| Accent / equations | Teal | `00B4D8` |
| Secondary accent | Muted gold | `D4A843` |
| Warning / limitations | Soft red | `E07A5F` |
| Positive / result | Sage green | `81B29A` |

Use a monospace or semi-monospace font for equations (Consolas or Courier New). Use Calibri or Calibri Light for body text. Slide titles in bold Calibri at 36pt.

## Slide Structure (20–24 slides)

---

### SECTION 1: Problem Setup (Slides 1–4)

**Slide 1 — Title Slide**
- Title: "Private Assets Frequency Upsampling: Model Architecture & Methodology"
- Subtitle: "Desmoothing, Factor Decomposition & Temporal Disaggregation"
- Footer: "Quantitative Research — Internal Use Only"
- Dark background

**Slide 2 — The Problem We're Solving**
- Left column: What we observe vs. what we need
  - Observe: quarterly NAV returns s_t (smoothed, lagged, low-frequency)
  - Need: monthly/daily true returns r_t (risk-accurate, high-frequency)
- Right column: Why it matters — show the key numbers from MSCI simulation study:
  - Raw returns: vol = 5.3%, β = 0.30, ρ = 57%
  - Desmoothed (Bayesian): vol = 12.6%, β = 1.01, ρ = 81%
  - True: vol = 12.5%, β = 1.00, ρ = 80%
- Takeaway: raw data understates risk by ~60%

**Slide 3 — Pipeline Architecture Overview**
- Visual flow diagram showing the 5 stages:
  - Stage 0: Preprocessing (carry/MTM decomp, reporting lag)
  - Stage 1: Desmoothing (AR(1) / MA(q) / Threshold)
  - Stage 2: Factor Decomposition
  - Stage 3: Temporal Disaggregation (Chow-Lin)
  - Stage 4: Daily Extension (Kalman / Simulation)
- Show which asset classes use which smoothing model
- Use colored boxes matching the palette

**Slide 4 — Notation Convention**
- Define all notation used throughout the deck:
  - s_t = observed (smoothed) return at time t
  - r_t = true (latent) return at time t
  - P_t = reported valuation (NAV)
  - V_t = true (unobservable) valuation
  - λ = smoothing parameter ∈ [0, 1]
  - θ_j = MA weight for lag j (hedge funds)
  - β_k = factor loading on factor k
  - F_{k,t} = return of factor k at time t
  - ε_t = idiosyncratic return
  - σ_ε = idiosyncratic volatility

---

### SECTION 2: AR(1) Bayesian Desmoothing (Slides 5–10)

**Slide 5 — The Smoothing Process (AR(1))**
- Title: "Appraisal Smoothing as AR(1)"
- The valuation dynamics equation:
  - P_t = P_{t-1} + (1 − λ)(V_t − P_{t-1})
- Implied return relationship:
  - s_t = (1 − λ) r_t + λ s_{t-1}
- Inversion (Geltner desmoothing):
  - r_t = (s_t − λ s_{t-1}) / (1 − λ)
- State assumptions explicitly:
  - **A1:** Smoothing follows a first-order autoregressive process on valuations
  - **A2:** λ is constant within each annual window (but may vary seasonally across quarters)
  - **A3:** True returns r_t and smoothing process are independent
- "Where this breaks" callout box in soft red: λ is not constant; appraisals are discretionary; extreme returns may trigger non-linear marking behavior

**Slide 6 — Why Rolling Annual Returns**
- Problem: quarterly smoothing is seasonal (Q4 has more marking activity than Q1–Q3)
- Solution: construct 4 overlapping annual return series, each starting in a different quarter
- Annual desmoothed return: r_annual = (s_t − λ s_{t-4}) / (1 − λ)
- This makes the AR(1) approximation robust even when true smoothing has seasonal or higher-order structure
- Show a small diagram of the 4 overlapping windows

**Slide 7 — Single-Step vs. Two-Step Estimation**
- Two-step: (1) estimate λ → desmooth → (2) regress on factors → estimate β
  - Problem: errors in λ propagate nonlinearly into β; noise is amplified, risk estimates are biased upward
- Single-step: jointly estimate (λ, β, α, σ_ε) in one regression:
  - r_annual = β · F_annual + α + ε
  - where r_annual is itself a function of λ
- Show the MSCI simulation result: single-step RMSE ~0.15 vs two-step RMSE ~0.35
- **A4:** Factor loadings β are constant over the estimation window

**Slide 8 — Bayesian Estimation Framework**
- The posterior we're computing:
  - P(λ, β, α, σ_ε | data, priors) ∝ L(data | λ, β, α, σ_ε) · π(λ) · π(β) · π(α) · π(σ_ε)
- Priors:
  - λ ~ Beta(2, 2) on [0, 1]
  - β ~ N(μ_β, σ²_β) — strategy-specific (e.g., buyout: 1.15 ± 0.5)
  - α ~ N(0, 0.05²)
  - σ_ε ~ InvGamma(3, 0.02)
- Computation strategy:
  - Grid over λ: 50 points on [0.01, 0.95]
  - For each λ: conditional posterior of (β, α, σ_ε) is Normal-Inverse-Gamma (conjugate) → closed form
  - Marginalize over λ via numerical quadrature
- Key benefit: no MCMC needed — exact (up to grid resolution) and deterministic
- **A5:** Conjugate Normal-Inverse-Gamma structure is appropriate for the regression residuals

**Slide 9 — Vasicek Shrinkage & Induced Priors**
- Bayesian estimation as shrinkage:
  - β_Vasicek = w · β_OLS + (1 − w) · β_Prior
  - w depends on signal-to-noise ratio (data informativeness vs prior tightness)
- Induced priors: cross-sectional shrinkage across peer strategies
  - The empirical distribution of estimated parameters across peer strategies acts as an additional prior
  - p̃(x_i; x̂_{j≠i}) acts as prior p(x_i)
  - Iterate: estimate independently → form induced priors → re-estimate → converge
- Show a table of strategy-specific beta priors:
  - Large Buyout: β ~ N(1.15, 0.5²)
  - Early VC: β ~ N(0.83, 0.25²)
  - Mezzanine: β ~ N(1.0, 0.5²)
  - Infra Core: β ~ N(0.5, 0.3²)
- **A6:** Peer group structure is known and stable (strategies are correctly classified)

**Slide 10 — Identifiability & Diagnostics**
- The core identifiability challenge: λ and β are weakly jointly identified
  - Multiple (λ, β) pairs explain the same observed data
  - The likelihood surface is often flat in the λ direction
- Diagnostics implemented:
  - 90% CI width on λ: warn if > 0.4
  - KL divergence between posterior and prior: warn if < 0.1 nats (data uninformative)
  - β stability across λ grid: warn if β changes sign or >50% magnitude shift
  - Effective sample size after accounting for autocorrelation
- "Where this breaks" callout: short histories (< 40 quarters), strategies with very high or very low smoothing, periods where public/private correlation genuinely changes

---

### SECTION 3: MA(q) Desmoothing — Hedge Funds (Slides 11–13)

**Slide 11 — Hedge Fund Smoothing is MA(q), Not AR(1)**
- Economic mechanism: illiquid holdings → stale prices → NAV is a moving average of true returns
- Smoothing equation (Getmansky-Lo-Makarov 2004):
  - s_t = θ_0 r_t + θ_1 r_{t-1} + ... + θ_q r_{t-q}
  - Constraints: θ_j ≥ 0, Σ θ_j = 1
- Contrast with PE: PE smoothing is AR on *valuations* (catch-up); HF smoothing is MA on *returns* (staleness)
- Show typical θ profiles:
  - Equity L/S (q=2): θ ≈ [0.7, 0.2, 0.1]
  - Event-Driven (q=4): θ ≈ [0.4, 0.25, 0.2, 0.1, 0.05]
- **A7:** Reported returns are a weighted sum of concurrent and lagged true returns
- **A8:** Weights are non-negative and sum to one (no over- or under-counting of returns)

**Slide 12 — MA(q) Inversion & Estimation**
- Recovery: in matrix form S = Θ · R → R = Θ⁻¹ · S
  - Θ is a banded lower-triangular Toeplitz matrix
- Identification from autocovariances:
  - Var(s_t) = σ²_r · Σ θ²_j
  - Cov(s_t, s_{t-k}) = σ²_r · Σ θ_j θ_{j+k}   for k ≤ q
- Bayesian extension: ordered Dirichlet prior on θ (θ should be decreasing)
- For q ≤ 3: grid integration over the simplex
- For q > 3: Laplace approximation (MAP + Hessian)
- Numerical stability: clip θ_0 ≥ 0.2; monitor condition number of Θ; warn if cond(Θ) > 100
- **A9:** The number of lags q is known or selected via BIC

**Slide 13 — MA(q) Fragility & Mitigations**
- Core risk: MA(q) is weakly identified in small samples (~100–200 monthly observations)
- Inversion amplifies noise: condition number grows exponentially with q
- Mitigations:
  - Aggressive priors (ordered Dirichlet, exponential decay taper)
  - Cap at q ≤ 3 for grid integration; Laplace approximation for q > 3
  - Fallback: if cond(Θ) > 100, reduce q by 1; if q=1 still fails, fall back to AR(1)
- Show a diagram: estimation RMSE vs q for different sample sizes — RMSE explodes for q > 3 with T < 150

---

### SECTION 4: Threshold AR(1) — Private Credit (Slides 14–15)

**Slide 14 — Regime-Dependent Smoothing for Credit**
- Credit returns = carry + mark-to-market (MTM)
  - Carry bypasses desmoothing (it's not smoothed — it genuinely accrues steadily)
  - Only the MTM component is desmoothed
- Threshold AR(1):
  - s_t = (1 − λ(regime_t)) r_t + λ(regime_t) s_{t-1}
  - Normal regime (no credit events): λ_high ∈ [0.5, 0.9] — loans at par, minimal marking
  - Stress regime (writedowns): λ_low ∈ [0.0, 0.3] — forced marking
- Regime identification: exogenous indicator (HY OAS > threshold = stress)
- **A10:** Returns are decomposable into deterministic carry and stochastic MTM
- **A11:** Regime transitions are determined by an observable indicator (not endogenous)

**Slide 15 — Threshold AR(1) Estimation & Limitations**
- Bayesian estimation: 2D grid over (λ_high, λ_low), conjugate conditionals for β
- Priors: λ_high ~ Beta(5, 2), λ_low ~ Beta(2, 5)
- Limitations:
  - Real regimes are not binary — stress is a continuum
  - Exogenous threshold introduces look-ahead bias if calibrated in-sample
  - Carry/MTM decomposition is approximate (yield is not purely deterministic)
- Fallback: if regime indicator unavailable, degrade to standard AR(1) on total return

---

### SECTION 5: Factor Decomposition (Slide 16)

**Slide 16 — Systematic + Idiosyncratic Decomposition**
- Post-desmoothing factor regression:
  - r_t = Σ_k β_k F_{k,t} + α + ε_t
- Residual diagnostics:
  - Ljung-Box (autocorrelation), ARCH-LM (heteroskedasticity), Jarque-Bera (normality)
  - Distribution fitting: Normal, Student-t, Skewed-t (Hansen 1994)
- Cross-strategy residual covariance matrix Σ_ε for joint simulation
- The decomposition determines what happens in disaggregation:
  - Systematic component → deterministic at monthly/daily via factor returns × β
  - Idiosyncratic component → distributed via Chow-Lin GLS or simulated
- **A12:** Factor loadings are stable over the estimation window
- **A13:** Residuals are uncorrelated with factors (otherwise model is misspecified)

---

### SECTION 6: Temporal Disaggregation (Slides 17–19)

**Slide 17 — Chow-Lin Framework**
- Problem: given quarterly series y_Q and monthly indicators X_m, find y_m such that:
  - (1) y_m is related to X_m via regression with autocorrelated residuals
  - (2) Temporal aggregation constraint: Σ or Π of monthly = quarterly
- The GLS solution:
  - ŷ_m = X_m β̂ + V C'(C V C')⁻¹ (y_Q − C X_m β̂)
  - C = temporal aggregation matrix
  - V = Cov(u_m) parameterized by AR(1) coefficient ρ
- Three variants:
  - Chow-Lin: AR(1) residual, estimate ρ
  - Fernández: ρ = 1 (random walk) — no ρ estimation
  - Litterman: AR(1) on first differences
- **A14:** Monthly residuals follow a stationary (or unit root) AR(1) process
- **A15:** The relationship between y_m and X_m is linear

**Slide 18 — Multiplicative vs Additive Aggregation**
- Additive: r_m1 + r_m2 + r_m3 = r_Q (linear approximation)
- Multiplicative: (1+r_m1)(1+r_m2)(1+r_m3) = (1+r_Q) (exact for returns)
- For PE-magnitude returns (±20% in crisis), the difference is material (50+ bps)
- Implementation: work in log-returns for Chow-Lin linear algebra → exponentiate
- Show a numerical example: Q4 2008 with r_Q = −22%
  - Additive allocation: −7.3%, −7.3%, −7.3%
  - Multiplicative allocation: −7.9%, −7.9%, −7.9% (compounding matters)
- Default: multiplicative

**Slide 19 — Chow-Lin: Why It Works Here**
- Common critique: "Chow-Lin is a macro tool, not a finance tool"
- Rebuttal: with a factor model providing indicators, the linearity assumption (A15) is satisfied by construction
  - PE return = β · F + ε is linear by design
  - Chow-Lin distributes the residual ε — for an AR(1) residual, this is appropriate
- Where it genuinely limits: nonlinear factor exposures (e.g., option-like payoffs in VC)
- Alternative for Stage 4 (daily): Kalman smoother handles the 21:1 ratio better than Chow-Lin

---

### SECTION 7: Daily Extension & Simulation (Slides 20–21)

**Slide 20 — Daily via Kalman Smoother (Mode A: Backtesting)**
- State-space model:
  - State transition: ε_d = φ ε_{d-1} + η_d
  - Observation: ε_monthly = Σ_{d∈month} ε_d
- Kalman smoother produces E[ε_d | all monthly observations]
- Total daily return: r_d = Σ_k β_k F_{k,d} + ε̂_d
- The daily path is a conditional expectation — not a realization
- **A16:** Daily idiosyncratic returns follow a stationary AR(1) process
- **A17:** Systematic daily returns are fully determined by daily factor returns × quarterly betas

**Slide 21 — Monte Carlo Simulation (Mode B: VaR / Risk)**
- For each simulation path:
  1. Draw (λ, β, σ_ε) from posterior (uncertainty propagation)
  2. Generate daily systematic: r_sys,d = β · F_d
  3. Draw daily idiosyncratic: ε_d ~ fitted distribution (Normal, t, skewed-t) with correct variance and correlation structure
  4. Apply compounding constraint within each month
- Multi-strategy: draw from joint distribution using cross-strategy residual correlation Σ_ε
- Output: N paths × T days × K strategies → VaR, CVaR, drawdown distributions
- Epistemic honesty: these are model-implied scenarios, not recovered historical paths

---

### SECTION 8: Summary & Assumptions Register (Slides 22–24)

**Slide 22 — Model Selection by Asset Class**
- Summary table:

| Asset Class | Smoothing Model | Key Factors | Typical λ / θ_0 | Special Handling |
|---|---|---|---|---|
| PE (Buyout) | AR(1) Bayesian | Equity, Size | λ ∈ [0.4, 0.7] | Rolling annual returns |
| PE (Venture) | AR(1) Bayesian | Small-cap growth | λ ∈ [0.5, 0.8] | Tighter β prior |
| Infrastructure | AR(1) Bayesian | Utilities, Inflation, Duration | λ ∈ [0.5, 0.8] | Duration + inflation factors |
| Private Credit | Threshold AR(1) | Credit spreads, Rates | λ_high/λ_low | Carry/MTM decomposition |
| Hedge Fund (Eq L/S) | MA(q=2) | Equity, SMB, MOM | θ_0 ∈ [0.6, 0.8] | Reporting lag adjustment |
| Hedge Fund (Event) | MA(q=4) | Equity, Credit | θ_0 ∈ [0.3, 0.5] | Higher illiquidity |
| Managed Futures | None | Trend factors | N/A | No smoothing |
| Real Estate | AR(1) Bayesian | REITs, Rates | λ ∈ [0.6, 0.85] | Strong Q4 seasonality |

**Slide 23 — Full Assumptions Register**
- List all numbered assumptions A1–A17 in a clean table with three columns:
  - Assumption | Where Used | Severity if Violated
- Color-code severity: green (minor impact), gold (moderate), red (critical)
- This is the slide the audience will photograph and take back to their desks

**Slide 24 — Known Limitations & Open Questions**
- Identifiability: λ and β are weakly jointly identified — outputs are prior-sensitive for short histories
- Model risk: the pipeline generates model-implied series, not recovered truth
- Factor stability: β estimated over full sample; time-varying exposures are not captured
- Tail behavior: Chow-Lin distributes residuals linearly; extreme month allocation within a crisis quarter is model-dependent
- Survivorship bias: PE/HF indices contain survivorship and backfill bias that desmoothing does not correct
- Open questions:
  - Should we move to a unified state-space formulation? (Trade-off: elegance vs modularity/debuggability)
  - Can we use fund-level data to estimate λ heterogeneity within a strategy?
  - Time-varying β via rolling-window or DCC?
- End with: "All models are wrong. These are less wrong than the alternatives — and we know exactly where they're wrong."

---

## Equation Rendering

Since PowerPoint cannot render LaTeX natively, render all equations as **formatted text using Consolas/Courier New font** with the following conventions:
- Use Unicode characters for Greek letters: λ (U+03BB), β (U+03B2), α (U+03B1), ε (U+03B5), θ (U+03B8), σ (U+03C3), ρ (U+03C1), π (U+03C0), φ (U+03C6), η (U+03B7), μ (U+03BC), Σ (U+03A3), Π (U+03A0)
- Use subscript formatting where pptxgenjs supports it, otherwise use underscore notation: s_t, r_{t-1}, β_k
- Display equations on their own line, centered, in teal (#00B4D8) on dark background, at 20–24pt
- Inline equation references in body text at 14–16pt in the same teal

For complex equations that look bad as text (fractions, integrals), render them as **SVG images** using a tool like MathJax or matplotlib's mathtext renderer, and embed the SVG/PNG in the slide. Specifically:
- Use matplotlib with transparent background and white text to render key equations as PNG
- Save to a temp directory and embed in the slide
- This applies to: the desmoothing fraction r_t = (s_t − λs_{t-1})/(1−λ), the Bayesian posterior integral, and the Chow-Lin GLS formula

## Design Notes

- **Do not use bullet points for equations** — display each equation in its own styled text box or as an embedded image
- Assumption labels (A1, A2, ...) should appear in small colored tags (gold background, dark text) next to the relevant equation or statement
- "Where this breaks" callout boxes should use the soft red (#E07A5F) as a left border accent with dark background
- Section divider slides (between major sections) should use the near-black background with a single large section title in teal
- Leave generous whitespace — quants will stare at these equations; don't crowd them
- No clip art, no stock photos, no decorative elements. Only diagrams, equations, and tables.
- Speaker notes are NOT needed — the slides should be self-contained

## Output

Save the final presentation to the project docs folder alongside the library spec:
`docs/private_assets_frequency_methodology.pptx`
