# Claude Code Prompt — Rudin-Mao-Zhang-Fink Reparameterized Unsmoothing Model

## Task

Implement the private-equity return unsmoothing model from **Rudin, Mao, Zhang & Fink (2019), "Fitting Private Equity into the Total Portfolio Framework," *Journal of Portfolio Management***, specifically the model described in their **"Risk Estimation Through Unsmoothing"** section (Equations 1–4).

The deliverable is a **new pluggable smoothing model** that slots into the existing `private_assets_frequency` library architecture as an implementation of the `SmoothingModel` protocol. Name the class `RudinReparamSmoothing` and register it under the model key `'rudin_reparam'`.

Read the existing library specification at `docs/claude_code_prompt_private_assets_frequency_v2.md` to understand the `SmoothingModel` protocol, the `DesmoothedResult` return type, the config classes, and the registry pattern this must conform to. The new model must be a drop-in alternative to the existing `ar1_bayesian`, `ma_glm`, and `threshold_ar1` models.

**Scope boundaries — implement ONLY the core econometric model:**
- IN SCOPE: the single-regression estimation of (α, β, θ), reconstruction of the unsmoothed true-return series, risk-property computation, selectable lag order Q, and the variance-ratio diagnostic for choosing Q.
- OUT OF SCOPE: the mini-program simulation machinery, the shrinkage-across-paths approach, the diversification study, and the total-portfolio construction framework. Do not implement any of these.

---

## The Model

### Notation

- `r̃ᵒ_t` — observed **smooth** return of the private equity asset at time t (this is the input data)
- `rᴱ_t` — unobserved **true economic** return at time t (this is what we reconstruct)
- `F^i_t` — return of systematic factor i at time t (i = 1..N; general multi-factor)
- `α` — intercept (alpha)
- `β_i` — loading on factor i
- `ε_t` — idiosyncratic (residual) return
- `Q` — maximum number of lags (selectable, 0 to 4)
- `w_j`, `θ_j` — two equivalent parameterizations of the smoothing weights

### Equation 1 — The smoothing relationship (Conner 2003; Pedersen, Page & He 2014)

The observed smooth return is a weighted average of current and past **true** returns:

```
r̃ᵒ_t = Σ_{j=0}^{Q} w_j · rᴱ_{t−j}          (Eq. 1)
```

where the weights `w_j` sum to one: `Σ_{j=0}^{Q} w_j = 1`.

### Equation 2 — The factor model for true returns

The true economic return follows a linear factor model:

```
rᴱ_t = α + Σ_{i=1}^{N} β_i · F^i_t + ε_t          (Eq. 2)
```

### Equation 3 — The reparameterization (KEY CONTRIBUTION)

Rather than expressing observed returns as a function of true returns (Eq. 1), Rudin et al. flip the relationship to express the **true** return as a function of the **observed** return and its lags (following Stefek & Suryanarayanan 2012):

```
rᴱ_t = θ_0 · r̃ᵒ_t + Σ_{j=1}^{Q} θ_j · r̃ᵒ_{t−j}          (Eq. 3)
```

**Important equivalence property:** Equation 3 is equivalent to Equation 1 — knowing the `θ_j`, one can compute the `w_j` and vice versa. The `θ_j` also satisfy a sum constraint:

```
Σ_{j=0}^{Q} θ_j = 1
```

This is the crux: **once θ is known, the true return is a simple linear combination of observed returns**, so reconstruction is trivial (no matrix inversion, no filtering).

### Equation 4 — The single estimating regression

Substituting Equation 2 into Equation 3 yields a single regression with a **non-autocorrelated** error term:

```
r̃ᵒ_t = α + (1 − Σ_{j=1}^{Q} θ_j) · [ Σ_{i=1}^{N} β_i · F^i_t ]  +  Σ_{j=1}^{Q} θ_j · r̃ᵒ_{t−j}  +  ε_t          (Eq. 4)
```

**Derivation note (verify during implementation):** Starting from Eq. 3, solve for `θ_0 · r̃ᵒ_t` and use `θ_0 = 1 − Σ_{j≥1} θ_j`:
- From Eq. 3: `θ_0 · r̃ᵒ_t = rᴱ_t − Σ_{j≥1} θ_j · r̃ᵒ_{t−j}`
- Substitute Eq. 2 for `rᴱ_t`: `θ_0 · r̃ᵒ_t = α + Σ_i β_i F^i_t + ε_t − Σ_{j≥1} θ_j r̃ᵒ_{t−j}`
- Divide by θ_0 and rearrange into regression form. The published Eq. 4 groups the factor term with the coefficient `(1 − Σθ_j) = θ_0`, giving the form above.

**Estimation:** Equation 4 is linear in the parameters `{α, β_1..β_N, θ_1..θ_Q}` and can be estimated by **ordinary least squares** in a single regression. Note the following structural features the implementation must respect:

1. The regressors are: a constant (for α), the N contemporaneous factor returns, and the Q lagged observed returns `r̃ᵒ_{t−1} .. r̃ᵒ_{t−Q}`.
2. **Nonlinearity caveat:** The coefficient on each factor `F^i_t` is `θ_0 · β_i = (1 − Σ_{j≥1} θ_j) · β_i`, i.e. the product of the factor-block scalar and β_i. The coefficients on the lagged observed returns are the `θ_j` directly. Therefore the raw OLS coefficients are:
   - lag coefficients: `θ_j` (j = 1..Q) — read directly
   - factor coefficients: `c_i = θ_0 · β_i` — must be **unwound** to recover β_i
   - After estimating θ_1..θ_Q from the lag coefficients, compute `θ_0 = 1 − Σ_{j≥1} θ_j`, then recover `β_i = c_i / θ_0`.
   - The intercept coefficient is `α` (read directly, but see note below on whether it needs scaling — verify: in Eq. 4 the intercept term is α with coefficient 1, so read directly).

   Because the factor coefficients enter as a product `θ_0 · β_i`, the model is technically nonlinear if one insists on jointly optimal (θ, β). However, the paper treats Eq. 4 as a **straightforward linear regression** — the regressors (constant, factors, lagged observed returns) are all observed, and OLS gives consistent estimates of the composite coefficients, from which θ and β are algebraically recovered as above. Implement it this way (linear OLS + algebraic unwinding), matching the paper. Do NOT use nonlinear least squares.

3. For **Q = 0** (no unsmoothing): there are no lag terms, θ_0 = 1, and Eq. 4 reduces to a plain factor regression `r̃ᵒ_t = α + Σ β_i F^i_t + ε_t`. The reconstructed "true" return equals the observed return. This is the baseline case.

### Reconstruction of the unsmoothed true-return series

Once `θ_0 .. θ_Q` are estimated, reconstruct the true economic return directly from Equation 3:

```
r̂ᴱ_t = θ_0 · r̃ᵒ_t + Σ_{j=1}^{Q} θ_j · r̃ᵒ_{t−j}
```

This is a simple weighted sum of observed returns — no inversion required. The first Q observations will be lost (no lags available); document this edge handling. Return the reconstructed series aligned to the valid time index.

---

## Risk-Property Computation

After reconstruction, compute and return the following risk properties of the unsmoothed series (these are the quantities reported in the paper's Exhibit 1):

1. **Alpha (α):** the regression intercept, annualized. Report both per-period and annualized. (Paper reports annualized %, e.g. 4.64% for Q=1.)
2. **Factor betas (β_i):** the recovered loadings, one per factor. (Paper reports 0.53 beta to S&P 500 for Q=1.)
3. **Volatility:** the annualized standard deviation of the reconstructed true-return series `r̂ᴱ_t`. (Paper reports 13.3% for Q=1 vs 9.5% for Q=0.)
4. **% Variance Explained (in-sample):** the R² of the factor model on the reconstructed returns — i.e. the fraction of the variance of `r̂ᴱ_t` attributable to the systematic factor component `Σ β_i F^i_t`. (Paper reports 46% for Q=1.)
5. **Idiosyncratic (residual) volatility:** annualized std of the residual `ε_t`, i.e. the complement to the systematic risk.

**Annualization:** infer periods-per-year from the data frequency (quarterly → 4, monthly → 12). Volatility annualizes as `σ · √(periods_per_year)`; alpha annualizes as `α · periods_per_year` (simple) — but make the alpha annualization convention configurable (simple vs geometric) and default to simple to match the paper's apparent convention. Document the choice.

---

## Lag Selection & Variance-Ratio Diagnostic

Implement selectable `Q ∈ {0, 1, 2, 3, 4}` and a diagnostic to compare models across Q.

### The paper's diagnostic (adapted to standalone scope)

In the paper, parameters (θ, β) are estimated on a broad **index**, then applied out-of-sample to simulated **mini-program paths**, measuring:

```
variance_ratio = Var(returns unexplained by the model) / Var(total mini-program returns)
```

A ratio **> 1** means the model *introduces* risk rather than explaining it (overfitting). The paper finds Q > 2 is counterproductive and **Q = 1 is optimal**.

**Since the mini-program simulation is out of scope,** implement the diagnostic in two modes:

1. **In-sample mode (default):** Fit Eq. 4 on the provided series for each Q from 0 to 4. Compute the variance ratio as `Var(ε_t) / Var(r̂ᴱ_t)` — residual variance over total reconstructed-return variance. Report the table across Q. Lower is better; note this in-sample version will mechanically favor higher Q (more regressors), so warn the user that in-sample variance ratio is a weak selector and out-of-sample is preferred.

2. **Out-of-sample mode (if the user supplies a train/test split or a second return series):** Estimate (θ, β, α) on the training series (or a supplied "index" series), then apply those fixed parameters to the test series: reconstruct true returns on the test data using the trained θ, compute the systematic prediction using trained β on the test-period factors, and measure `variance_ratio = Var(test_actual − model_prediction) / Var(test_actual)`. This reproduces the paper's methodology faithfully. A ratio > 1 flags overfitting for that Q. Select the Q minimizing the out-of-sample ratio.

Provide a method `select_lag_order(...)` that returns the diagnostic table (one row per Q with: α, β_i, volatility, % explained, variance_ratio) and the recommended Q. Default recommendation logic: pick the Q with the lowest out-of-sample variance ratio if available; otherwise apply a parsimony rule that mirrors the paper's finding (prefer the smallest Q whose in-sample % explained is within a small tolerance of the best, defaulting to Q=1 when ambiguous). Make the tolerance configurable.

---

## Integration with the Existing Library

### Conform to the `SmoothingModel` protocol

Implement the class to satisfy the existing protocol (defined in `core/protocols.py`):

```python
class RudinReparamSmoothing:
    """
    Reparameterized unsmoothing model of Rudin, Mao, Zhang & Fink (2019).

    Estimates smoothing weights θ and factor betas β simultaneously via a
    single OLS regression (their Eq. 4), then reconstructs the true economic
    return series directly from observed returns (their Eq. 3).

    References
    ----------
    Rudin, A., Mao, J., Zhang, N. R., & Fink, A.-M. (2019). Fitting Private
    Equity into the Total Portfolio Framework. The Journal of Portfolio
    Management, 46(2), 60-77.
    Stefek, D., & Suryanarayanan, R. (2012). [reparameterization approach for
    real estate].
    Pedersen, N., Page, S., & He, F. (2014). Asset Allocation: Risk Models for
    Alternative Investments. Financial Analysts Journal.
    Conner, A. (2003). [unsmoothing methodology].
    """

    def __init__(self, n_lags: int = 1, alpha_annualization: str = 'simple'):
        ...

    def fit(self, observed_returns, factor_returns, priors=None) -> DesmoothedResult:
        """
        Estimate (α, β, θ) via the single OLS regression of Eq. 4.

        Note: `priors` is accepted for protocol compatibility but is ignored —
        this is a frequentist OLS estimator with no priors. If priors are
        supplied, emit an informational note that they are not used by this model.
        """
        ...

    def log_likelihood(self, params, observed_returns, factor_returns) -> float:
        """Gaussian log-likelihood of the OLS residuals (for model comparison)."""
        ...

    def log_prior(self, params, prior_config) -> float:
        """Returns 0.0 — this model uses no priors. Present for protocol compliance."""
        ...

    def desmooth(self, observed_returns, smoothing_params) -> np.ndarray:
        """Reconstruct true returns via Eq. 3 given known θ. No estimation."""
        ...
```

### `DesmoothedResult` contents

Populate the existing `DesmoothedResult` NamedTuple:
- `true_returns`: reconstructed `r̂ᴱ_t` series (pandas Series, aligned index, first Q obs dropped)
- `smoothing_params`: dict with `theta` (array θ_0..θ_Q), `w` (equivalent w_j weights), `beta` (array), `alpha`, `n_lags`
- `posterior_summary`: for this frequentist model, report OLS point estimates plus standard errors and t-stats for α, β_i, and θ_j (from the regression covariance matrix). Label clearly that these are OLS/frequentist, not Bayesian posteriors.
- `diagnostics`: dict with volatility, pct_variance_explained, idiosyncratic_vol, residual autocorrelation test (Ljung-Box — should be insignificant if the reparameterization worked, since the error term is claimed non-autocorrelated), Durbin-Watson statistic, and the regression R².

### θ ↔ w conversion utility

Implement the bidirectional conversion between the θ parameterization (Eq. 3) and the w parameterization (Eq. 1), since the paper stresses their equivalence and users may want the w_j weights (which have the direct interpretation of "how observed return is built from true returns"). Derive the mapping carefully:
- Eq. 1: `r̃ᵒ_t = Σ_j w_j rᴱ_{t−j}` (observed as MA of true)
- Eq. 3: `rᴱ_t = Σ_j θ_j r̃ᵒ_{t−j}` (true as function of observed)
- These are inverse relationships in lag-operator terms: if `r̃ᵒ = W(L) rᴱ` then `rᴱ = W(L)⁻¹ r̃ᵒ = Θ(L) r̃ᵒ`, so Θ(L) = W(L)⁻¹. Compute w from θ by polynomial inversion (and vice versa), truncated/normalized appropriately. Document that the finite-Q θ implies an infinite-order w in general; provide the exact conversion for the reported cases and warn when truncation is applied. Verify the constraint Σθ_j = 1 corresponds to Σw_j = 1.

### Registry

Register the model in the existing registry (`core/registry.py` or `pipeline/presets.py`) under key `'rudin_reparam'` so it can be selected via `AssetClassConfig(smoothing_model='rudin_reparam', ...)`.

---

## Library Placement

Add these files to the existing structure (do not restructure the library):

```
private_assets_frequency/
├── desmoothing/
│   ├── rudin_reparam.py          # NEW: RudinReparamSmoothing class
│   └── theta_w_conversion.py     # NEW: θ ↔ w bidirectional conversion utilities
└── tests/
    └── test_rudin_reparam.py     # NEW: unit + synthetic-recovery tests
```

---

## Testing Requirements

1. **Synthetic recovery test.** Generate synthetic data with known θ and β:
   - Simulate true returns `rᴱ_t = α + β·F_t + ε_t` with known α, β, σ_ε.
   - Apply known smoothing weights w_j to produce observed returns via Eq. 1.
   - Fit `RudinReparamSmoothing` and assert it recovers θ (and the implied β, α) within tolerance.
   - Assert the reconstructed `r̂ᴱ_t` correlates > 0.95 with the true `rᴱ_t`.

2. **Q=0 identity test.** With Q=0, assert reconstructed returns exactly equal observed returns and the model reduces to a plain factor regression (β, α match a direct OLS of observed on factors).

3. **Non-autocorrelation test.** After fitting on data generated by the model, assert the residuals ε_t show no significant autocorrelation (Ljung-Box p > 0.05, Durbin-Watson ≈ 2). This validates the paper's central claim that Eq. 4 has a non-autocorrelated error term (unlike the Pedersen-Page-He two-step form).

4. **θ ↔ w conversion test.** Round-trip: convert θ → w → θ and assert recovery within tolerance. Assert Σθ = 1 ⟺ Σw = 1.

5. **Volatility inflation test.** Assert that unsmoothed volatility ≥ observed volatility for Q ≥ 1 (unsmoothing reveals hidden volatility — matches Exhibit 1 where vol rises from 9.5% to 13.3%+ as Q increases).

6. **Variance-ratio diagnostic test.** On data with a known optimal lag structure, assert `select_lag_order` returns a sensible table and that out-of-sample variance ratio exceeds 1 for over-parameterized Q (reproducing the paper's overfitting finding).

7. **Multi-factor test.** Verify correct estimation and β recovery with N > 1 factors.

8. **Exhibit 1 sanity check (optional, documented).** If the user has SSPE-like data, the model applied to buyouts with the S&P 500 factor should produce results in the neighborhood of Exhibit 1 (Q=1: α≈4.6%, β≈0.53, vol≈13.3%, %explained≈46%). Add this as a documented reference, not an automated assertion (data-dependent).

---

## Critical Implementation Notes

1. **The factor-coefficient unwinding is the subtle part.** OLS on Eq. 4 gives you composite coefficients `θ_0·β_i` on the factors and `θ_j` on the lagged returns. You must estimate θ_0 = 1 − Σθ_j from the lag coefficients first, then divide the factor coefficients by θ_0 to recover β_i. Guard against θ_0 near zero (would blow up β). If θ_0 < 0.1, warn that unsmoothing is extreme and β estimates are unstable.

2. **Match the paper: linear OLS, not NLS.** Even though (θ_0·β_i) is a product, the paper estimates Eq. 4 as a linear regression on observed regressors and recovers structural parameters algebraically. Do not "improve" this with nonlinear least squares — fidelity to the published method matters here.

3. **Lag alignment and lost observations.** Building the lagged regressors `r̃ᵒ_{t−1..t−Q}` drops the first Q observations. Be explicit and consistent about index alignment between the dependent variable, the factor matrix, and the lag matrix. Document that reconstruction also loses the first Q points.

4. **Sum constraint is a property, not an imposed restriction.** The paper does NOT constrain Σθ = 1 during estimation — it's a consequence of the equivalence with Eq. 1. Estimate θ freely via OLS, then report the realized Σθ and how close it is to 1 as a diagnostic (large deviation suggests model misspecification). Do not force the constraint unless the user explicitly requests a constrained variant.

5. **Standard errors.** Because the error term is claimed non-autocorrelated, plain OLS standard errors are appropriate (no Newey-West needed — that's the whole point of the reparameterization). Report heteroskedasticity-robust (HC) standard errors as an option, but default to classical OLS SEs to match the paper's framing. Optionally, verify residual non-autocorrelation and warn if it's violated (which would indicate Newey-West is actually still needed).

6. **Frequency-agnostic.** The model works on quarterly or monthly data. Infer periods-per-year for annualization from the index frequency.

7. **Dependencies:** numpy, scipy, statsmodels, pandas only — consistent with the parent library's constraint (no additional dependencies).

---

## Build Order

1. `desmoothing/theta_w_conversion.py` — θ ↔ w conversion + tests
2. `desmoothing/rudin_reparam.py` — the `RudinReparamSmoothing` class:
   a. `fit` (build regressors, OLS, unwind β, assemble result)
   b. `desmooth` (Eq. 3 reconstruction)
   c. risk-property computation
   d. `select_lag_order` (variance-ratio diagnostic, both modes)
   e. `log_likelihood` / `log_prior` (protocol compliance)
3. Registry registration under `'rudin_reparam'`
4. `tests/test_rudin_reparam.py` — all tests above
5. Run the full test suite and fix failures.

At the end, produce a short `docs/rudin_reparam_notes.md` documenting: the equations as implemented, the factor-coefficient unwinding step, the two diagnostic modes, and any deviations from the paper (there should be none in the core model — only the mini-program simulation is omitted by scope).
