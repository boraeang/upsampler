# Claude Code Prompt — Okunev & White (2003) Generalized Unsmoothing Model

## Task

Implement the return-unsmoothing procedure of **Okunev & White (2003), "Hedge Fund Risk Factors and Value at Risk of Credit Trading Strategies"** (Section III.A, "Adjusting Reported Returns to Remove Autocorrelation"). The paper is in the project documents (SSRN id 460641).

The deliverable is a **new pluggable smoothing model** that slots into the existing `private_assets_frequency` library as an implementation of the `SmoothingModel` protocol. Name the class `OkunevWhiteSmoothing` and register it under the model key `'okunev_white'`.

Read the existing library specification at `docs/claude_code_prompt_private_assets_frequency_v2.md` to conform to the `SmoothingModel` protocol, the `DesmoothedResult` return type, the config classes, and the registry pattern. This must be a drop-in alternative to the existing `ar1_bayesian`, `ma_glm`, `threshold_ar1`, and `rudin_reparam` models. The existing code is at https://github.com/boraeang/upsampler/tree/main/private_assets_frequency — match its conventions, typing style, and test patterns.

**Scope — implement ONLY the core unsmoothing procedure:**
- IN SCOPE: the iterative autocorrelation-removal algorithm (remove up to m orders of autocorrelation), reconstruction of the true-return series, and the associated risk-property computation (variance inflation, autocorrelation before/after).
- OUT OF SCOPE: the factor-mapping regressions, the Value-at-Risk analysis, the nonlinear factor exposures, and everything in Sections IV–V of the paper. Do not implement these.

**Important conceptual note:** Unlike the MSCI single-step model or the Rudin reparameterized model, the Okunev-White procedure is a **pure time-series method** — it uses ONLY the return series' own autocorrelation structure and does NOT use factor returns at all. It is a generalization of Geltner that removes autocorrelation up to an arbitrary lag m, rather than only lag 1. The `fit` method must therefore work with no factor input. If factor returns are passed in (for protocol compatibility), ignore them and emit an informational note that this model does not use factors.

---

## The Model

### Notation (following the paper)

- `r_{0,t}` — the observed (reported) return at time t, with **0** adjustments applied. This is the input data.
- `r_{m,t}` — the true underlying return at time t, obtained after **m** adjustments to reported returns. This is the output.
- `a_{m,n}` — the **n-th order autocorrelation** of the return series *after m adjustments have been made*. (Double subscript: first index = number of adjustments, second index = lag.)
- `c_m` — the adjustment parameter applied at the m-th adjustment step, chosen to zero out (or set to a target) the relevant-order autocorrelation.
- `d_n` — the *desired* target level for the n-th order autocorrelation (default 0 = remove it entirely; the method also supports leaving a nonzero target).

### The smoothing assumption (Eq. 3 in the paper)

Okunev-White assume the fund manager smooths returns as a linear combination of the true current return and lagged **reported** returns:

```
r_{0,t} = (1 − α) · r_{m,t} + Σ_i β_i · r_{0,t−i}          (Eq. 3)
where (1 − α) = Σ_i β_i
```

The goal is to invert this — recover `r_{m,t}` — **without assuming a specific time-series process** for the underlying returns. The only assumption is that the observed autocorrelation is entirely an artifact of smoothing.

### Step 1 — Remove first-order autocorrelation (Eq. 4)

The first adjustment generalizes Geltner. Define the once-adjusted series:

```
r_{1,t} = ( r_{0,t} − c_1 · r_{0,t−1} ) / ( 1 − c_1 )          (Eq. 4)
```

The parameter `c_1` is chosen so that the first-order autocorrelation of `r_{1,t}` equals the desired target `d_1` (default 0). The new first-order autocorrelation as a function of c_1 is:

```
a_{1,1} = [ a_{0,1} − c_1(1 + a_{0,2}) + c_1² a_{0,1} ] / [ 1 + c_1² − 2 c_1 a_{0,1} ]          (Eq. 5)
```

Setting `a_{1,1} = d_1` and solving the resulting quadratic in c_1 gives (Eq. 6):

```
c_1 = { (2 a_{0,1} d_1 − a_{0,2} d_1 + d_1 − ... )   — SOLVE THE QUADRATIC NUMERICALLY }
```

**Implementation guidance for c_1:** Rather than hard-coding the messy closed form of Eq. 6 (which is error-prone to transcribe), form the quadratic in c_1 from the condition `a_{1,1}(c_1) = d_1` using Eq. 5, i.e.:

```
a_{0,1} − c_1(1 + a_{0,2}) + c_1² a_{0,1}  =  d_1 · (1 + c_1² − 2 c_1 a_{0,1})
```

Rearrange into standard form `A·c_1² + B·c_1 + C = 0` with:
```
A = a_{0,1} − d_1
B = −(1 + a_{0,2}) + 2 d_1 a_{0,1}
C = a_{0,1} − d_1
```
Solve with the quadratic formula and **select the root** with the smaller absolute value (the paper's solutions correspond to the economically sensible, smaller-magnitude adjustment; verify by checking which root actually drives the realized autocorrelation to d_1 and yields a stable, invertible adjustment). Guard the discriminant: a real root requires the condition in Eq. 7:
```
(a_{0,1} − d_1)²  ≤  (1 + a_{0,2} − 2 d_1 a_{0,1})² / 4
```
If no real root exists, raise a clear error (or trigger the fallback policy — see Integration section).

**AR(1) sanity check:** the paper notes that if the true process is genuinely AR(1) and we remove first-order autocorrelation completely (d_1 = 0), then `c_1 = a_{0,1}` (or its reciprocal). Use this as a unit test.

### Step 2 — Remove second-order autocorrelation (Eq. 12), then iterate

To remove the second-order autocorrelation, apply the SAME form of adjustment but at lag 2, to the already-once-adjusted series `r_{1,t}`:

```
r_{2,t} = ( r_{1,t} − c_2 · r_{1,t−2} ) / ( 1 − c_2 )          (Eq. 12)
```

`c_2` is chosen to zero the second-order autocorrelation of the once-adjusted series. By the same quadratic construction as Step 1 (Eqs. 13–14), but using the autocorrelations `a_{1,2}` and `a_{1,4}` of the once-adjusted series:

```
Form the quadratic from:  a_{2,2}(c_2) = d_2   using Eq. 13, analogous to Eq. 5.
Real-root condition:  (a_{1,2} − d_2)²  ≤  (1 + a_{1,4} − 2 d_2 a_{1,2})² / 4          (Eq. 15)
```

**Critical subtlety — removing lag 2 reintroduces lag 1.** After applying the lag-2 adjustment, the first-order autocorrelation of `r_{2,t}` is no longer exactly zero (Eq. 19). The paper's remedy (Section III.A.2, Eqs. 20–22) is to then remove first-order autocorrelation again on `r_{2,t}` (a small adjustment, since `a_{2,1}` is typically tiny), producing `r_{3,t}` — which in turn slightly perturbs lag 2 — and so on. **You iterate this cycle until BOTH the first and second order autocorrelations fall below a convergence threshold.**

### General case — remove up to m orders of autocorrelation (Eqs. 24–28)

The general algorithm to remove the first `m` orders of autocorrelation:

1. Remove 1st-order autocorrelation (Step 1).
2. Remove 2nd-order, then iterate 1↔2 until both < threshold.
3. Remove 3rd-order, then iterate 1↔2↔3 until all three < threshold.
4. ... continue up to lag m.

The general adjustment at lag m (Eq. 24):
```
r_{m,t} = ( r_{m−1,t} − c_m · r_{m−1,t−m} ) / ( 1 − c_m )
```

The general coefficient (Eq. 27), for the target `a_{m,m} = 0`:
```
c_m = [ 1 + a_{m−1,2m} ± sqrt( (1 + a_{m−1,2m})² − 4 a_{m−1,m}² ) ] / ( 2 a_{m−1,m} )
```
with the real-root condition (Eq. 28):
```
a_{m−1,m}²  ≤  (1 + a_{m−1,2m})² / 4
```

**Implementation of the general algorithm (this is the heart of the model):**

```
function unsmooth(observed_returns, m_lags, target_acf=zeros, tol=1e-4, max_iter=100):
    r = observed_returns.copy()
    for target_lag in 1..m_lags:
        # bring target_lag's autocorrelation to its target, then re-clean all lower lags
        repeat up to max_iter:
            # (a) remove/adjust the current target_lag
            a_k = sample_autocorrelation(r, lag=target_lag)
            a_2k = sample_autocorrelation(r, lag=2*target_lag)
            c = solve_c(a_k, a_2k, d=target_acf[target_lag])   # quadratic, choose stable root
            r = (r - c * shift(r, target_lag)) / (1 - c)
            # (b) re-clean all lags 1..target_lag that were perturbed
            for lower_lag in 1..target_lag:
                a_low  = sample_autocorrelation(r, lag=lower_lag)
                a_2low = sample_autocorrelation(r, lag=2*lower_lag)
                c_low = solve_c(a_low, a_2low, d=target_acf[lower_lag])
                r = (r - c_low * shift(r, lower_lag)) / (1 - c_low)
            # (c) check convergence: all lags 1..target_lag within tol of their targets
            if max_k(|acf(r, k) - target_acf[k]| for k in 1..target_lag) < tol:
                break
    return r, collected_c_parameters
```

Notes:
- Each adjustment at lag k loses the first k observations (the shift). Track cumulative observation loss and align the final series' index. Document this.
- Store every `c` parameter applied (there will be more than m of them because of the re-cleaning iterations). Report the *net* effect and the per-lag final adjustment.
- Default `m_lags = 4` (the paper "successfully eliminated the first four autocorrelations" for hedge fund indices; make it configurable 1–8).
- Default `target_acf` is all zeros (remove all autocorrelation up to lag m). Support a user-supplied target vector, since the paper stresses the method can set "any desired level of autocorrelation at any lag" — including leaving a deliberate residual.

### Reconstruction & the implied smoothing relationship

The final `r_{m,t}` IS the unsmoothed (true) return series — reconstruction is intrinsic to the iterative adjustment (no separate inversion step, unlike a Kalman or Chow-Lin approach). Also expose the implied relationship between reported and true returns. For the two-lag case the paper gives (Eq. 23):
```
r_{0,t} ≈ (1 − c_1)(1 − c_2) r_{2,t} + c_1 r_{0,t−1} + c_2 r_{0,t−2} − c_1 c_2 r_{0,t−3}
```
Provide a method that returns the implied MA-style weights on lagged reported returns for the general m case (this is diagnostic — it shows the effective smoothing kernel the model inferred).

---

## Risk-Property Computation

After unsmoothing, compute and return:

1. **Variance inflation.** The variance of the unsmoothed series relative to the observed. For a single lag (Eq. 8): `Var[r_{1,t}] = (1 + c_1² − 2 c_1 a_{0,1}) / (1 − c_1)² · Var[r_{0,t}]`. For multiple lags, the approximate cumulative form (Eq. 29): `Var[r_{m,t}] ≈ Π_i (1 + c_i² − 2 c_i a_{i−1,·}) / (1 − c_i)² · Var[r_{0,t}]`. Since the paper notes the iteration makes this only approximate, ALSO report the exact empirical variance of the final reconstructed series and flag any discrepancy between the formula and the empirical value.
2. **Autocorrelation function before and after**, for lags 1..(m+2), so the user can confirm the target lags were driven to (near) zero and see what happened at higher lags.
3. **Annualized volatility** of observed vs. unsmoothed (infer periods-per-year from the index frequency). The paper reports risk increases of 60–100% for smoothed hedge fund indices — expect and surface a large inflation for autocorrelated inputs.
4. **The c-parameter trace** (all adjustments applied) and the **number of iterations** to convergence per target lag.
5. **Convergence diagnostics:** whether every target lag reached its target within tolerance, and the final residual autocorrelation at each lag.

---

## Integration with the Existing Library

### Conform to the `SmoothingModel` protocol

```python
class OkunevWhiteSmoothing:
    """
    Generalized return-unsmoothing of Okunev & White (2003).

    An iterative extension of Geltner desmoothing that removes autocorrelation
    up to an arbitrary lag m — not merely first-order — by repeatedly applying
    a Geltner-style adjustment at each lag and re-cleaning perturbed lower lags
    until all targeted autocorrelations fall below a threshold. Pure time-series
    method: uses only the return series' own autocorrelation structure, no
    factors.

    References
    ----------
    Okunev, J., & White, D. (2003). Hedge Fund Risk Factors and Value at Risk
    of Credit Trading Strategies. Working paper (SSRN 460641).
    Geltner, D. (1991, 1993). [return unsmoothing for appraisal-based returns].
    Brooks, C., & Kat, H. (2001). [smoothing in hedge fund returns].
    """

    def __init__(
        self,
        n_lags: int = 4,
        target_acf: list[float] | None = None,   # default: zeros up to n_lags
        tol: float = 1e-4,
        max_iter: int = 100,
        root_selection: str = 'stable',          # 'stable' = smaller |c|; see notes
    ):
        ...

    def fit(self, observed_returns, factor_returns=None, priors=None) -> DesmoothedResult:
        """
        Run the iterative unsmoothing. `factor_returns` and `priors` are accepted
        for protocol compatibility but are NOT used — emit an informational note
        if they are supplied. Estimation uses only the return series' ACF.
        """
        ...

    def desmooth(self, observed_returns, smoothing_params) -> np.ndarray:
        """Apply a known sequence of c-adjustments to reconstruct true returns."""
        ...

    def log_likelihood(self, params, observed_returns, factor_returns=None) -> float:
        """Gaussian log-likelihood of the unsmoothed residuals (for model comparison)."""
        ...

    def log_prior(self, params, prior_config) -> float:
        """Returns 0.0 — this model uses no priors. Present for protocol compliance."""
        ...
```

### `DesmoothedResult` contents

- `true_returns`: the final unsmoothed `r_{m,t}` (pandas Series, aligned, with the first observations lost to lagging dropped)
- `smoothing_params`: dict with the full `c` trace, the per-lag net adjustment, the implied smoothing kernel (Eq. 23 generalization), `n_lags`, and iterations-to-convergence
- `posterior_summary`: N/A for this frequentist method — populate with the point estimates and note there is no posterior (this is not a Bayesian model). Include bootstrap confidence intervals on the variance-inflation ratio if cheap to compute (resample blocks of the observed series, re-run, report the spread).
- `diagnostics`: observed vs. unsmoothed ACF (lags 1..m+2), variance-inflation (formula and empirical), annualized vol before/after, Ljung-Box on the unsmoothed series (should now be insignificant up to lag m), convergence flags.

### Registry

Register under `'okunev_white'` so it can be selected via `AssetClassConfig(smoothing_model='okunev_white', ...)`.

### Fallback behavior

If the real-root condition (Eq. 7 / 15 / 28) fails at some lag — meaning the target autocorrelation cannot be achieved by a real adjustment — honor the library's existing `FallbackPolicy`:
- `strict`: raise with a clear message naming the offending lag and the violated condition.
- `warn`: skip that lag (leave its autocorrelation unadjusted), record a warning, and continue.
- `auto`: fall back to removing only the lags that are solvable (typically the first 1–2), and warn that the requested depth m was not reached.

---

## Library Placement

Add to the existing structure (do not restructure):

```
private_assets_frequency/
├── desmoothing/
│   └── okunev_white.py            # NEW: OkunevWhiteSmoothing class + solver helpers
└── tests/
    └── test_okunev_white.py       # NEW
```

Put the quadratic `c`-solver and the ACF utilities as private module-level helpers in `okunev_white.py` (or reuse existing ACF utilities from `utils/time_series.py` if present).

---

## Testing Requirements

1. **AR(1) recovery.** Generate an AR(1)-smoothed series with known first-order autocorrelation. Assert the model with `n_lags=1` recovers `c_1 ≈ a_{0,1}` and drives lag-1 autocorrelation to ~0.

2. **MA(2)/AR(2) higher-order test.** Generate a series with significant first AND second order autocorrelation (e.g., an MA(2) or AR(2) process). Assert that `n_lags=1` (Geltner-like) leaves significant second-order autocorrelation, but `n_lags=2` removes both. This is the central point of the paper — the test must demonstrate Okunev-White succeeds where single-lag Geltner fails.

3. **Convergence test.** Assert the iterative cycle converges (all target lags < tol) within max_iter for a range of realistic autocorrelation structures, and that removing lag 2 does perturb lag 1 (Eq. 19) but the re-cleaning restores it.

4. **Variance inflation direction.** Assert that for positively autocorrelated input, unsmoothed variance > observed variance (matches the paper's finding of 60–100% risk increases). Assert the formula-based variance (Eq. 29) is close to the empirical variance of the reconstructed series.

5. **Target-ACF flexibility.** Assert that a nonzero `target_acf` (e.g., leave lag-1 at 0.1) is achieved — the paper stresses the method can hit "any desired level at any lag," not just zero.

6. **No-op test.** A series with zero autocorrelation should pass through essentially unchanged (all c ≈ 0).

7. **Real-root failure / fallback test.** Construct an input that violates the real-root condition at some lag and assert each `FallbackPolicy` behaves as specified.

8. **Mean preservation.** The paper notes the adjusted series has (near) the same mean as the observed. Assert the unsmoothed mean is close to the observed mean.

9. **Protocol compliance.** Verify it satisfies the `SmoothingModel` protocol and is selectable through the registry and `AssetClassConfig`.

---

## Critical Implementation Notes

1. **Solve the quadratic numerically; don't transcribe the closed forms.** Eqs. 6, 14, 21, 27 are algebraically equivalent to solving `A c² + B c + C = 0` built from the ACF condition. Build the quadratic from the autocorrelation-target equation and solve with `np.roots` or the quadratic formula. This avoids transcription errors and makes the code readable. Cross-check against the AR(1) special case `c_1 = a_{0,1}`.

2. **Root selection matters.** Each quadratic has two roots. The paper's usable solution is the one giving a stable, invertible adjustment (|c| < 1 keeps `1 − c` well away from zero, avoiding division blow-up). Default to the smaller-magnitude root but VERIFY it achieves the target ACF; if it doesn't, use the other root. Expose `root_selection` for control.

3. **Re-cleaning is mandatory, not optional.** The defining feature vs. naive sequential lag removal is that adjusting lag k reintroduces autocorrelation at lower lags. If you skip the inner re-cleaning loop, the output will NOT have zero autocorrelation at all target lags. This is the single most important correctness requirement.

4. **Observation loss.** Each lag-k adjustment consumes k initial observations. Over m lags plus iterations, track alignment carefully. Return the final series on the valid (trimmed) index and report how many observations were lost.

5. **Sample ACF estimation.** Use a consistent ACF estimator (e.g., statsmodels `acf` or a documented biased/unbiased choice) and recompute the ACF from the *current adjusted series* at each iteration — the coefficients depend on the running autocorrelations `a_{m,·}`, not the original `a_{0,·}`.

6. **Convergence guard.** Cap iterations; if a target lag oscillates without converging, stop, warn, and return the best-so-far with a diagnostic. Do not loop forever.

7. **Frequency-agnostic; factor-free.** Works on monthly or quarterly data. Do not require or use factor returns. This model is especially suited to hedge fund indices (the paper's application) where higher-order autocorrelation is common — note this in the docstring and consider adding an `'okunev_white'` option to the hedge fund presets as an alternative to `ma_glm`.

8. **Dependencies:** numpy, scipy, statsmodels, pandas only — consistent with the parent library.

---

## Build Order

1. `desmoothing/okunev_white.py`:
   a. ACF helper (or reuse `utils/time_series.py`)
   b. `_solve_c(a_k, a_2k, d)` — build & solve the quadratic, select stable root, check real-root condition
   c. `_adjust(series, lag, c)` — apply Eq. 24 and re-align
   d. the iterative `fit` (remove lag 1; then for each higher target lag, remove it and re-clean lower lags to convergence)
   e. `desmooth`, risk-property computation, `log_likelihood`/`log_prior`
2. Registry registration under `'okunev_white'`
3. `tests/test_okunev_white.py` — all tests above
4. Run the suite and fix failures.

At the end, produce `docs/okunev_white_notes.md` documenting the equations as implemented, the quadratic-solver approach, the re-cleaning iteration, the observation-loss accounting, and any deviations from the paper (there should be none in the core procedure — only Sections IV–V are omitted by scope).
