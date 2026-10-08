# `nowcast/` — Implementation Notes

Produces provisional estimates of private markets index returns that have not yet
been published, from public market data that is already available. The reference
is Cohen, Mantoan, Nesheim, de Paula, Turrell & Yang (2023), "Nowcasting using
regression on signatures," arXiv:2305.10256v3 — specifically its Theorem 1 on
linear signature terms and its Section 3.3 on irregularly observed data.

The motivating situation, which is also the package's end-to-end acceptance test
(`TestMotivatingScenario::test_the_whole_path`): on **7 October 2026** the Q2-2026
PE index return has not been published, Q3-2026 has just closed, and public market
data runs to today. Both missing quarters are nowcast, with prediction intervals,
and handed to `FrequencyPipeline` as clearly flagged provisional observations.

Files added:

| File | Purpose |
| --- | --- |
| `nowcast/DESIGN_NOTES.md` | Written before any code: the three ways this module could silently report over-optimistic validation results, and how each is prevented. |
| `nowcast/information_set.py` | `InformationSet`, `build_information_set`, vintage handling. The correctness foundation. |
| `nowcast/base.py` | `Nowcaster` protocol, `NowcastResult`. |
| `nowcast/uncertainty.py` | Residual pools, the rolling-origin backtest, recursive predictive simulation. |
| `nowcast/signatures.py` | Path construction, `compute_signature` adapter, pure-numpy backend, `FactorPCA`. |
| `nowcast/models/naive.py` | Model 0 — `NaiveCarryForward`. |
| `nowcast/models/smoothing_regression.py` | Model 1 — `SmoothingRegressionNowcaster`, the benchmark. |
| `nowcast/models/structural.py` | Model 2 — `StructuralSmoothingNowcaster`, plus the λ-convention arithmetic. |
| `nowcast/models/signature.py` | Model 3 — `SignatureNowcaster`, with the feature-budget guard. |
| `nowcast/models/_base.py` | Shared machinery (not in the spec's file list — see *Deviations*). |
| `nowcast/evaluation.py` | Rolling-origin harness, Diebold-Mariano, subperiod reports, plots. |
| `nowcast/integration.py` | `extend_with_nowcasts`, `to_pipeline_returns`, `reconcile`. |
| `tests/nowcast/` | 499 tests across 7 files, including all 10 tests the module spec requires. |

The only change to existing code is the `allow_provisional` flag on
`FrequencyPipeline` — see *The permitted change* below. `pyproject.toml` gained
the `nowcast` / `signatures` extras and two warning filters.

---

## 1. The two targets

The API makes the choice explicit because the two answer different questions and
only one of them is a nowcast.

**`target='reported'` (default)** — the smoothed return the index provider will
publish for quarter Q. This is the genuine nowcasting problem: the quantity
exists, is unknown, and will be revealed, so it can be scored.

**`target='true'`** — the unsmoothed economic return. Its predictable part is
*only* the systematic component α + β·F_Q from the factor model; the idiosyncratic
part has expectation zero by construction. Implemented as a thin wrapper over the
fitted factor decomposition, with three consequences the code enforces rather than
merely documents:

- **No recursion.** `r_Q = α + β F_Q + ε_Q` does not depend on the lagged
  *reported* value, and `F_Q` is already known for a closed quarter. So the
  interval has the **same width at every horizon** — tested to 3 %.
- **The interval is dominated by irreducible idiosyncratic risk.** With
  `θ₀ ≈ 0.4`, `σ_ε = σ_e/θ₀` makes the `target='true'` interval about 2.5× wider
  than the `target='reported'` one. A poor R² here is the expected result, not a
  model failure, and no quantity of additional public data narrows it.
- **It cannot extend a reported series.** `extend_with_nowcasts` raises on a
  `target='true'` result: the two are different quantities on different scales,
  and appending one to the other would make the series mean two things at once.

Every `NowcastResult` with `target='true'` carries that caveat in
`diagnostics['interpretation']`, so it travels with the number.

`SignatureNowcaster` does **not** support `target='true'` and says so: its
coefficients sit on signature terms, not factor returns, so there is no α, β or θ₀
to read.

---

## 2. Information-set rules

The information set at a date `as_of` is exactly:

- public factor observations with timestamp ≤ `as_of`;
- reported private index values whose **publication date** ≤ `as_of` — *not* whose
  reference quarter ends before `as_of`;
- optional early-signal observations with timestamp ≤ `as_of`.

The second bullet is the whole point. A Q2-2026 index value has a reference
quarter that ended 30 June 2026 but is not published until roughly October. On
7 October the quarter is *closed and unpublished*, which is precisely the object
being nowcast.

### Which reported values are known — the one place this is decided

**Vintage mode** (preferred, selected by passing `vintages`). For each quarter,
take the rows with `release_date <= as_of` and keep the one with the latest
`release_date` (ties resolve to the later row in the caller's ordering). A quarter
with no visible release is simply absent, and that absence is what makes it a
target. `_validate_vintages` additionally rejects any row dated before its own
quarter end — a leak hiding in the input data.

**Lag-rule mode** (fallback). Quarter Q is known iff
`Q.end_time + publication_lag <= as_of`, default 100 days. The value used is the
*final revised* one, which is why this mode emits a `NowcastDataModeWarning`
naming the mechanism, and escalates to an exception under `FallbackPolicy.STRICT`.

### Why the mode matters more than it looks

Revisions in private markets indices are mean-reverting as late-reporting funds
arrive, so the revised series is systematically **easier to predict** than the
first print. Training and scoring on revised data therefore flatters every model
twice over — the lagged predictor is a value that did not exist yet, and the
target is partly an average of information that arrived after the nowcast date.

`evaluation_target ∈ {'first_print', 'latest'}` is explicit, defaults to
`'first_print'` in vintage mode, and is recorded in every result. The effect is
measurable: `TestCoverage::test_coverage_degrades_with_revisions` injects a 1pp
revision standard deviation and realised 80 % coverage falls from ~0.80 to the
0.45–0.80 band, in the expected direction.

### Horizons and the ragged edge

`horizon h` is the number of quarters between the last published quarter and the
target. Horizons come from the **calendar**, not the data:

```python
last = self.last_closed_quarter       # Period(as_of,'Q') - 1 unless as_of >= its end_time
out  = [last_pub + h for h in range(1, (last - last_pub).n + 1)]
return out[: self.max_horizon]
```

On the motivating case this gives `last_published=2026Q1`,
`unpublished=[2026Q2, 2026Q3]`, horizons 1 and 2. The open quarter 2026Q4 is
excluded by default; its factor coverage is 0.07 and `factor_row("2026Q4")` raises
unless `require_complete=False`.

Multi-quarter horizons are handled **recursively**: the h=2 nowcast conditions on
draws from the h=1 predictive distribution, re-evaluating the conditional mean at
each draw rather than plugging in a point estimate.

### The guarantee, and how it is enforced

Three independent barriers, in order:

1. **`InformationSet` is the only model input.** It is built by one function, which
   truncates at construction and stores copies. The untruncated frames are never
   attributes of the object, so there is no code path to a post-`as_of`
   observation. `fit` raises `TypeError` on anything that is not an
   `InformationSet`.
2. **Design assertions.** Models 1 and 3 check
   `info.unpublished_quarters(include_open_quarter=True) ∩ design.index` and raise
   on any overlap.
3. **The harness verifies the models' own records.**
   `evaluation._assert_nested_selection` reads each model's recorded
   `selection_index` and `training_quarters` and raises `NestedSelectionError` —
   re-raised ahead of the general handler, so a leak stops the run even under
   `FallbackPolicy.WARN`. It runs at every evaluation date for every model (264
   checks in a 132-date run).

The acceptance criterion is behavioural, not architectural: perturbing any
post-`as_of` factor row by +0.25, any future vintage value by +0.5, any
unpublished reported value by +1.0, or any future early signal by +10.0 leaves
predictions **bit-for-bit identical** — `np.testing.assert_array_equal` on the
draws, never `allclose`.

---

## 3. The model hierarchy, and the Kalman-nesting theorem

### Model 0 — `NaiveCarryForward`

`ŝ_{Q+h} = s_Q`. Not a strawman: reported private markets returns are heavily
autocorrelated *by construction* — appraisal smoothing is an AR(1) filter with λ
often 0.5–0.8 — so carrying the last print forward is a genuinely strong
predictor, and a factor model that cannot beat it has demonstrated nothing. It
carries the **vintage in force**, not the revised value, so the benchmark is
real-time too.

Two properties worth knowing. Its default is `interval_method='empirical_direct'`
because it has no dynamics to propagate. And its directional hit rate is
undefined, not zero: a carry-forward's predicted change is identically zero, so it
makes no directional call at all. The harness reports `NaN` with
`directional_coverage = 0.0` rather than a spurious 0 %.

### Model 1 — `SmoothingRegressionNowcaster` (the benchmark)

```
s_Q = a + Σ_k b_k F_{k,Q} + Σ_{j=1}^{q} c_j s_{Q−j} + e_Q
```

This is the **reduced form of the library's own Stage-1 models**, which is why it
is the benchmark rather than a baseline. Substituting `r_Q = α + β F_Q + ε_Q` into
the AR(1) appraisal filter `s_Q = (1−λ)r_Q + λ s_{Q−1}`:

```
s_Q = (1−λ)α  +  (1−λ)β F_Q  +  λ s_{Q−1}  +  (1−λ)ε_Q
      └─ a ─┘    └─── b ───┘    └─ c₁ ─┘    └── e_Q ──┘
```

so one OLS on observables recovers `b = (1−λ)β` and `c₁ = λ` without ever
estimating λ separately. For `q ≥ 2` the same algebra is the Rudin-Mao-Zhang-Fink
Eq. 4 form with `θ₀ = 1 − Σ_j c_j`, and `unwind_to_structural` reuses that
mapping.

Recovery on the linear DGP, across λ ∈ {0.3, 0.6, 0.85}: at λ=0.6, `b = 0.4469`
against `(1−λ)β = 0.44`; `c₁ = 0.6075` against `λ = 0.6`; unwound `β = 1.139`
against 1.10; R² = 0.956.

The ridge penalty applies to the **factor coefficients only** — never the
intercept, never the lagged-return coefficients. `c_j` *is* the smoothing
parameter; shrinking it toward zero would quietly undo the thing being modelled.

### Model 2 — `StructuralSmoothingNowcaster`

Reuses an already-fitted `DesmoothedResult` instead of re-estimating, so a nowcast
is consistent with the λ and β Stage 1 has already committed to.

- From `ar1_bayesian`: `s_Q = (1−λ)(α + β·F_Q) + λ s_{Q−1}`. That model regresses
  `y(λ) = (s_t − λ s_{t−1})/(1−λ)` on `[F_t, 1]`, so its reported `alpha`, `beta`
  and `sigma_eps` are already on the **unsmoothed** scale and enter directly.
- From `rudin_reparam`: its Eq. 4 form as fitted,
  `s_Q = α + θ₀ Σ_i β_i F_{i,Q} + Σ_j θ_j s_{Q−j}`.
- `okunev_white` raises `UnsupportedSmoothingModelError`: it is factor-free, so
  there is no `α + β F_Q` to build a nowcast from.

The Rudin route reproduces Model 1's point nowcast **to 0.0 exactly** — both are
the same OLS reduced form, and multiplying the unwound `β` back by `θ₀` recovers
the fitted composite coefficient. That identity is the tightest available check on
the Eq. 4 reconstruction.

The AR(1) route differs slightly (λ = 0.6506 against OLS 0.6075), and the reason is
instructive: the `N(1.15, 0.5)` β prior pulls β to 1.273, and because λ and β are
only *jointly* identified, λ follows. This is the λ-β confound the parent library's
own diagnostics flag, demonstrated directly in
`test_beta_prior_moves_lambda_the_confound_in_action`. It is also why
`integrate_lambda_posterior=True` warns: drawing λ while holding β at its marginal
posterior mean cuts *across* the posterior ridge rather than along it. Pass
`smoother=<fitted AR1BayesianSmoother>` for joint draws;
`diagnostics['posterior_draw_mode']` records which you got.

### Model 3 — `SignatureNowcaster`, and Theorem 1

Regresses `s_Q` on standardised path-signature features of the public market path
over a lookback window ending at Q, plus `s_{Q−1}`.

The reference paper's **Theorem 1** states that regression on the *linear*
signature terms contains the linear/Kalman predictor as a special case. The
implementation makes that nesting **exact**, not asymptotic:

- `keep_sigs='linear'` keeps only words in which each **non-time** channel appears
  at most once. Time is exempt, because repeated time indices contribute
  polynomial-in-time *weights*, not products of data increments: `S^(0,1) = ∫s dX_s`
  and `S^(1,0) = ∫X_s ds` are both linear in the path, while `S^(1,1) = ½X₁²` is
  not. At `d=2, level=2` exactly one term, `S^(1,1)`, is dropped.
- **Time-only terms are dropped from the regressors**, because time is rescaled to
  `[0,1]` over the realised window and so `S^(0) = 1`, `S^(0,0) = ½`,
  `S^(0,0,0) = ⅙` on *every* row regardless of the data. Keeping them would add
  exact collinearity with the intercept.
- `level_channel='simple'` builds channels as the cumulative simple-return level
  `Π(1+r) − 1`, whose level-1 term equals `F_Q` **to the last bit** (the `'log'`
  default gives `log(1+F_Q)`, which nests only to second order).

With those three, at `level=1` with one factor channel the retained feature set is
`[intercept, F_Q, s_lag1]` — the benchmark's own design — and the predictions agree
to **1.4e-17**.

Representational nesting is not the same as out-of-sample improvement at ~90
observations, and the module is built to tell the two apart. Measured over 73
rolling origins, everything refitted at each:

| DGP | h=1 RMSE ratio vs benchmark | h=2 |
| --- | --- | --- |
| Path-dependent, `level=2` | **0.564** | **0.531** |
| Path-dependent, `level=1` | 1.003 | — |
| Linear, `level=2` | 1.052 | 1.056 |

Level 2 cuts RMSE 44–47 % where the within-quarter path genuinely matters, and is
5 % *worse* where it does not. The `level=1` row is what makes this
interpretable: it sits at parity, so the gain is attributable to the level-2 path
terms specifically rather than to signatures in general. On the path-dependent DGP
the term `S^(x,time)` tracks the appraiser's average-level statistic at ρ = 0.99995
while the quarter-end return manages only 0.898.

### The feature budget

Mandatory, computed from word counts alone so it runs before a single signature is
built. `n_features > n_train / 5` is refused under `FallbackPolicy.STRICT` and
skipped with a warning under `WARN`/`AUTO`; a mixed grid proceeds on its admissible
members. Where it bites, at 90 quarters (budget 18, including `s_lag1` and the
intercept):

```
d=2 (time + 1 factor)   linear L2 =  5   ok
d=3 (time + 2 factors)  linear L2 = 10   ok
d=4 (time + 3 factors)  linear L2 = 17   ok     <- 3 channels still fits
d=5 (time + 4 factors)  linear L2 = 26   over   <- 4 is the cap at level 2
d=3 (time + 2 factors)  linear L3 = 22   over   <- level 3 admits 1 channel
d=2 (time + 1 factor)   all    L4 = 28   over
```

Counts are from `feature_budget()` and include `s_lag1` and the intercept while
**excluding** the time-only terms the model drops. On that basis the spec's
"time + 1–2 channels at level 2" is **conservative**: three factor channels do
fit at level 2 on a ~90-quarter sample. Level 3 is the tighter constraint — it
admits one factor channel and no more.

The linear restriction is what buys the headroom: at `d=3, level=3` it cuts 39
signature terms to 23.

### The signature backend

Two facts make the computation **exact** for a piecewise-linear path, with no
quadrature: a straight-line segment with increment Δ has `S^k = Δ^⊗k / k!`, and
Chen's identity multiplies signatures under concatenation. `signature_numpy` folds
each segment's tensor exponential left to right; the flattening is consistent under
concatenation by construction, which is why no index arithmetic appears in
`chen_product`.

On this machine the numpy backend is not a fallback — it is the only backend
(`iisignature` does not build; `esig` requires numpy ≥ 2, which breaks the pinned
pandas 2.1.4). So the library cross-check skips and correctness rests on
identities tested directly: the closed form for a straight line (1e-14), level 1
equalling the total increment (1e-13), the shuffle identity
`S^i S^j = S^(i,j) + S^(j,i)` (7e-16), **Chen's identity** at levels 1–3 (2e-15),
three-way associativity (1e-12), and reparametrisation invariance (1e-12).

---

## 4. The λ convention — read this before trusting a borrowed λ

λ is **not dimensionless**. It is the per-period decay of the appraisal filter, so
its value depends on the frequency the model was fitted at, and a nowcast of a
*quarter* needs a *quarterly* λ.

**The spec's premise does not hold for the code as built.** `ar1_bayesian` fits the
raw observed series at whatever frequency the caller passes, with **no**
rolling-annual transform — `use_rolling_annual` is described in its own docstring
as planned, not implemented — and `FrequencyPipeline` passes the native quarterly
series. So in the shipped configuration **no conversion is needed**, and applying
one would corrupt λ. `infer_fitted_convention` therefore reads the fitted result's
own index rather than assuming, and converts only when it must.

`convert_lambda_to_quarterly` holds the two conversions that do exist.

**`'annual_non_overlapping'` → `λ_q = λ_a^{1/4}`.** The quarterly filter is an
EWMA, `s_t = (1−λ_q) Σ_j λ_q^j r_{t−j}`. Sampling every fourth quarter and
unrolling four steps:

```
s_4T = λ_q⁴ s_{4T−4} + (1−λ_q) Σ_{j=0}^{3} λ_q^j r_{4T−j}
```

The coefficient on the lagged *observation* is `λ_a = λ_q⁴`. The weights are
exactly consistent — `(1−λ_q)Σ_{j=0}^{3}λ_q^j = 1 − λ_q⁴ = 1 − λ_a` — but the
implied annual true return is the λ-weighted average of four quarterly returns,
not their equal-weighted compound. That is the approximation `ar1_bayesian`'s
docstring warns about. Verified by simulating the actual appraisal filter: true
λ_q ∈ {0.3, 0.6, 0.85} recovers {0.282, 0.595, 0.849}.

**`'annual_rolling'` → numerical inversion, *not* a power law.** An overlapping sum
of an AR(1) is an **ARMA(1,3)**. For window k:

```
ρ₁(S) = Σ_{m=−(k−1)}^{k−1} (k−|m|) φ^{|m+1|}  /  Σ_{m=−(k−1)}^{k−1} (k−|m|) φ^{|m|}
```

confirmed against a 4-million-draw simulation to four decimals. At φ=0 it equals
**0.75**: a rolling four-quarter return is 75 % autocorrelated even when the
quarterly series is white noise (Working, 1960). The map therefore compresses all
of λ_q ∈ [0,1) into λ_a ∈ [0.75,1) and must be inverted numerically.

The cost of getting this wrong is the largest arithmetic error available in the
module:

```
true λ_q = 0.60  ->  AR(1) on rolling-annual recovers λ_a ≈ 0.908
                     λ_a^(1/4)           = 0.976   WRONG
                     numerical inversion = 0.600   correct
```

Desmoothing divides by `(1−λ)`, so mistaking 0.60 for 0.976 inflates recovered
volatility by a factor of ~17. The power law is not offered for the rolling case;
`power_law_would_give` is recorded in the diagnostics so a reviewer can see what
was avoided, and a λ below the 0.75 floor raises rather than returning a fiction.

**Rolling-annual fits cannot be auto-detected** and must be declared. A rolling
four-quarter series is indexed *quarterly* — only its values are annual — so
inference reports `'quarterly'` and leaves λ unconverted. Pass
`fitted_convention='annual_rolling'` explicitly. A test pins that limitation so it
cannot be quietly forgotten.

---

## 5. Prediction intervals

Three mechanisms, in the order to prefer them.

**`'empirical_recursive'` (default for models with dynamics).** The innovation law
is the model's own **out-of-sample one-step** rolling-origin errors, estimated
inside the information set. Multi-quarter horizons are simulated: draw `s_Q`,
re-evaluate the conditional mean of `s_{Q+1}` at *each draw*, add a fresh
innovation.

**`'empirical_direct'` (default for models without dynamics).** Point nowcast plus
a resample of the errors at that same horizon.

**`'parametric'`.** Gaussian with the regression's prediction variance
`σ̂²(1 + x'(X'X)⁻¹x)`. Narrower, smoother, and wrong in the left tail of real data.

### Why recursion uses the one-step pool

The spec asks for two things that pull against each other: residuals from the
errors *at the same horizon*, and recursive simulation for h ≥ 2. Doing both double
counts — the historical h=2 errors were themselves produced by a nowcast that
plugged in an h=1 nowcast, so they already contain the accumulated uncertainty the
recursion is about to add again.

Resolution: recursion is the mechanism, and every step draws from the **one-step**
pool. For `s_Q = a + b F_Q + c s_{Q−1} + e_Q` this reproduces the analytic answer
exactly — `Var(h=2) = σ_e²(1 + c²)`, so at c ≈ 0.6 the h=2 interval is ~17 %
wider — with no double counting. Verified to 3 % against 200,000 draws at levels 1–3.

The literal horizon-matched reading survives as `'empirical_direct'`, and it is not
merely a fallback: recursion with i.i.d. one-step innovations assumes the model is
correctly specified, whereas the direct pool absorbs compounding misspecification.
Every recursive result therefore carries `diagnostics['interval_80_direct']`, so a
materially wider direct interval is visible as a misspecification signal rather
than hidden by the choice of method.

### The pool is deliberately not centred

Out-of-sample errors in a small sample have a non-zero mean, and that mean is
kept. The consequence is that `point` is the model's raw forecast while the draws
sit `mean(pool)` away from it — so **`point` is not the centre of its own
interval**. The alternative is worse: centring the pool would bias-correct the
interval while leaving the point forecast uncorrected. `diagnostics['pool_mean']`
records the gap, the evaluation harness reports mean error separately, and a test
pins the identity `draw_mean = point + (1 + c₁)·pool_mean` at h=2 so the gap cannot
be mistaken for a bug or silently change.

### Small-sample fallback, and one bug it caught

A pool with fewer than 20 out-of-sample errors falls back to in-sample residuals
inflated by `√(1 + p/n)`, and the result's `interval_method` is suffixed
`'+in_sample_inflated'` so the weaker basis travels with the number.

The internal backtest's `min_train` defaults to **40** — the spec's figure for the
evaluation harness, reused deliberately. At 24, pooling errors from very short
windows mixes estimation-error regimes and inflates the pool ~23 %, which showed up
as h=1 coverage of **0.867** against 0.80 nominal. At 40 the coverage is 0.797 /
0.820 at the 80 % level and 0.947 / 0.940 at 95 %, holding across three seeds.

Note that spec test 7's ±5pp tolerance is about *one* standard error at 73
evaluation points (`√(.8×.2/73) = 4.7pp`), so the coverage fixture uses 200 quarters
to get ~133 points and measure calibration rather than seed luck.

---

## 6. How to read the evaluation report

`rolling_origin_evaluation(models, data, start, end, ...)` replays every model
quarter by quarter, re-estimating **everything** at each date — coefficients,
scalers, PCA rotations, hyperparameters, residual pools. That is deliberately
wasteful; reusing a global fit is the leak the harness exists to avoid.

`EvaluationResult.summary()` prints provenance, then metrics, then
Diebold-Mariano, then a verdict paragraph. A real run on the linear DGP:

```
data mode            : vintage  (evaluation_target='first_print')
evaluation dates used: 201 of 244
primary offset       : 0 days after quarter end (grid: (0, 30))
benchmark            : smoothing_regression

               model  horizon   n   rmse    mae  mean_error  directional_hit_rate  coverage_80  coverage_95
smoothing_regression        1 100 0.0103 0.0083     -0.0012                0.9500       0.7700       0.9200
           signature        1 100 0.0105 0.0084     -0.0010                0.9300       0.7700       0.9300
               naive        1 100 0.0424 0.0332      0.0004                   NaN       0.7600       0.8900

    model  horizon   n  rmse_ratio  statistic  p_value  lags             verdict
signature        1 100      1.0253     1.0261   0.3073     4  no significant difference
    naive        1 100      4.1232     6.9335   0.0000     4  worse than smoothing_regression

No model beats smoothing_regression at the 5 % level in subperiod 'full'. ...
this is a substantive result rather than a failed run: at this sample size the
extra features do not pay for themselves.
```

Reading notes, in the order they tend to mislead:

**`mean_error` is not an error.** It is the bias. Read it alongside `coverage_80`:
a model with low RMSE and poor coverage is overconfident, which is the failure mode
that matters for risk use.

**A negative DM statistic favours the model named in the row.** The p-value uses
HAC (Newey-West) variance with the **Harvey-Leybourne-Newbold** small-sample
correction and a `t(n−1)` reference, because at ~70–100 points the uncorrected test
over-rejects noticeably. `lags` defaults to `max(h−1, ⌊4(n/100)^{2/9}⌋)`.

**"no significant difference" is the common outcome and does not mean equality.**
At this sample size the test has limited power. When a positive claim *is* made the
summary attaches the multiple-comparison caveat automatically — the p-values are
not corrected for testing several models across several horizons.

**Subperiod rows have small `n`.** GFC, COVID and 2022 are 4–8 quarters each.
Treat them as descriptive, not inferential. An empty subperiod is omitted rather
than reported with `n=0`.

**`directional_hit_rate` excludes abstentions.** A forecast of no change is not a
wrong call. `directional_coverage` reports the fraction of rows on which a call was
made — 0 for a carry-forward, > 0.9 for the benchmark.

**A model with no rows is a failure, not a result.** If a model fails at every
date it would otherwise vanish silently from the comparison, so
`models_with_no_rows` and an `EvaluationWarning` surface it (escalating under
`strict`).

### Nowcast evolution is event-driven, not continuous

Worth internalising before configuring `as_of_offsets_days`. Once a quarter has
**closed**, no further public factor data about *that quarter* can arrive, so its
nowcast cannot move with the calendar — offsets 30, 60 and 90 return byte-identical
numbers. What moves it is a **publication event**: when the previous quarter is
released, the target drops from h=2 to h=1 and conditions on a published value
instead of a nowcast of one.

So a target quarter has as many distinct nowcasts as horizons it passes through,
and `(0, 30)` is the right default. Offset 0 is also the **only** one that reaches
h=2, because a ~100-day publication lag leaves only a ~9-day window per quarter
with two quarters simultaneously closed-and-unpublished. The motivating example is
that window. `plot_nowcast_evolution` shows the progression.

---

## 7. Integration with the pipeline

```python
extended = extend_with_nowcasts(pe_index, nowcasts)      # 6-column flat frame
returns  = to_pipeline_returns({"us_buyout": extended})  # + provisional_mask in .attrs
result   = FrequencyPipeline(..., allow_provisional=True).run()
extended = reconcile(extended, official_prints)          # when the print lands
log      = reconciliation_log(extended)                  # feeds the empirical pools
```

### The guarantee

**Provisional quarters never enter an estimate.** Stage 1 (desmoothing) and
Stage 2 (the factor model) are fitted on published rows only; the desmoothed
series is then extended across the provisional tail by applying the **fitted**
filter via `SmoothingModel.desmooth`, and Stages 3–4 upsample the extended series.

Measured: λ = 0.650614901761 on both paths, byte-identical, along with β, α, σ_ε,
the factor betas and `n_obs`. The test asserts `==`, not `approx` — a tolerance
would let a small influence through and call it success.

The reason is circularity, not tidiness. A nowcast is produced *by* a factor model;
feeding it back as data would shrink the estimated λ and inflate R² for no
informational reason, and the numbers would improve precisely in proportion to how
confident the nowcaster was.

Two details make the splice sound. Every smoothing filter in the library is
**causal** (each `r_t` reads `s_t` and its lags), so extending the input leaves the
prefix unchanged — but `_extend_desmoothed` checks it rather than assuming,
refusing the extension if the published values move by more than 25 % of their
standard deviation. The check is a tolerance rather than an identity because
`ar1_bayesian` reports a posterior-*integrated* `true_returns` while `desmooth` is
the plug-in filter at the posterior mean; the provisional tail is explicitly the
plug-in estimate.

### What provisional rows *do* reach

Stage 3's Chow-Lin nuisance parameters (`rho`, the indicator betas) are estimated
on the series handed to it, which includes the provisional tail. This is
unavoidable: the temporal-aggregation constraint must hold on *every* low-frequency
row including the provisional ones, so Stage 3 cannot be fitted short and
extrapolated. These are disaggregation nuisances rather than the desmoothing or
factor parameters the spec names, and the span is recorded in
`disaggregation.diagnostics['estimation_window']`. A test confirms the aggregation
constraint does hold on the provisional quarters to 1e-10.

`uncertainty_mode='full'` with `allow_provisional=True` warns: the
posterior-sampling path re-runs Stages 1–3 on the full frame and does not honour
the split.

### Reconciliation

`reconcile` overwrites provisional rows with official prints, keeping the nowcast,
the realisation, the error, the interval and whether it covered. The log
accumulates across calls, so reconciling quarter by quarter builds the realised
out-of-sample error history — which is the honest basis for prediction intervals:
errors from nowcasts actually made at the time, against values actually printed.
Group by `nowcast_model` **and** `horizon` before pooling: pooling across models
describes none of them, and pooling across horizons discards the horizon widening
the intervals exist to express.

A revision to an already-published row is overwritten with a warning and **not**
logged — a revision is not a reconciliation, because the nowcast was never scored
against that vintage.

---

## 8. The permitted change to existing code

`FrequencyPipeline` gained one flag, `allow_provisional: bool = False`, plus the
behaviour it implies: three private helpers (`_mask_has_provisional`,
`_smoothing_param_vector`, `_extend_desmoothed`), the provisional split at the top
of `_run_strategy`, and the `is_provisional` diagnostics block.

The spec allows exactly one new parameter, so the mask travels in
`returns.attrs['provisional_mask']`, set by `to_pipeline_returns()`.
`allow_provisional=True` **without** the mask raises — the pipeline must never
guess which rows are nowcasts, because guessing wrong means estimating on them.
The reverse case (mask present, flag off, and the mask actually marking something)
warns that the nowcasts *will* enter the estimates.

`DataFrame.attrs` is metadata and pandas does not promise to propagate it through
every operation, so pass the returned frame to the pipeline **directly** rather
than slicing or concatenating it first. The pipeline reads the attribute in
`__post_init__`, before touching the frame. If a future revision is willing to
spend a second parameter, an explicit `provisional` argument would be more robust.

`runner.py` carries 7 pre-existing ruff findings at lines unrelated to this change
(unused imports, quoted annotations, unused locals). They were left alone.

---

## 9. Deviations from the spec

| Deviation | Why |
| --- | --- |
| No λ conversion is applied by default | `ar1_bayesian` fits raw quarterly data, not rolling-annual. Applying the spec's conversion would corrupt λ. Both conversions are implemented and the convention is inferred. |
| `λ_a^{1/4}` is refused for rolling-annual fits | It is wrong there by a factor of ~17 in recovered volatility. Numerical inversion of `ρ₁` is used instead. |
| `multiplier_terms='time'` warns that it is degenerate | Time-only signature terms are structurally constant, so interacting them with `s_{Q−1}` is exactly collinear with `s_{Q−1}`. Implemented for fidelity; `'all'` is the non-degenerate version. |
| `level_channel='simple'` added | Makes the Theorem-1 nesting exact rather than second-order. `'log'` remains the default. |
| Recursion draws from the one-step pool | Horizon-matched pools plus recursion would double count. `'empirical_direct'` preserves the literal reading. |
| `min_train=40` for the internal pool | At 24 the pooled error sd runs ~23 % high and intervals over-cover. 40 is the spec's own evaluation minimum. |
| Only `alpha` is selected by default in Model 3 | A four-point grid. `DESIGN_NOTES.md` Risk 3: every extra selection dimension is paid for in selection bias at ~90 observations. |
| `models/_base.py` added | Shared horizon bookkeeping, the recursive/direct branch, and the fit–predict guard, so four models do not reimplement them. Private. |
| `vintage_backfill=True` by default | Real vintage tables start years after the index. Silently discarding that history is worse than using it under a flagged lag rule. |
| Tests live in `private_assets_frequency/tests/nowcast/` | Matches the repo's existing layout and `pytest testpaths`, not the spec's top-level `tests/`. |
| The reference PDF is not in `docs/references/` | Not available offline; no network fetch was attempted. |

---

## 10. Known limitations

**The sample is ~90–100 quarterly observations, and that is the binding
constraint.** It is why the feature budget exists, why the default hyperparameter
grid has four points, why the Diebold-Mariano test carries a small-sample
correction, and why "no significant difference" is the usual verdict. Asymptotic
intuitions do not apply.

**Prediction intervals are calibrated on synthetic DGPs, not on real data.** Spec
test 7 holds to ±5pp on the linear DGP with ~133 evaluation points. Real data has
revisions, regime breaks and a fat left tail that 90 observations cannot
characterise. Treat the intervals as a lower bound on uncertainty, especially under
stress — and note that the GFC/COVID/2022 subperiod rows have n = 4–8.

**Most users will run in lag-rule mode, and it is optimistic.** Vintage tables for
private markets indices are hard to obtain and usually start later than the index.
The module cannot fix that; it refuses to let it pass unremarked, warns by default,
and escalates under `strict`.

**`target='reported'` and `target='true'` are not comparable.** Reporting their
accuracy side by side without the caveat compares two different questions. The
unsmoothed return's interval is dominated by irreducible idiosyncratic risk.

**Mid-quarter nowcasting of the open quarter is not supported.** It is arguably the
most valuable case — and it would give genuinely continuous evolution — but a
partial-quarter factor aggregate is not comparable to the full-quarter aggregates
the models train on. Supporting it properly needs either scaling or models trained
on partial windows.

**Signature libraries are absent in this environment**, so the numpy backend is
unvalidated against an independent implementation. Its correctness rests on
algebraic identities (Chen, shuffle, reparametrisation invariance, closed forms),
which is strong evidence but not the same thing. Install `iisignature` where it
builds and the cross-check test will run.

**Stage 3's disaggregation parameters see provisional rows**, as described above.
Bounded and recorded, but not zero.

**The λ-β confound limits Model 2.** The parent library flags it as the central
identification weakness of the Bayesian AR(1) fit, and it propagates: λ-only
posterior draws understate parameter uncertainty in a *biased* way, and a β prior
moves the recovered λ. Prefer joint draws via the `smoother` argument.

**`strict` mypy is not met**, by `nowcast/` (147 findings) or by the parent library
(308). The composition is the same in both: bare `np.ndarray` annotations and
missing pandas stubs. The findings that could hide real bugs — argument types,
redefinitions, stale ignores, missing annotations — were fixed.

---

## References

- Cohen, Mantoan, Nesheim, de Paula, Turrell & Yang (2023) — "Nowcasting using
  regression on signatures," arXiv:2305.10256v3. Theorem 1 (linear signature
  terms); Section 3.3 (irregular data); Eq. 11 (multiplier terms); Figures 8–9
  (nowcast evolution).
- Chen (1957) — "Integration of paths, geometric invariants and a generalized
  Baker-Hausdorff formula." Chen's identity.
- Lyons, Caruana & Lévy (2007) — *Differential Equations Driven by Rough Paths*.
- Chevyrev & Kormilitzin (2016) — "A primer on the signature method in machine
  learning." Basepoints, time augmentation, rectilinear interpolation.
- Geltner (1993) — AR(1) appraisal smoothing.
- Rudin, Mao, Zhang & Fink (2019) — "Fitting Private Equity into the Total
  Portfolio Framework," Eqs. 3–4.
- Working (1960) — "Note on the correlation of first differences of averages in a
  random chain." The overlap floor behind the rolling-annual conversion.
- Diebold & Mariano (1995) — "Comparing predictive accuracy."
- Harvey, Leybourne & Newbold (1997) — "Testing the equality of prediction mean
  squared errors." The small-sample correction.
- Newey & West (1987) — HAC covariance estimation.
- Croushore & Stark (2001) — "A real-time data set for macroeconomists." Why
  pseudo-real-time validation flatters a model.
- Tashman (2000) — "Out-of-sample tests of forecasting accuracy."
- Zou & Hastie (2005) — "Regularization and variable selection via the elastic
  net."
