# OkunevWhiteSmoothing — Implementation Notes

Implements the generalized return-unsmoothing procedure of **Okunev & White
(2003)**, "Hedge Fund Risk Factors and Value at Risk of Credit Trading
Strategies" (Section III.A, SSRN 460641). The model removes autocorrelation up
to an arbitrary lag *m* — an iterative extension of Geltner desmoothing beyond
first order — using **only the return series' own autocorrelation structure**.
It is a *pure time-series* method and uses no factor returns.

Files added:

| File | Purpose |
| --- | --- |
| `private_assets_frequency/desmoothing/okunev_white.py` | `OkunevWhiteSmoothing` class + `_solve_c` / `_adjust` / `_sample_acf` / `_implied_kernel` helpers. |
| `private_assets_frequency/tests/test_okunev_white.py` | Unit + synthetic-recovery tests (all 9 spec requirements). |

Registered under the key `'okunev_white'`:

- `SmoothingModelKind.OKUNEV_WHITE` added to `core/protocols.py`.
- `pipeline/runner.py::_make_smoother` constructs it; `AssetClassConfig.ma_lags`
  is reused as the depth selector *m* (default 4).
- `core/config.py` exempts it (with `no_smoothing`) from the `beta_priors`
  requirement via the new `_FACTOR_FREE` set — it takes no factors.

Scope: **only** the core unsmoothing procedure and its risk-property
computation. The factor-mapping regressions, VaR analysis, and nonlinear
factor exposures of the paper's Sections IV–V are out of scope, as specified.

## Equations as implemented

Notation follows the paper: `r_{m,t}` is the return after *m* adjustments
(`r_{0,·}` = observed input, the final `r_{m,·}` = reconstructed truth);
`a_{m,n}` is the lag-*n* autocorrelation after *m* adjustments; `c_k` is the
adjustment coefficient at lag *k*; `d_k` the desired target for lag *k*.

**Eq. 3** — smoothing assumption (the manager mixes the true current return
with lagged *reported* returns):

```
r_{0,t} = (1 - α) r_{m,t} + Σ_i β_i r_{0,t-i},   (1 - α) = Σ_i β_i.
```

**Eq. 24** — the general Geltner-style adjustment at lag *k* (`_adjust`):

```
r_{k,t} = ( r_{k-1,t} - c_k · r_{k-1,t-k} ) / ( 1 - c_k ).
```

**Eqs. 5/13/27** — the coefficient. Rather than transcribing the paper's
closed forms, `_solve_c` builds the quadratic directly from the condition
`a_{k,k}(c) = d` (Eq. 5 rearranged):

```
A c² + B c + C = 0,
A = a_k - d,
B = -(1 + a_2k) + 2 d a_k,
C = a_k - d,
```

where `a_k = a_{k-1,k}` and `a_2k = a_{k-1,2k}` are the *current* sample
autocorrelations of the series being adjusted. Because `A = C`, the two roots
are **reciprocals**; the default `root_selection='stable'` returns the
smaller-magnitude root (`|c| ≤ 1`), keeping `1 - c` away from zero.

**Real-root condition** (Eqs. 7/15/28) — a real root exists iff the
discriminant is non-negative, equivalently:

```
(a_k - d)²  ≤  (1 + a_2k - 2 d a_k)² / 4.
```

`_solve_c` raises `_NoRealRoot` when this fails; `fit` routes that through the
`FallbackPolicy` (see below).

**AR(1) special case.** For a genuine AR(1) input `a_2k = a_k²`, and with
`d = 0` the quadratic factorises to roots `a_k` and `1/a_k`; the stable root
gives `c = a_k` — the Geltner result, asserted as a unit test.

## Quadratic-solver approach

`np.roots` is avoided in favour of the explicit two-root quadratic formula so
the reciprocal-root structure and the stable-root selection are transparent.
Guards:

- `|A| < 1e-15` → already at target, return `c = 0` (no-op).
- discriminant `< 0` (beyond float noise) → `_NoRealRoot`.
- selected root with `|1 - c| < 1e-8` → `ValueError` (unstable transform).

## The re-cleaning iteration (the defining feature)

Removing lag *k* **reintroduces** autocorrelation at lower lags (Eq. 19). The
algorithm therefore processes target lags `1, 2, …, m` in turn; at each target
lag it repeatedly (a) adjusts the target lag and (b) re-cleans every lower lag,
until all lags `1..target` are simultaneously within tolerance
(Eqs. 20–22). Skipping the inner re-clean would leave residual
autocorrelation at the lower lags — this is the single most important
correctness requirement, and is verified in the tests (a lag-2 adjustment is
shown to perturb lag 1, which the re-clean then restores).

`_run_stage` runs one target-lag cascade and returns the **best-so-far**
iterate (minimum residual autocorrelation over the lags targeted so far).
Rationale: the coupled cascade oscillates around a finite-sample floor rather
than reaching an arbitrarily tight tolerance, so we keep the best iterate and
stop on either convergence, a stagnation `patience` counter, or `max_iter`
(the spec's convergence guard). Best-so-far also **bounds observation loss** to
the loss at the best iterate.

## Significance band — a deliberate small-sample refinement

The paper's idealised statement is "drive the autocorrelation to zero." Taken
literally against a *sample* autocorrelation, that would chase finite-sample
sampling noise: white noise of length *N* has sample ACFs of order
`±1/√N` (e.g. ±0.13 at N=240), all far above an absolute `tol=1e-4`, so the
procedure would "adjust" genuinely-uncorrelated data and needlessly consume
observations.

To respect **statistical validity** and **small-sample robustness** (the
project's decision priorities), a lag is only adjusted when its sample
autocorrelation is *statistically distinguishable* from its target, using a
Bartlett-style band:

```
band(n) = max(tol, acf_significance / √n),   acf_significance default 1.96.
```

Convergence/`target_met` reporting uses the same band. Setting
`acf_significance=0` recovers the literal "zero the sample ACF" behaviour
(governed by `tol`). This is the only deviation from the paper's core
procedure, and it is a refinement, not a change of method: on data with
genuinely significant higher-order autocorrelation (the paper's setting) the
band is easily exceeded and all targeted lags are removed, exactly as intended.

## Observation-loss accounting

Each lag-*k* adjustment references `r_{t-k}` and so loses the first *k*
observations of the series it is applied to. `fit` tracks the cumulative loss
exactly: `n_observations_lost == Σ (lag of each applied adjustment)`, and

```
n_effective + n_observations_lost == n_observations,
true_returns.index == observed_returns.index[n_observations_lost:].
```

Both are asserted in the tests. Because near-target adjustments are skipped
(within the band) and the cascade keeps only the best iterate, the loss stays
small in practice (single digits for the MA(2) datasets in the tests, versus
hundreds if every micro-adjustment were applied).

## Reconstruction & the implied smoothing kernel

Reconstruction is intrinsic — the final `r_{m,·}` *is* the reconstructed truth
(no separate inversion step). Since every adjustment is an LTI filter
`(1 - c_i L^{k_i}) / (1 - c_i)`, the net transform from `r_0` to `r_m` is the
product of their transfer functions. `_implied_kernel` expands that product
into a finite polynomial `P(L)` such that

```
r_{m,t} = (1 / Π_i (1 - c_i)) · Σ_k P_k · r_{0,t-k},
```

equivalently the paper's Eq. 23 form
`r_{0,t} = Π_i(1-c_i) r_{m,t} + Σ_{k≥1}(-P_k) r_{0,t-k}`. For the two-lag case
this reproduces Eq. 23 exactly
(`P = (1 - c_1 L)(1 - c_2 L²) = [1, -c_1, -c_2, c_1 c_2]`), asserted in the
tests. `desmooth` applies a stored `(lag, c)` trace sequentially and
reproduces `fit`'s reconstruction bit-for-bit.

## Risk properties (returned in `diagnostics` / `posterior_summary`)

| Quantity | Source |
| --- | --- |
| `acf_before` / `acf_after` (lags 1..m+2) | sample ACF of observed vs reconstructed |
| `variance_inflation_formula` | Eq. 29: `Π_i (1 + c_i² - 2 c_i a_pre)/(1 - c_i)²` |
| `variance_inflation_empirical` | `Var(r_m)/Var(r_0)` (exact) |
| `variance_inflation_discrepancy` | `|formula - empirical|` (Eq. 29 is only approximate under iteration) |
| `vol_annualised_observed/unsmoothed` | `std · √ppy`, `ppy` inferred from the index frequency |
| `ljung_box_stat` / `ljung_box_pvalue` | on the reconstructed series (should be insignificant up to lag *m*) |
| `mean_observed` / `mean_unsmoothed` | the adjustment preserves the mean (asserted) |
| `converged`, `target_met`, `residual_acf`, `skipped_lags` | convergence diagnostics |
| `c_trace`, `per_lag_net_c`, `iterations_to_convergence` | the full adjustment trace |
| `variance_inflation_bootstrap` (in `posterior_summary`) | optional moving-block bootstrap CI on the empirical ratio (`n_bootstrap > 0`) |

`posterior_summary` is labelled `okunev_white_frequentist` with an explicit
note that there is no posterior — this is a frequentist method.

## Fallback behaviour (`FallbackPolicy`)

When the real-root condition fails at some lag:

- **STRICT** — raise `ValueError` naming the offending lag and the violated
  condition.
- **WARN** — skip that lag (leave it unadjusted), record a warning in
  `diagnostics['notes']` and the lag in `diagnostics['skipped_lags']`,
  continue.
- **AUTO** — skip the unsolvable lag(s), continue with the solvable ones, and
  additionally flag that the requested depth *m* was not reached.

All three are exercised by constructing an infeasible target
(`target_acf=[0.99]` on white noise).

## Deviations from the paper

Only two, both in service of the project's decision priorities and documented
above:

1. The **significance band** (`acf_significance`) governs when a lag is
   adjusted, instead of literally zeroing the finite-sample ACF. Set it to `0`
   to recover the literal behaviour.
2. **Best-so-far + stagnation stopping** replaces "iterate until exactly below
   `tol`," because the coupled cascade oscillates around a finite-sample floor
   (the spec's sanctioned convergence guard).

Everything in the core mathematical procedure — the Eq. 24 adjustment, the
quadratic coefficient, the real-root condition, and the mandatory re-cleaning
iteration — follows the paper exactly. Sections IV–V are omitted by scope.
