# RudinReparamSmoothing — Implementation Notes

Implements the reparameterised unsmoothing model from Rudin, Mao, Zhang & Fink
(2019), "Fitting Private Equity into the Total Portfolio Framework," *Journal
of Portfolio Management* 46(2), 60-77 — specifically the "Risk Estimation
Through Unsmoothing" section (their Eqs. 1-4).

Files added:

| File | Purpose |
| --- | --- |
| `private_assets_frequency/desmoothing/theta_w_conversion.py` | Bidirectional θ ↔ w conversion (polynomial inversion of the lag-operator series). |
| `private_assets_frequency/desmoothing/rudin_reparam.py` | The `RudinReparamSmoothing` class implementing the `SmoothingModel` protocol. |
| `private_assets_frequency/tests/test_theta_w_conversion.py` | Unit tests for the conversion utility. |
| `private_assets_frequency/tests/test_rudin_reparam.py` | Unit + synthetic-recovery tests for the smoother. |

Registered in `pipeline/runner.py::_make_smoother` under the key
`'rudin_reparam'`; the `SmoothingModelKind.RUDIN_REPARAM` enum entry was added
to `core/protocols.py`. When invoked via a preset, `AssetClassConfig.ma_lags`
is reused as the lag-order selector (Q).

## Equations as implemented

Notation matches the paper: `r̃ᵒ` is the smooth observed return, `rᴱ` the
true economic return, `F` the systematic-factor matrix.

**Eq. 1** — smoothing relationship (Conner 2003; Getmansky-Lo-Makarov 2004;
Pedersen-Page-He 2014):

```
r̃ᵒ_t = Σ_{j=0..Q_w} w_j · rᴱ_{t-j},    Σ w_j = 1.
```

**Eq. 2** — factor model on the true series:

```
rᴱ_t = α + Σ_i β_i · F^i_t + ε_t,        ε_t ~ N(0, σ²).
```

**Eq. 3** — Stefek-Suryanarayanan reparameterisation (true as function of
observed and its lags):

```
rᴱ_t = θ_0 · r̃ᵒ_t + Σ_{j=1..Q} θ_j · r̃ᵒ_{t-j},   Σ θ_j = 1.
```

**Eq. 4** — the single estimating regression (spec's stated form):

```
r̃ᵒ_t = α + θ_0 · (Σ_i β_i · F^i_t) + Σ_{j=1..Q} θ_j · r̃ᵒ_{t-j} + ε_t.
```

Estimated by OLS with regressors `[1, F_t, r̃ᵒ_{t-1}, …, r̃ᵒ_{t-Q}]` on
`r̃ᵒ_t`, dropping the first Q observations. The paper's linear-OLS +
algebraic-unwinding recipe is followed exactly — **no nonlinear least
squares** as the composite factor coefficient `θ_0·β_i` might otherwise
tempt.

## Factor-coefficient unwinding

The raw OLS coefficients partition as:

* intercept coefficient → `α` (read directly);
* factor coefficients   → `c_i = θ_0 · β_i` (composite);
* lag coefficients      → `θ_1, …, θ_Q` (read directly).

Then:

```
θ_0  = 1 − Σ_{j≥1} θ_j       # sum-to-one identity (Eq. 3)
β_i  = c_i / θ_0             # unwind
```

Two safeguards protect the β unwinding:

* `|θ_0| < 0.1` → warning recorded in `diagnostics['notes']`;
* `|θ_0| < 1e-12` → hard `ValueError` (division blows up; sample too short
  or degenerate for the requested lag order).

## Reconstruction (Eq. 3)

`r̂ᴱ_t = θ_0 · r̃ᵒ_t + Σ_{j=1..Q} θ_j · r̃ᵒ_{t-j}` for `t ≥ Q`. No matrix
inversion. The first Q observations are lost per the finite-lag construction
and dropped from `DesmoothedResult.true_returns.index`.

## Reported risk properties

All annualisation uses `periods_per_year` inferred from the return index
(quarterly → 4, monthly → 12, weekly → 52, daily → 252). Overrideable via
the constructor.

| Quantity | Formula |
| --- | --- |
| `alpha` (per period) | OLS intercept |
| `alpha_annualised` | `α · ppy` (`"simple"`) or `(1+α)^ppy - 1` (`"geometric"`); default `simple` to match the paper. |
| `beta[factor]` | `c_i / θ_0` (unwound) |
| `vol_annualised` | `std(r̂ᴱ) · √ppy` |
| `idio_vol_annualised` | `std(r̂ᴱ − α − β·F) · √ppy` |
| `pct_variance_explained` | `1 − Var(r̂ᴱ − α − β·F) / Var(r̂ᴱ)` |

`diagnostics` additionally contains the Eq. 4 OLS residual autocorrelation
tests (Ljung-Box + Durbin-Watson), the regression R², and the realised
`Σ θ_j` and `θ_0`. Standard errors default to classical OLS; set
`use_hc_se=True` for heteroskedasticity-consistent (HC1) SEs.

## Lag-selection diagnostic (`select_lag_order`)

Sweeps `Q ∈ {0, …, max_lag}` (default `max_lag=4`) and returns a
`LagSelectionResult` with a per-Q DataFrame and a recommended lag.

Two diagnostic modes:

1. **In-sample (default).** The table's `variance_ratio_in_sample` column
   is `Var(ε_{Eq.4}) / Var(r̂ᴱ)`. Because in-sample R² mechanically favours
   higher Q, the default recommendation applies a parsimony rule: pick the
   smallest Q whose `pct_variance_explained` is within `tolerance` of the
   best; break ties by preferring Q=1 (the paper's empirical finding).
2. **Out-of-sample.** Supply `test_observed_returns` and
   `test_factor_returns`. The trained `(θ, β, α)` are applied to the test
   window, and the table gains a `variance_ratio_out_of_sample` column
   equal to `Var(test_actual − model_prediction) / Var(test_actual)` on the
   reconstructed series. Ratios > 1 signal overfitting; the recommendation
   minimises the OOS ratio. This mirrors the paper's methodology
   (parameters fitted on the broad index, applied out-of-sample to
   mini-programs).

The paper's mini-program simulation and its shrinkage-across-paths
approach are **out of scope**; only the standalone-scope in-sample /
supplied-test-window variants are implemented.

## θ ↔ w conversion

`theta_w_conversion.theta_to_w` and `w_to_theta` are formal inverses under
the lag operator: `Θ(L) · W(L) = 1`. Both take a finite polynomial and
return the truncated formal-series inverse via the standard convolution
recursion:

```
b_0 = 1 / a_0
b_k = -(1/a_0) · Σ_{j=1..min(k, Q)} a_j · b_{k-j},   k ≥ 1.
```

The finite-Q θ implies an **infinite-order** w in general and vice versa;
the default truncation is 200 lags (already ≪ 1e-30 for the paper's
typical θ). `conversion_diagnostics(x)` reports the tail magnitude and the
transferred sum-to-one error; a warning string is populated if the tail is
above `1e-6`.

## Deviations from the paper

Nothing in the core model. Only the mini-program simulation, the
shrinkage-across-paths estimator, the diversification study, and the total-
portfolio framework are omitted — all four are declared out of scope in the
spec.

## Sanity anchors from Exhibit 1

Not asserted in the automated test suite (data-dependent), but for
reference: on SSPE Buyouts with the S&P 500 factor the paper's Q=1 results
are approximately α ≈ 4.64 %, β ≈ 0.53, vol ≈ 13.3 %, %-explained ≈ 46 %.
Users applying the model to comparable data should recover values in that
neighbourhood; large deviations warrant investigating whether the input
series matches SSPE-style aggregation.
