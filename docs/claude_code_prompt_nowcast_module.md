# Claude Code Prompt — Nowcasting Module for Lagged Private Markets Returns

## Task

Add a `nowcast/` sub-package to the existing `private_assets_frequency` library (repo: github.com/boraeang/upsampler). It must produce **provisional estimates of private markets index returns that have not yet been published**, using public market data that is already available.

Read these before writing any code:
- `docs/claude_code_prompt_private_assets_frequency_v2.md` — parent library spec (`SmoothingModel` protocol, `DesmoothedResult`, `AssetClassConfig`, registry, `FallbackPolicy`, presets).
- `core/protocols.py`, `core/config.py`, `pipeline/presets.py` — to confirm actual signatures in the code as built.
- `desmoothing/ar1_bayesian.py`, `desmoothing/rudin_reparam.py` — the structural nowcaster below reuses their fitted parameters.
- Reference paper: Cohen, Mantoan, Nesheim, de Paula, Turrell & Yang, "Nowcasting using regression on signatures" (arXiv 2305.10256v3). Copy the PDF into `docs/references/` if available.

**Motivating situation.** On 7 October 2026 the Q2-2026 private equity index return has not been published, Q3-2026 has just closed, and public market data is available through today. The module must produce nowcasts for both missing quarters, with prediction intervals, and hand them to the rest of the pipeline as clearly flagged provisional observations.

**Do not modify** any existing desmoothing, decomposition, disaggregation or daily module. The nowcast package only consumes them.

---

## Your Role & Decision Priorities

Act as implementer and senior reviewer. Same priority order as the parent library:
1. Statistical validity > architectural purity
2. Numerical stability > theoretical elegance
3. Robustness to small samples > asymptotic correctness (≈ 80–100 quarterly observations is the whole sample)
4. Interpretability > marginal accuracy gains

Before coding, write `nowcast/DESIGN_NOTES.md` listing the top 3 ways this module could silently produce over-optimistic validation results (look-ahead leakage, revision/vintage effects, hyperparameter selection on the test set are the expected ones) and how the implementation prevents each.

---

## Core Concepts

### What is being nowcast

Two possible targets. The API must make the choice explicit:

- **`target='reported'` (default):** the smoothed return the index provider will publish for quarter Q. This is the genuine nowcasting problem.
- **`target='true'`:** the unsmoothed economic return. This is *not* a nowcast in the same sense — its predictable part is simply the systematic component α + β·F_Q from the factor model, and the idiosyncratic part has expectation zero. Implement it as a thin wrapper around the fitted factor decomposition. Document clearly that the uncertainty is dominated by idiosyncratic risk and is not reducible with more public data.

### Information set and the "as-of" date

Every nowcast is made **as of a date** `as_of`. The information set at `as_of` is exactly:
- public factor observations with timestamp ≤ `as_of`;
- reported private index values whose **publication date** ≤ `as_of` (not whose reference quarter ≤ `as_of`).

All model fitting, feature scaling, hyperparameter selection and feature construction must use only this information set. This is the single most important correctness requirement in the module.

### Publication dates and vintages

Private markets indices are published with a lag and revised for several quarters as late-reporting funds arrive. Support two data modes:

1. **Vintage mode (preferred):** user supplies a long DataFrame `vintages` with columns `[quarter, release_date, value]` — one row per published value of each quarter, including revisions. The value known at `as_of` for quarter Q is the latest row with `release_date ≤ as_of`.
2. **Lag-rule mode (fallback):** user supplies only the final revised series plus a `publication_lag` (default 100 calendar days after quarter end, configurable per index). Emit a warning that validation in this mode is pseudo-real-time and likely optimistic, because revised data is used as if it had been available at first release.

Also expose `evaluation_target ∈ {'first_print', 'latest'}`: whether nowcasts are scored against the first published value or the latest revised value. Default `'first_print'` in vintage mode.

### Ragged edge and multi-quarter horizons

At a given `as_of`, there may be one or more closed-but-unpublished quarters (two in the motivating example). Define **horizon h** = number of quarters between the last published quarter and the quarter being nowcast (h=1 for Q2, h=2 for Q3 in the example). Nowcast recursively: the h=2 nowcast conditions on the h=1 nowcast. Propagate uncertainty by simulation — draw the h=1 value from its predictive distribution, then nowcast h=2 conditional on each draw. Do not plug in point estimates and report a naive interval.

---

## Models

All nowcasters implement one protocol:

```python
class Nowcaster(Protocol):
    name: str
    def fit(self, info: InformationSet) -> "Nowcaster": ...
    def predict(self, info: InformationSet, quarters: list[pd.Period],
                n_draws: int = 2000, rng: np.random.Generator | None = None) -> list[NowcastResult]: ...
```

`InformationSet` is an immutable container built by a single function `build_information_set(reported, public_factors, as_of, vintages=None, publication_lag=None, early_signals=None)`. It truncates everything to `as_of`. Models never receive raw untruncated data.

### Model 0 — `NaiveCarryForward` (baseline)

Nowcast = last published quarterly return. Intervals from empirical out-of-sample errors (see Uncertainty). This is the bar every other model must beat.

### Model 1 — `SmoothingRegressionNowcaster` (primary baseline)

The joint smoothing regression, which is the reduced form of the AR(1)/Rudin smoothing models:

```
s_Q = a + Σ_k b_k · F_{k,Q} + Σ_{j=1}^{q} c_j · s_{Q−j} + e_Q
```

- `s_Q` = reported quarterly return; `F_{k,Q}` = public factor returns over quarter Q, compounded from the high-frequency data.
- `q ∈ {1, 2}`, default 1. Selected by rolling-origin validation only, never in-sample.
- Estimate by OLS, with optional ridge on the factor coefficients.
- Factor columns come from the strategy's `AssetClassConfig` (reuse the existing factor-selection helper; respect whatever resolution of the `default_factors` vs `beta_priors` naming mismatch is in the code).
- Optional lagged factor terms `F_{k,Q−1}`, off by default.

This is the model the reference paper's theorem says signature regression contains as a special case. It must be implemented carefully and treated as the main benchmark.

### Model 2 — `StructuralSmoothingNowcaster`

Uses an already-fitted desmoothing model's parameters instead of re-estimating:
- From `ar1_bayesian` (smoothing `s_Q = (1−λ)·r_Q + λ·s_{Q−1}`, with `E[r_Q] = α + β·F_Q`): nowcast = `(1−λ)·(α + β·F_Q) + λ·s_{Q−1}`.
- From `rudin_reparam`: use its fitted θ, β, α in its Eq. 4 form.

Takes a `DesmoothedResult` as input. Verify the exact parameter names and quarterly vs rolling-annual conventions in the existing code before implementing (the Bayesian AR(1) model is estimated on rolling annual returns — convert λ to the quarterly convention and document the conversion). For `ar1_bayesian`, optionally integrate over the λ grid posterior rather than using the posterior mean, so parameter uncertainty enters the predictive distribution. Not available for `okunev_white` (no factors) — raise a clear error.

### Model 3 — `SignatureNowcaster` (challenger, from the reference paper)

Regression of the reported return on **path-signature features of the public market path**, plus the last known reported return.

**Path construction (per target quarter Q):**
- Lookback window: `[end_of_Q − lookback, end_of_Q]`, with `lookback` a hyperparameter in {1, 2, 4} quarters (default 1 = within-quarter only). Never extend past `as_of`.
- Channels: time (rescaled to [0, 1] over the window) plus the **cumulative log-return level** of each selected public factor, rebased to 0 at window start. Use levels, not returns — signatures are built from path increments.
- Data frequency for the path: daily if available, else monthly, controlled by `path_frequency`.
- Missing / irregular observations: forward fill (default) or rectilinear interpolation (option), as described in the paper's Section 3.3.
- Optional basepoint (prepend a zero point) — default `True`.
- Optional dimension reduction: PCA of factor returns to `n_components` channels before building the path, fitted on training data only.

**Signature computation:**
- Truncation level `level ∈ {1, 2, 3}`, default 2. Do not offer level 4 by default (see feature budget).
- `keep_sigs ∈ {'all', 'linear'}`: `'linear'` keeps only terms in which each non-time channel appears at most once (the "linear signature terms" of the paper's Theorem 1). Default `'linear'`.
- Backend: an adapter `compute_signature(path, level)` that uses `iisignature` if installed, else `esig`, else a **pure-numpy implementation** for piecewise-linear paths up to level 3 (tensor exponential of each segment's increment, combined with Chen's identity). Implement the numpy version regardless and use it to cross-check the library backend in tests.

**Regression:**
- Features: standardized signature terms (scaler fit on training rows only) + `s_{Q−1}` (last known reported value; for h=2 this is the h=1 draw) + intercept.
- Optional "multiplier" terms (paper Eq. 11): interactions of `s_{Q−1}` with the time-only signature terms. Off by default.
- Estimator: scikit-learn `ElasticNet` / `Ridge`; hyperparameters (`alpha`, `l1_ratio`, `level`, `lookback`, `keep_sigs`) chosen by inner rolling-origin validation within each training window (nested — see Evaluation).

**Feature budget guard (mandatory):** compute `n_features` before fitting. If `n_features > n_train / 5`, refuse the configuration under `FallbackPolicy.strict`, or skip it with a warning under `warn`/`auto`. With ~90 training quarters this effectively caps you at time + 1–2 channels at level 2. State the computed budget in the fitted model's diagnostics.

**Early signals (optional channels):** accept an `early_signals` DataFrame of irregularly timestamped series that may proxy private marks — e.g. listed private equity index levels (LPX50-style), NAV announcements of listed PE fund-of-funds, preliminary index prints. Each is an extra path channel, forward-filled, truncated to `as_of`. This is where the irregular-data strength of signatures is most relevant; keep it fully optional.

---

## Uncertainty

Every `NowcastResult` contains:

```python
@dataclass(frozen=True)
class NowcastResult:
    quarter: pd.Period
    horizon: int
    as_of: pd.Timestamp
    model: str
    target: Literal['reported', 'true']
    point: float
    draws: np.ndarray            # predictive draws, shape (n_draws,)
    interval_80: tuple[float, float]
    interval_95: tuple[float, float]
    interval_method: str
    diagnostics: dict            # features used, n_train, feature budget, fallback events
```

Predictive distribution:
- **Default (`'empirical'`):** point forecast + residuals drawn from that model's own **out-of-sample** rolling-origin errors at the same horizon (historical nowcast errors, not in-sample residuals). If fewer than 20 OOS errors exist, fall back to in-sample residuals inflated by `sqrt(1 + p/n)` and flag it.
- **Optional (`'parametric'`):** Gaussian with the regression's prediction variance (Models 1–2), plus posterior parameter uncertainty for Model 2 with `ar1_bayesian`.
- For h ≥ 2: recursive simulation as described above.

---

## Evaluation Harness

`nowcast/evaluation.py` with `rolling_origin_evaluation(models, data, start, end, horizons=(1, 2), inner_validation_quarters=12, ...)`:

1. For each evaluation date `as_of` (default: the date each quarter would have been nowcast — e.g. 30 days after quarter end, configurable; also support a grid of several dates per quarter to show how the nowcast evolves as data arrives):
   - Build the `InformationSet` at `as_of`.
   - **Nested selection:** hyperparameters are chosen using only quarters inside the information set, via an inner rolling-origin split on the last `inner_validation_quarters` published quarters. The outer evaluation quarter is never used for selection.
   - Fit, predict all unpublished quarters at horizons 1..H.
2. Score against `evaluation_target` (first print / latest).
3. Report per model × horizon: RMSE, MAE, mean error (bias), directional hit rate, 80%/95% interval coverage, and the Diebold-Mariano test vs `SmoothingRegressionNowcaster` with HAC variance.
4. Report the same metrics on subperiods: full sample, GFC (2008Q3–2009Q2), COVID (2020Q1–2020Q4), 2022.
5. Plot: nowcast vs realized over time with 80% bands; error by horizon; for the "grid of as-of dates" option, the nowcast path within each quarter as information arrives (in the style of the paper's Figures 8–9).

Minimum initial training window: 40 quarters, configurable.

---

## Integration with the Pipeline

`nowcast/integration.py`:

```python
def extend_with_nowcasts(reported: pd.Series, nowcasts: list[NowcastResult]) -> pd.DataFrame:
    """Returns columns [value, is_provisional, nowcast_model, as_of, lower_80, upper_80].
    Provisional rows are appended after the last published quarter."""

def reconcile(extended: pd.DataFrame, new_reported: pd.Series) -> pd.DataFrame:
    """Overwrite provisional rows once official values are published; keep a log of
    the nowcast error for each reconciled quarter (feeds the empirical intervals)."""
```

- The extended series can be passed to the existing `FrequencyPipeline` so monthly/daily upsampled series extend to the present. Add an `allow_provisional: bool = False` flag to the pipeline entry point (the only permitted change to existing code); when true, provisional quarters are used, and every downstream output carries an `is_provisional` marker.
- Provisional quarters must never be used to *estimate* desmoothing or factor parameters — only to extend the series after estimation. Enforce this in code.

---

## Library Placement

```
private_assets_frequency/
└── nowcast/
    ├── __init__.py
    ├── DESIGN_NOTES.md
    ├── information_set.py       # InformationSet, build_information_set, vintage handling
    ├── base.py                  # Nowcaster protocol, NowcastResult
    ├── models/
    │   ├── naive.py
    │   ├── smoothing_regression.py
    │   ├── structural.py
    │   └── signature.py
    ├── signatures.py            # path construction, compute_signature adapter, numpy backend
    ├── uncertainty.py           # empirical / parametric / recursive simulation
    ├── evaluation.py            # rolling-origin harness, DM test, plots
    └── integration.py           # extend_with_nowcasts, reconcile
tests/nowcast/
    ├── conftest.py              # synthetic generators (below)
    ├── test_information_set.py
    ├── test_signatures.py
    ├── test_models.py
    ├── test_uncertainty.py
    ├── test_evaluation.py
    └── test_integration.py
```

Dependencies: the parent library is numpy/scipy/pandas/statsmodels only. This module adds `scikit-learn` (required for the nowcast extra) and `iisignature` / `esig` (optional). Declare them as an optional extra in `pyproject.toml` (`pip install private_assets_frequency[nowcast]`). The package must import and run Models 0–2 without these extras installed.

---

## Synthetic Data Generators (`tests/nowcast/conftest.py`)

1. **Linear smoothing DGP:** daily public factor → true quarterly return `r_Q = α + β·F_Q + ε_Q` → reported `s_Q = (1−λ)·r_Q + λ·s_{Q−1}` with known λ, β. Add a publication lag and a revision process (first print = final value + noise that decays over 3 releases) to produce a vintage table.
2. **Path-dependent DGP:** same, but the appraiser marks against the **average** public level over the quarter (or reacts more to moves in the last month than the first) rather than the quarter-end return. This is a DGP where within-quarter path information genuinely matters.
3. **Early-signal DGP:** add an irregular "listed PE NAV" series correlated with the true return and released at random dates.

---

## Tests (must all pass)

1. **Leakage test (most important):** for any model, perturbing any public or reported data dated after `as_of` (and any vintage with `release_date > as_of`) leaves predictions bit-for-bit unchanged.
2. **Vintage test:** the value seen for a quarter at `as_of` is the latest release ≤ `as_of`; unpublished quarters are correctly identified as horizons 1..H.
3. **Signature backend test:** numpy backend matches `iisignature` (if installed) to 1e-10 for random piecewise-linear paths at levels 1–3; Chen's identity holds (signature of concatenation = tensor product).
4. **Nesting test:** `SignatureNowcaster` with `level=1`, one factor channel + time, no regularization, gives the same predictions as `SmoothingRegressionNowcaster` with that factor (to numerical tolerance, after accounting for the time term being constant across same-length windows).
5. **Recovery test:** on the linear DGP, `SmoothingRegressionNowcaster` recovers `b ≈ (1−λ)β`, `c ≈ λ`, and `StructuralSmoothingNowcaster` matches it when given the true parameters.
6. **Path-dependence test:** on the path-dependent DGP, `SignatureNowcaster` (level 2) beats `SmoothingRegressionNowcaster` out-of-sample; on the linear DGP it does **not** beat it by a significant margin (overfitting check).
7. **Coverage test:** on the linear DGP, empirical 80%/95% interval coverage over a long rolling-origin run is within ±5pp of nominal, for h=1 and h=2.
8. **Recursive horizon test:** h=2 intervals are wider than h=1 intervals.
9. **Feature budget test:** a configuration exceeding `n_train / 5` features is refused under `strict` and skipped with a warning under `warn`.
10. **Integration test:** `extend_with_nowcasts` → `FrequencyPipeline(allow_provisional=True)` runs end to end; provisional rows are flagged in all outputs; desmoothing parameters are identical with and without provisional rows; `reconcile` overwrites them correctly.

---

## Build Order (with checkpoints)

Stop at each **CHECKPOINT** and show the requested code before continuing.

1. `DESIGN_NOTES.md`.
2. `information_set.py` + `test_information_set.py` (leakage and vintage tests).
   **CHECKPOINT:** show `build_information_set` and how it decides which reported values are known at `as_of`.
3. `base.py`, `models/naive.py`, `models/smoothing_regression.py`, `uncertainty.py` (empirical intervals + recursive simulation) and their tests.
4. `models/structural.py`.
   **CHECKPOINT:** show how λ is converted from the Bayesian model's annual convention to the quarterly nowcast formula, with the derivation.
5. `signatures.py` (path construction, numpy backend, adapter) + `test_signatures.py`.
   **CHECKPOINT:** show the numpy signature implementation and the `keep_sigs='linear'` term-selection logic.
6. `models/signature.py` with feature budget guard + nesting test.
7. `evaluation.py` with nested hyperparameter selection, DM test, subperiod reports, plots.
   **CHECKPOINT:** show where the inner validation split is built and confirm the outer quarter cannot enter it.
8. `integration.py` + the `allow_provisional` flag + integration test.
9. Run the full test suite (including the parent library's tests) and fix failures.
10. `docs/nowcast_notes.md`: target definitions, information-set rules, the model hierarchy and its link to the paper's Kalman-nesting theorem, how to read the evaluation report, and known limitations (small sample, revisions, the reported vs true distinction).

---

## Critical Implementation Notes

1. **No look-ahead anywhere.** Scalers, PCA, hyperparameters, residual pools for intervals, and the factor model are all fit inside the information set. The leakage test is the acceptance criterion.
2. **Overlapping training rows are not independent.** If the evaluation grid creates several nowcasts per quarter, use only one row per quarter for fitting (the latest as-of date before publication), or use HAC-aware inference. Never treat within-quarter rows as extra independent observations.
3. **The smoothing regression is the benchmark, not a strawman.** If the signature model does not beat it out of sample on real data, report that plainly in the evaluation output — that is a valid and useful result.
4. **Compounding:** public factor quarterly returns are compounded from daily/monthly (`Π(1+r) − 1`), consistent with the rest of the library.
5. **Reproducibility:** every random operation takes an `rng`.
6. **Model selection for this build:** use Opus for steps 2, 4, 5 and 7 (information set, structural conversion, signatures, nested evaluation); Sonnet is fine for steps 3, 8 and 10.
