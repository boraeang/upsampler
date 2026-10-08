# Nowcast module — design notes

Written before any code in `nowcast/`, per the module spec. The question this
document answers is narrow and specific:

> **In what ways could this module silently report validation results that are
> better than what a user would actually have obtained in real time?**

"Silently" is the operative word. A nowcaster that is simply bad is harmless —
the evaluation harness will say so. A nowcaster that *looks* good because the
harness fed it information it could not have had is actively dangerous, because
it will be trusted. Everything below is about that failure class.

---

## Risk 1 — Look-ahead leakage through the fitting path, not the feature path

**The failure.** Everybody remembers not to put `F_{Q+1}` in the feature vector
for quarter `Q`. Almost nobody remembers that the *scaler*, the *PCA rotation*,
the *hyperparameter choice*, the *residual pool used for prediction intervals*,
and the *desmoothing/factor parameters borrowed from Stage 1* are all estimated
quantities too, and every one of them leaks the whole sample if fitted once
globally and reused across evaluation dates.

This is the leakage that survives code review, because the feature matrix looks
clean. Concretely, the ways it gets in here:

- `StandardScaler` fit on all rows, then used to transform training rows — the
  test quarter's mean and variance are in the scaler.
- PCA of the factor panel fitted on the full history before the rolling loop.
- `level`, `lookback`, `alpha`, `l1_ratio`, `q` chosen once by looking at the
  whole-sample out-of-sample error, then "validated" on that same sample.
- Empirical prediction intervals built from residuals that include the quarter
  being predicted.
- `StructuralSmoothingNowcaster` handed a `DesmoothedResult` fitted on the full
  sample — including the target quarter's reported value.

**Prevention in this implementation.**

1. `InformationSet` is the only object a model ever sees, and it is built by
   `build_information_set(...)`, which truncates **at construction** and stores
   copies. There is no code path by which a model reaches untruncated data,
   because the untruncated frames are never attributes of the object. The
   evaluation harness owns the full data; models own an `InformationSet`.
2. Every model's `fit(info)` re-fits *everything* from `info`: scaler, PCA,
   coefficients, residual pool. Nothing is cached across `as_of` dates. This is
   wasteful and deliberately so — the cost is seconds, the alternative is a
   wrong answer.
3. Hyperparameter selection lives inside `fit`, on an **inner** rolling-origin
   split over published quarters only (see Risk 3).
4. `StructuralSmoothingNowcaster` does not accept a globally-fitted
   `DesmoothedResult` silently. It records the index span of the result it was
   given in its diagnostics, and the evaluation harness re-fits the desmoothing
   model inside each information set. Passing a full-sample result is allowed
   for a *production* nowcast at today's `as_of` (where there is no later data
   to leak) but is flagged in `diagnostics['structural_fit_span']` so that a
   leaky backtest is visible in the output rather than invisible.
5. **Test 1 (`test_information_set.py::test_no_leakage_*`) is the acceptance
   criterion, not this document.** It perturbs every post-`as_of` observation —
   public factors, reported values, and vintage rows with
   `release_date > as_of` — and asserts predictions are bit-for-bit identical.
   `np.array_equal` on the draws, not `allclose`: any dependence at all shows up.

---

## Risk 2 — Revision and vintage effects

**The failure.** A private markets index print is not a fact, it is an estimate
that gets revised for several quarters as late-reporting funds arrive. If the
harness trains on the *final revised* series and scores against the *final
revised* series, it is doing two things wrong at once:

- The lagged predictor `s_{Q−1}` is the revised value, which in real time did
  not exist yet. The model is conditioning on the future.
- The target is the revised value, which is a partially-smoothed average of
  information that arrived after the nowcast date. Revisions are typically
  mean-reverting toward the truth, so the revised series is *easier to predict*
  than the first print. RMSE against revised data flatters every model.

The second effect is the subtle one, and it is large for PE: the first print of
a quarter reflects maybe 60–80 % of fund NAVs, and the gap closes over the next
two to three releases.

**Prevention in this implementation.**

1. **Vintage mode is the supported path.** `vintages` is a long frame
   `[quarter, release_date, value]` with one row per published value including
   revisions. The value known at `as_of` for quarter `Q` is the latest row with
   `release_date <= as_of`. Lagged predictors therefore carry the vintage the
   model would actually have seen.
2. `evaluation_target ∈ {'first_print', 'latest'}` is explicit and defaults to
   `'first_print'` in vintage mode. Scoring against `'latest'` is available
   because it answers a different, legitimate question ("how close is the
   nowcast to the truth as we now understand it"), but it is never the default
   and the evaluation report labels which was used.
3. **Lag-rule mode warns, loudly and by default.** If the user supplies only a
   final revised series plus a `publication_lag`, `build_information_set` emits
   a `NowcastDataModeWarning` stating that validation is pseudo-real-time and
   likely optimistic. The warning text names the mechanism, not just the mode,
   so a user reading a log understands why they should care. Under
   `FallbackPolicy.STRICT` it escalates to an exception, because a strict run
   asking for a defensible number should not get a pseudo-real-time one.
4. The information set records `data_mode` in `diagnostics`, and every
   `NowcastResult` inherits it, so a results table always carries the provenance
   of its own honesty.

**Residual risk we accept and document:** vintage tables for private markets
indices are hard to obtain and often start later than the index itself. Most
users will run in lag-rule mode. The module cannot fix that; it can only refuse
to let it pass unremarked.

---

## Risk 3 — Hyperparameter selection that touches the evaluation quarter

**The failure.** The signature nowcaster has five hyperparameters (`level`,
`lookback`, `keep_sigs`, `alpha`, `l1_ratio`) and roughly 90 quarterly
observations. That ratio is the whole problem. Select those five by minimising
error over the evaluation sample and the reported RMSE is a *selection*
statistic, not a forecast statistic — it is the minimum of many random
variables, and its expectation is below the truth by an amount that grows with
the number of configurations tried. With a grid of a few dozen configurations
and 90 points, the bias is comfortably large enough to manufacture a
"signatures beat the benchmark" result out of pure noise.

The same mechanism, in a quieter form, applies to `q` in the smoothing
regression and to the choice of which model to present as the headline.

A second, related failure: **overlapping rows**. The harness supports a grid of
several `as_of` dates per quarter, to show how a nowcast evolves as data
arrives. Those rows share a target. Treating them as independent training
observations inflates the effective sample size, shrinks standard errors, and
lets the regularisation path pick a much more complex model than the data
supports.

**Prevention in this implementation.**

1. **Nested selection.** Selection happens inside `fit(info)`, on an inner
   rolling-origin split over the last `inner_validation_quarters` (default 12)
   **published** quarters of the information set. The outer evaluation quarter
   is, by construction, not published at `as_of` — it is not in `info.reported`
   at all — so it cannot enter the inner split. This is stated as an assertion
   in `evaluation.py`, not merely as a comment: the harness checks that the
   outer target quarter is absent from the inner selection index and raises if
   it is not.
2. **The grid is small and defaults are conservative.** `level=2`,
   `lookback=1`, `keep_sigs='linear'`, basepoint on. Level 4 is not offered by
   default. The point of a default is to be the thing you report when you have
   not earned the right to tune.
3. **A hard feature budget.** `n_features > n_train / 5` is refused under
   `FallbackPolicy.STRICT` and skipped with a warning under `WARN`/`AUTO`. The
   computed budget appears in `diagnostics` of every fitted model. With ~90
   training quarters this caps the configuration at time plus one or two
   channels at level 2, which is the honest capacity of the sample.
4. **One row per quarter for fitting.** When the evaluation grid produces
   several `as_of` dates per quarter, only one row per quarter enters the
   training matrix (the latest `as_of` before publication). The multi-date grid
   is a *reporting* device for showing nowcast evolution, never an
   observation-count multiplier. Inference on the evaluation metrics uses HAC
   variance (Diebold-Mariano with a Newey-West kernel) for the same reason.
5. **The benchmark is honest.** `SmoothingRegressionNowcaster` is the reduced
   form of the library's own AR(1)/Rudin smoothing models — the thing a careful
   practitioner would actually do. It gets the same nested selection treatment
   as the challenger. If signatures do not beat it, the evaluation output says
   so in plain words; that is a result, not a bug.

---

## Honourable mention — two further ways to mislead, both guarded

**`target='true'` is not a nowcast.** The unsmoothed economic return has a
predictable part (`α + β·F_Q`) and an idiosyncratic part with expectation zero
and variance that no amount of public data reduces. An R² on `target='true'`
will look poor and *should*. It is implemented as a thin wrapper over the fitted
factor decomposition, and its docstring and diagnostics both say that the
interval is dominated by irreducible idiosyncratic risk. Reporting
`target='true'` accuracy next to `target='reported'` accuracy without that
caveat would be comparing two different questions.

**Provisional values contaminating parameter estimates.** Once nowcasts are
appended to the reported series and handed to `FrequencyPipeline`, there is an
obvious temptation for the desmoothing and factor estimation to use them. That
would be circular: the nowcast was produced *from* a factor model, so feeding it
back in would shrink the estimated λ and inflate R² for no informational reason.
`integration.py` marks provisional rows with `is_provisional`, and the pipeline's
`allow_provisional=True` path estimates all parameters on published rows only,
using provisional rows solely to extend the series afterwards. Integration test
10 asserts the desmoothing parameters are *identical* with and without
provisional rows — not close, identical.

---

## What the module does not claim

- It does not claim ~90 quarterly observations can support a richly
  parameterised path model. The feature budget exists because they cannot.
- It does not claim the signature features will help. The reference paper's
  Theorem 1 says signature regression *contains* the linear/Kalman benchmark as
  a special case, which guarantees representational nesting, not out-of-sample
  improvement at this sample size.
- It does not claim prediction intervals are calibrated on real data. They are
  calibrated on the synthetic DGPs (test 7, ±5pp of nominal). Real data has
  revisions, regime breaks, and a fat left tail that 90 observations cannot
  characterise.
