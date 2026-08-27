r"""
End-to-end pipeline orchestration.

The :class:`FrequencyPipeline` class wires Stages 0 → 4 across all
asset-class smoothing models, returning a single :class:`PipelineResult`
with the disaggregated monthly / daily series and every per-stage
intermediate. Per-strategy execution is independent so a panel of mixed
asset classes (PE + HF + credit) runs in one call.

High-level flow per strategy
----------------------------

    Stage 0 (preprocessing)         — ``carry_mtm_decomposition`` (credit) or
                                      ``reporting_lag_adjustment`` (HF), or pass-through.
    Stage 1 (desmoothing)           — chosen by ``config.smoothing_model``.
    Stage 2 (factor decomposition)  — :class:`FactorModel` on the desmoothed
                                      series (re-runs even if Stage 1 had its
                                      own factor regression — keeps the
                                      output consistent across models).
    Stage 3 (temporal disaggregation) — :class:`ChowLinDisaggregator` to the
                                      target frequency: quarterly→monthly for
                                      PE / RE / infra / credit, monthly→daily
                                      for HF.
    Stage 4 (daily extension)       — :class:`KalmanDailySmoother` if daily
                                      factors are supplied for quarterly-native
                                      strategies. Skipped for HF (already
                                      daily after Stage 3).

Validation gates and sanity anchors are run at the end of each stage and
collected on the result; the :class:`FallbackHandler` decides whether to
escalate, warn, or silently auto-fix.

Limitations
-----------
* Daily Stage 3 / Stage 4 disaggregation requires a business-day calendar
  and uses the irregular-block aggregation matrix from :mod:`daily.calendar`.
  For PE-style two-step quarterly→monthly→daily, Stage 3 produces monthly,
  Stage 4 produces daily.
* ``uncertainty_mode='full'`` is recognised but currently emits a warning
  and falls back to point-estimate mode; full posterior propagation is a
  follow-up. The hooks are in place — see ``_run_uncertainty_band`` stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import (
    AssetClassConfig,
    BetaDist,
    FallbackPolicy,
    InverseGammaPrior,
    NormalPrior,
)
from ..core.protocols import (
    AggregationType,
    DesmoothedResult,
    DisaggregationMethod,
    Frequency,
    PreprocessingKind,
    SmoothingModel,
    SmoothingModelKind,
    StageValidationResult,
    UncertaintyMode,
    ValidationStatus,
)
from ..daily.kalman import DailyDisaggregationResult, KalmanDailySmoother
from ..decomposition.factor_model import FactorModel, FactorRegressionResult
from ..desmoothing.ar1_bayesian import AR1BayesianSmoother
from ..desmoothing.geltner_classic import GeltnerClassicSmoother
from ..desmoothing.ma_glm import MAGLMSmoother
from ..desmoothing.no_smoothing import NoSmoothing
from ..desmoothing.okunev_white import OkunevWhiteSmoothing
from ..desmoothing.rudin_reparam import RudinReparamSmoothing
from ..desmoothing.threshold_ar1 import ThresholdAR1Smoother
from ..disaggregation.chow_lin import ChowLinDisaggregator, DisaggregationResult
from ..preprocessing.credit_decomposition import (
    decompose_credit_return,
    reattach_carry,
)
from ..preprocessing.reporting_lag import adjust_reporting_lag
from ..utils.returns import realised_volatility
from ..validation.diagnostics import DiagnosticBundle, run_residual_diagnostics
from ..validation.sanity_anchors import SanityAnchorResult, run_all_anchors
from ..validation.stage_validation import (
    validate_stage_1_to_2,
    validate_stage_2_to_3,
    validate_stage_3_to_4,
)
from .fallback import FallbackHandler


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategyResult:
    """Per-strategy intermediate results for one pipeline run."""

    config: AssetClassConfig
    desmoothed: DesmoothedResult
    factor_regression: FactorRegressionResult
    disaggregation: DisaggregationResult
    daily: DailyDisaggregationResult | None
    diagnostics: DiagnosticBundle
    sanity_anchors: dict[str, SanityAnchorResult] = field(default_factory=dict)


@dataclass(frozen=True)
class UncertaintyBands:
    """Posterior percentile bands across the disaggregated output.

    Produced when ``uncertainty_mode='full'`` is requested on the pipeline.

    Attributes
    ----------
    percentiles
        Tuple of percentile probabilities used (default ``(5, 25, 50, 75, 95)``).
    monthly
        Mapping ``{percentile: DataFrame}`` of monthly returns at each
        percentile. Each DataFrame has the same shape and columns as
        ``PipelineResult.monthly_returns``.
    daily
        Mapping ``{percentile: DataFrame}`` of daily returns, or ``None``
        if Stage 4 wasn't requested.
    n_samples
        Number of posterior samples drawn per strategy.
    diagnostics
        Free-form per-strategy diagnostics (sample acceptance counts, etc.).
    """

    percentiles: tuple[int, ...]
    monthly: dict[int, pd.DataFrame]
    daily: dict[int, pd.DataFrame] | None
    n_samples: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end pipeline output.

    Attributes
    ----------
    monthly_returns
        Wide DataFrame of monthly returns per strategy (or daily for HF
        runs whose target was daily directly).
    daily_returns
        Wide DataFrame of daily returns per strategy if Stage 4 ran;
        ``None`` otherwise.
    per_strategy
        Mapping ``{strategy_name: StrategyResult}``.
    warnings
        Free-form warning messages accumulated by the
        :class:`FallbackHandler`.
    stage_validations
        Inter-stage :class:`StageValidationResult` objects.
    sanity_anchors
        Mapping ``{strategy_name: {anchor_name: SanityAnchorResult}}``.
    diagnostics
        Free-form mapping of pipeline-level diagnostics (timings, etc.).
    """

    monthly_returns: pd.DataFrame
    daily_returns: pd.DataFrame | None
    per_strategy: dict[str, StrategyResult]
    warnings: list[str]
    stage_validations: list[StageValidationResult]
    sanity_anchors: dict[str, dict[str, SanityAnchorResult]]
    uncertainty_bands: "UncertaintyBands | None" = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Smoother factory
# ──────────────────────────────────────────────────────────────────


def _make_smoother(config: AssetClassConfig) -> SmoothingModel:
    kind = config.smoothing_kind
    if kind is SmoothingModelKind.AR1_BAYESIAN:
        return AR1BayesianSmoother()
    if kind is SmoothingModelKind.MA_GLM:
        return MAGLMSmoother(q=int(config.ma_lags))
    if kind is SmoothingModelKind.THRESHOLD_AR1:
        return ThresholdAR1Smoother()
    if kind is SmoothingModelKind.NO_SMOOTHING:
        return NoSmoothing()
    if kind is SmoothingModelKind.GELTNER_CLASSIC:
        return GeltnerClassicSmoother()
    if kind is SmoothingModelKind.RUDIN_REPARAM:
        # Reuse ma_lags as the Q selector when supplied; else default to 1
        # (the paper's preferred order and also the RudinReparamSmoothing default).
        n_lags = int(config.ma_lags) if config.ma_lags is not None else 1
        return RudinReparamSmoothing(n_lags=n_lags)
    if kind is SmoothingModelKind.OKUNEV_WHITE:
        # Pure time-series unsmoother (no factors). Reuse ma_lags as the depth
        # selector when supplied; else default to 4 (the paper's value).
        n_lags = int(config.ma_lags) if config.ma_lags is not None else 4
        return OkunevWhiteSmoothing(n_lags=n_lags)
    raise ValueError(f"unknown smoothing model {kind!r}")


def _build_priors(
    config: AssetClassConfig,
    *,
    regime_indicator: pd.Series | None = None,
) -> dict[str, Any]:
    """Translate a config into the priors dict each smoother expects."""
    alpha_prior = config.alpha_prior or NormalPrior(0.0, 0.05)
    sigma_prior = config.sigma_eps_prior or InverseGammaPrior(3.0, 0.02)
    kind = config.smoothing_kind
    if kind is SmoothingModelKind.AR1_BAYESIAN:
        return {
            "lambda": config.lambda_prior or BetaDist(2.0, 2.0),
            "beta": config.beta_priors,
            "alpha": alpha_prior,
            "sigma_eps": sigma_prior,
        }
    if kind is SmoothingModelKind.MA_GLM:
        return {
            "beta": config.beta_priors,
            "alpha": alpha_prior,
            "sigma_eps": sigma_prior,
        }
    if kind is SmoothingModelKind.THRESHOLD_AR1:
        if regime_indicator is None:
            raise ValueError(
                "threshold_ar1 requires a regime_indicator on the pipeline call."
            )
        return {
            "lambda_normal": config.lambda_prior_normal or BetaDist(5.0, 2.0),
            "lambda_stress": config.lambda_prior_stress or BetaDist(2.0, 5.0),
            "regime_indicator": regime_indicator,
            "beta": config.beta_priors,
            "alpha": alpha_prior,
            "sigma_eps": sigma_prior,
        }
    return {}


# ──────────────────────────────────────────────────────────────────
# FrequencyPipeline
# ──────────────────────────────────────────────────────────────────


@dataclass
class FrequencyPipeline:
    """End-to-end frequency upsampling pipeline.

    Parameters
    ----------
    returns
        Wide DataFrame of native-frequency returns; one column per strategy.
    factor_returns_monthly
        Monthly factor returns (used as Stage 3 indicators when native is
        quarterly).
    factor_returns_daily
        Daily factor returns (used as Stage 3 indicators for HF runs and
        as Stage 4 systematic drivers for two-step quarterly→daily).
    configs
        Mapping ``{strategy_name: AssetClassConfig}``. Strategy names must
        match ``returns.columns``.
    disaggregation_method
        ``'chow_lin'`` (default) | ``'fernandez'`` | ``'litterman'``.
    aggregation_type
        ``'multiplicative'`` (default) | ``'additive'``.
    fallback_policy
        ``'warn'`` (default) | ``'strict'`` | ``'auto'``.
    uncertainty_mode
        ``'point'`` (default) | ``'full'`` (currently warns and falls back).
    yield_series
        Optional wide DataFrame of running yields per strategy at the
        native frequency. Required for credit strategies that use
        ``carry_mtm_decomposition``.
    regime_indicator
        Optional Series (or wide DataFrame keyed by strategy) of binary
        regime labels at the native frequency for threshold-AR(1) credit.
    """

    returns: pd.DataFrame
    configs: dict[str, AssetClassConfig]
    factor_returns_monthly: pd.DataFrame | None = None
    factor_returns_daily: pd.DataFrame | None = None
    disaggregation_method: DisaggregationMethod | str = DisaggregationMethod.CHOW_LIN
    aggregation_type: AggregationType | str = AggregationType.MULTIPLICATIVE
    fallback_policy: FallbackPolicy | str = FallbackPolicy.WARN
    uncertainty_mode: UncertaintyMode | str = UncertaintyMode.POINT
    yield_series: pd.DataFrame | pd.Series | None = None
    regime_indicator: pd.DataFrame | pd.Series | None = None
    public_proxies: dict[str, pd.Series] | None = None
    uncertainty_n_samples: int = 100
    uncertainty_percentiles: tuple[int, ...] = (5, 25, 50, 75, 95)
    uncertainty_seed: int | None = None

    def __post_init__(self) -> None:
        self.disaggregation_method = (
            DisaggregationMethod(self.disaggregation_method)
            if not isinstance(self.disaggregation_method, DisaggregationMethod)
            else self.disaggregation_method
        )
        self.aggregation_type = (
            AggregationType(self.aggregation_type)
            if not isinstance(self.aggregation_type, AggregationType)
            else self.aggregation_type
        )
        self.fallback_policy = (
            FallbackPolicy(self.fallback_policy)
            if not isinstance(self.fallback_policy, FallbackPolicy)
            else self.fallback_policy
        )
        self.uncertainty_mode = (
            UncertaintyMode(self.uncertainty_mode)
            if not isinstance(self.uncertainty_mode, UncertaintyMode)
            else self.uncertainty_mode
        )

        if not isinstance(self.returns, pd.DataFrame) or self.returns.empty:
            raise TypeError("returns must be a non-empty DataFrame.")
        missing_configs = [c for c in self.returns.columns if c not in self.configs]
        if missing_configs:
            raise KeyError(
                f"configs missing entries for strategies {missing_configs!r}"
            )

    # ── Public API ─────────────────────────────────────────────────

    def run(self) -> PipelineResult:
        """Execute Stage 0 → Stage 4 for every strategy and return aggregated results.

        Returns
        -------
        PipelineResult
            Wide ``monthly_returns`` and (optionally) ``daily_returns``
            DataFrames, per-strategy intermediates, accumulated warnings,
            inter-stage validation outcomes, sanity-anchor results, and —
            when ``uncertainty_mode='full'`` — posterior percentile bands.

        Raises
        ------
        Exception
            Any per-strategy failure propagates immediately; the
            :class:`FallbackHandler` does not swallow it. With
            ``fallback_policy='strict'`` even soft validation warnings
            escalate to :class:`PipelineFailure`.
        """
        handler = FallbackHandler(policy=self.fallback_policy)
        per_strategy: dict[str, StrategyResult] = {}
        monthly_panel: dict[str, pd.Series] = {}
        daily_panel: dict[str, pd.Series] = {}
        sanity_panel: dict[str, dict[str, SanityAnchorResult]] = {}

        for strategy_name in self.returns.columns:
            cfg = self.configs[strategy_name]
            strat = self._run_strategy(strategy_name, cfg, handler)
            per_strategy[strategy_name] = strat
            monthly_panel[strategy_name] = strat.disaggregation.high_frequency
            if strat.daily is not None:
                daily_panel[strategy_name] = strat.daily.daily_returns
            sanity_panel[strategy_name] = strat.sanity_anchors

        monthly_df = pd.DataFrame(monthly_panel)
        daily_df = (
            pd.DataFrame(daily_panel)
            if daily_panel and len(daily_panel) == len(self.returns.columns)
            else None
        )

        bands: UncertaintyBands | None = None
        if self.uncertainty_mode is UncertaintyMode.FULL:
            bands = self._compute_uncertainty_bands(per_strategy, handler)

        return PipelineResult(
            monthly_returns=monthly_df,
            daily_returns=daily_df,
            per_strategy=per_strategy,
            warnings=list(handler.accumulated_warnings),
            stage_validations=list(handler.accumulated_validations),
            sanity_anchors=sanity_panel,
            uncertainty_bands=bands,
            diagnostics={
                "n_strategies": len(per_strategy),
                "fallback_policy": self.fallback_policy.value,
                "disaggregation_method": self.disaggregation_method.value,
                "aggregation_type": self.aggregation_type.value,
                "uncertainty_mode": self.uncertainty_mode.value,
            },
        )

    # ── Per-strategy orchestration ─────────────────────────────────

    def _run_strategy(
        self,
        name: str,
        cfg: AssetClassConfig,
        handler: FallbackHandler,
    ) -> StrategyResult:
        observed = self.returns[name]
        if observed.isna().any():
            observed = observed.dropna()

        # ── Stage 0: preprocessing ─────────────────────────────────
        carry_native: pd.Series | None = None
        if cfg.preprocessing == PreprocessingKind.CARRY_MTM_DECOMPOSITION.value:
            yld = self._yield_for(name, observed.index)
            carry_native, mtm = decompose_credit_return(
                observed,
                yld,
                frequency=cfg.frequency,
            )
            stage1_input = mtm.rename(name)
        elif cfg.preprocessing == PreprocessingKind.REPORTING_LAG_ADJUSTMENT.value:
            stage1_input = adjust_reporting_lag(
                observed, lag_months=cfg.reporting_lag_months
            )
        else:
            stage1_input = observed

        # ── Stage 1: desmoothing ───────────────────────────────────
        smoother = _make_smoother(cfg)
        regime = self._regime_for(name, stage1_input.index, cfg)
        priors = _build_priors(cfg, regime_indicator=regime)
        factors_native = self._factors_at(cfg.frequency)
        if factors_native is None:
            raise ValueError(
                f"no factor returns supplied at native frequency "
                f"{cfg.frequency.value!r} for strategy {name!r}."
            )
        # Use only the columns referenced by the priors / config
        factor_cols = list(cfg.beta_priors)
        if cfg.smoothing_kind is SmoothingModelKind.NO_SMOOTHING:
            factor_cols = [c for c in factor_cols if c in factors_native.columns]
        missing_cols = [c for c in factor_cols if c not in factors_native.columns]
        if missing_cols:
            raise KeyError(
                f"strategy {name!r}: native-frequency factor returns missing "
                f"columns {missing_cols!r}."
            )
        F_native = factors_native[factor_cols].reindex(stage1_input.index)
        if F_native.isna().any().any():
            raise ValueError(
                f"strategy {name!r}: factor returns have NaN over the "
                "Stage 1 window."
            )
        desmoothed = smoother.fit(stage1_input, F_native, priors)
        handler.ingest_warnings(
            desmoothed.diagnostics.get("warnings", []),
            source=f"{name}/Stage1",
        )

        # ── Inter-stage 1 → 2 validation ──────────────────────────
        v12 = validate_stage_1_to_2(stage1_input, desmoothed.true_returns)
        handler.record_stage_validation(v12)

        # ── Stage 2: factor decomposition ─────────────────────────
        factor_model = FactorModel(method="ols")
        factor_fit = factor_model.fit(desmoothed.true_returns, F_native)
        v23 = validate_stage_2_to_3(factor_fit.residuals, F_native)
        handler.record_stage_validation(v23)

        # ── Stage 3: temporal disaggregation ──────────────────────
        # For PE / RE / infra / credit (quarterly native): produce monthly
        # using factor_returns_monthly. For HF (monthly native): produce
        # daily using factor_returns_daily.
        target_factors, target_index, ratio = self._stage3_target(name, cfg, factor_cols)
        # The Stage 3 disaggregator regresses on indicator columns; we only
        # use the columns referenced in the desmoothed factor model.
        ind_df = target_factors[factor_cols]
        # If carry was decomposed, disaggregate the desmoothed MTM only
        lf_for_disagg = desmoothed.true_returns
        disagg = ChowLinDisaggregator(
            method=self.disaggregation_method,
            aggregation=self.aggregation_type,
        ).fit(
            lf_for_disagg,
            ind_df,
            ratio=ratio,
            high_frequency_index=target_index,
        )
        # Inter-stage 3 → 4 validation runs against the raw Chow-Lin output
        # (i.e. on the desmoothed-MTM series for credit; on the desmoothed
        # total return for PE / RE / infra / HF). Carry reattachment is
        # arithmetic-additive within each block; combining it with the
        # multiplicative round-trip on MTM produces a non-zero round-trip
        # error for the total, which is *expected* and reported as an
        # additional diagnostic rather than a validation gate.
        v34 = validate_stage_3_to_4(
            disagg.low_frequency_input,
            disagg.high_frequency,
            ratio,
            aggregation=self.aggregation_type,
        )
        handler.record_stage_validation(v34)

        # Reattach carry post-validation if Stage 0 split it off
        if carry_native is not None:
            disagg_with_carry = reattach_carry(
                disagg.high_frequency,
                carry_native,
                high_freq_index=target_index,
            )
            # Recompute the "total return" round-trip error in additive space
            # (carry was distributed linearly).
            from ..utils.returns import aggregate_returns as _agg

            try:
                add_back = _agg(
                    disagg_with_carry, ratio=ratio, method=AggregationType.ADDITIVE
                )
                add_err = float(
                    np.max(np.abs(add_back.to_numpy() - observed.to_numpy()))
                )
            except Exception:
                add_err = float("nan")
            disagg = DisaggregationResult(
                high_frequency=disagg_with_carry,
                low_frequency_input=observed.copy(),
                method=disagg.method,
                aggregation=disagg.aggregation,
                rho=disagg.rho,
                betas=disagg.betas,
                sigma2=disagg.sigma2,
                log_likelihood=disagg.log_likelihood,
                aggregation_error=disagg.aggregation_error,
                diagnostics={
                    **disagg.diagnostics,
                    "carry_reattached": True,
                    "additive_total_round_trip_error": add_err,
                },
            )

        # ── Stage 4: optional daily extension ─────────────────────
        daily_result: DailyDisaggregationResult | None = None
        if (
            cfg.frequency is Frequency.QUARTERLY
            and self.factor_returns_daily is not None
        ):
            try:
                daily_result = self._run_stage4(
                    name=name,
                    monthly_returns=disagg.high_frequency,
                    factor_betas=factor_fit.betas,
                    alpha=factor_fit.alpha,
                )
            except Exception as exc:
                handler.warn(
                    f"Stage 4 (daily) failed for {name!r}: {exc}",
                    source=f"{name}/Stage4",
                )

        # ── Diagnostics & sanity anchors ──────────────────────────
        diag_bundle = run_residual_diagnostics(factor_fit.residuals)
        handler.ingest_warnings(diag_bundle.warnings, source=f"{name}/diagnostics")
        sanity = self._run_sanity(
            name=name,
            cfg=cfg,
            disagg=disagg,
            handler=handler,
        )

        return StrategyResult(
            config=cfg,
            desmoothed=desmoothed,
            factor_regression=factor_fit,
            disaggregation=disagg,
            daily=daily_result,
            diagnostics=diag_bundle,
            sanity_anchors=sanity,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _factors_at(self, frequency: Frequency) -> pd.DataFrame | None:
        if frequency is Frequency.QUARTERLY:
            # No quarterly factor table; we resample monthly→quarterly on demand.
            if self.factor_returns_monthly is None:
                return None
            from ..utils.returns import aggregate_returns

            arr = aggregate_returns(
                self.factor_returns_monthly,
                ratio=3,
                method=self.aggregation_type,
                trim="leading",
            )
            return arr  # already DataFrame
        if frequency is Frequency.MONTHLY:
            return self.factor_returns_monthly
        if frequency is Frequency.DAILY:
            return self.factor_returns_daily
        return None

    def _stage3_target(
        self,
        name: str,
        cfg: AssetClassConfig,
        factor_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex, int]:
        """Return ``(indicator_df, target_index, ratio)`` for Stage 3."""
        if cfg.frequency is Frequency.QUARTERLY:
            if self.factor_returns_monthly is None:
                raise ValueError(
                    f"strategy {name!r}: quarterly→monthly disaggregation "
                    "needs factor_returns_monthly."
                )
            target = self.factor_returns_monthly
            ratio = 3
        elif cfg.frequency is Frequency.MONTHLY:
            if self.factor_returns_daily is None:
                raise ValueError(
                    f"strategy {name!r}: monthly→daily disaggregation "
                    "needs factor_returns_daily."
                )
            target = self.factor_returns_daily
            # For monthly→daily Chow-Lin we need a *uniform* ratio. The
            # production path uses the irregular-block Stage 4 Kalman; for
            # Stage 3 of HF runs we approximate with ratio=21 and trim.
            ratio = 21
        else:
            raise ValueError(
                f"strategy {name!r}: unsupported native frequency "
                f"{cfg.frequency.value!r} for Stage 3."
            )
        # Ensure required columns are present
        missing = [c for c in factor_cols if c not in target.columns]
        if missing:
            raise KeyError(
                f"strategy {name!r}: target-frequency factor returns missing "
                f"columns {missing!r}."
            )
        # Trim to a multiple of ratio aligned to the native-frequency span
        n_target = len(self.returns) * ratio
        if len(target) < n_target:
            raise ValueError(
                f"strategy {name!r}: target factor returns have "
                f"{len(target)} rows; need at least {n_target}."
            )
        target = target.iloc[:n_target]
        return target, target.index, ratio

    def _run_stage4(
        self,
        *,
        name: str,
        monthly_returns: pd.Series,
        factor_betas: dict[str, float],
        alpha: float,
    ) -> DailyDisaggregationResult:
        if self.factor_returns_daily is None:
            raise ValueError("factor_returns_daily not supplied.")
        smoother = KalmanDailySmoother(
            phi=0.0, aggregation=self.aggregation_type
        )
        # Filter daily factors to columns we know betas for
        cols = [c for c in factor_betas if c in self.factor_returns_daily.columns]
        daily_F = self.factor_returns_daily[cols]
        if daily_F.empty:
            raise KeyError(
                f"strategy {name!r}: factor_returns_daily missing all of "
                f"{list(factor_betas)!r}"
            )
        daily_index = daily_F.index
        return smoother.fit(
            monthly_returns,
            daily_F,
            {k: factor_betas[k] for k in cols},
            alpha=alpha,
            daily_index=daily_index,
        )

    def _yield_for(self, name: str, index: pd.DatetimeIndex) -> pd.Series:
        if self.yield_series is None:
            raise ValueError(
                f"strategy {name!r}: yield_series required for "
                "carry_mtm_decomposition."
            )
        if isinstance(self.yield_series, pd.Series):
            return self.yield_series.reindex(index)
        if name in self.yield_series.columns:
            return self.yield_series[name].reindex(index)
        if len(self.yield_series.columns) == 1:
            return self.yield_series.iloc[:, 0].reindex(index)
        raise KeyError(
            f"strategy {name!r}: yield_series does not contain a column for it."
        )

    def _regime_for(
        self,
        name: str,
        index: pd.DatetimeIndex,
        cfg: AssetClassConfig,
    ) -> pd.Series | None:
        if cfg.smoothing_kind is not SmoothingModelKind.THRESHOLD_AR1:
            return None
        if self.regime_indicator is None:
            raise ValueError(
                f"strategy {name!r}: regime_indicator required for threshold_ar1."
            )
        if isinstance(self.regime_indicator, pd.Series):
            return self.regime_indicator.reindex(index)
        if name in self.regime_indicator.columns:
            return self.regime_indicator[name].reindex(index)
        if len(self.regime_indicator.columns) == 1:
            return self.regime_indicator.iloc[:, 0].reindex(index)
        raise KeyError(
            f"strategy {name!r}: regime_indicator does not contain a column for it."
        )

    # ── Uncertainty bands (uncertainty_mode='full') ───────────────

    def _compute_uncertainty_bands(
        self,
        per_strategy: dict[str, StrategyResult],
        handler: FallbackHandler,
    ) -> "UncertaintyBands | None":
        """Generate posterior percentile bands for ``uncertainty_mode='full'``.

        Per the spec (Stage 1 spec point 7): draw N samples of the
        smoothing parameters (and the regression weights conditional on
        them), re-run Stage 3 disaggregation for each draw, and report
        pointwise percentiles across the resulting monthly paths.
        """
        n = max(int(self.uncertainty_n_samples), 1)
        rng = np.random.default_rng(self.uncertainty_seed)
        eligible: dict[str, np.ndarray] = {}
        diagnostics: dict[str, Any] = {}

        for name, strat in per_strategy.items():
            cfg = strat.config
            kind = cfg.smoothing_kind
            # AR(1) Bayesian and MA(q) both expose sample_posterior; threshold-AR(1)
            # / NoSmoothing / Geltner-classic do not, so skip them.
            if kind not in (
                SmoothingModelKind.AR1_BAYESIAN,
                SmoothingModelKind.MA_GLM,
            ):
                handler.warn(
                    f"strategy {name!r}: smoothing model {kind.value!r} does not "
                    "yet support posterior sampling; skipping uncertainty bands.",
                    source=f"{name}/uncertainty",
                )
                continue
            try:
                paths = self._sample_strategy_paths(name, strat, n, rng)
            except Exception as exc:  # surface but don't kill the run
                handler.warn(
                    f"strategy {name!r}: posterior sampling failed: {exc}",
                    source=f"{name}/uncertainty",
                )
                continue
            eligible[name] = paths
            diagnostics[name] = {"n_paths": int(paths.shape[0])}

        if not eligible:
            return None

        first = next(iter(eligible.values()))
        n_high = first.shape[1]
        # Build percentile DataFrames from the per-strategy stack
        monthly_per_pct: dict[int, pd.DataFrame] = {}
        target_index = per_strategy[next(iter(eligible))].disaggregation.high_frequency.index
        for pct in self.uncertainty_percentiles:
            cols: dict[str, np.ndarray] = {}
            for name in self.returns.columns:
                if name in eligible:
                    cols[name] = np.percentile(
                        eligible[name], pct, axis=0
                    )
                else:
                    # Strategy didn't sample — use the point estimate
                    cols[name] = per_strategy[name].disaggregation.high_frequency.to_numpy()
            monthly_per_pct[int(pct)] = pd.DataFrame(
                cols, index=target_index, columns=list(self.returns.columns)
            )
        return UncertaintyBands(
            percentiles=tuple(int(p) for p in self.uncertainty_percentiles),
            monthly=monthly_per_pct,
            daily=None,  # Stage 4 propagation is a follow-up
            n_samples=n,
            diagnostics=diagnostics,
        )

    def _sample_strategy_paths(
        self,
        name: str,
        strat: StrategyResult,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw ``n_samples`` posterior re-disaggregations for one strategy.

        Returns an ``(n_samples, n_high)`` ndarray of monthly returns.
        """
        cfg = strat.config
        observed = self.returns[name].dropna()
        # Re-derive Stage 0 input identically to the point run
        carry_native: pd.Series | None = None
        if cfg.preprocessing == PreprocessingKind.CARRY_MTM_DECOMPOSITION.value:
            yld = self._yield_for(name, observed.index)
            carry_native, mtm = decompose_credit_return(
                observed, yld, frequency=cfg.frequency
            )
            stage1_input = mtm.rename(name)
        elif cfg.preprocessing == PreprocessingKind.REPORTING_LAG_ADJUSTMENT.value:
            stage1_input = adjust_reporting_lag(
                observed, lag_months=cfg.reporting_lag_months
            )
        else:
            stage1_input = observed

        # Posterior samples of smoothing params from the desmoother
        smoother = strat.desmoothed  # we need the live model; refit not necessary
        # The smoother instance is cached inside the desmoothed model fit;
        # re-build a fresh smoother and re-fit so we have a `_last_fit` cache
        # to sample from.
        live_smoother = _make_smoother(cfg)
        regime = self._regime_for(name, stage1_input.index, cfg)
        priors = _build_priors(cfg, regime_indicator=regime)
        factor_cols = list(cfg.beta_priors)
        factors_native = self._factors_at(cfg.frequency)
        F_native = factors_native[factor_cols].reindex(stage1_input.index)
        live_smoother.fit(stage1_input, F_native, priors)
        if not hasattr(live_smoother, "sample_posterior"):
            raise RuntimeError(
                f"smoothing model for {name!r} does not implement sample_posterior."
            )
        samples = live_smoother.sample_posterior(n_samples=n_samples, rng=rng)

        # For each sample, run desmoothing → Stage 3 disaggregation
        target_factors, target_index, ratio = self._stage3_target(name, cfg, factor_cols)
        ind_df = target_factors[factor_cols]
        s_arr = stage1_input.to_numpy(dtype=float)
        n_lf = len(s_arr)
        n_hf = ratio * n_lf
        out = np.empty((n_samples, n_hf))

        # The disaggregator only depends on the desmoothed *low-frequency*
        # series (it re-fits its own β internally). We re-run Chow-Lin per
        # sample with a fresh desmoothed series at the sampled λ / θ.
        disaggregator = ChowLinDisaggregator(
            method=self.disaggregation_method,
            aggregation=self.aggregation_type,
        )
        for i in range(n_samples):
            r_lf_arr = self._desmooth_at_sample(
                s_arr, samples, i, cfg.smoothing_kind, live_smoother
            )
            r_lf = pd.Series(r_lf_arr, index=stage1_input.index, name=name)
            disagg_i = disaggregator.fit(
                r_lf, ind_df, ratio=ratio, high_frequency_index=target_index
            )
            r_hf = disagg_i.high_frequency.to_numpy()
            if carry_native is not None:
                # Reattach carry per draw (linear within block)
                r_hf_full = reattach_carry(
                    pd.Series(r_hf, index=target_index, name=name),
                    carry_native,
                    high_freq_index=target_index,
                ).to_numpy()
                out[i] = r_hf_full
            else:
                out[i] = r_hf
        return out

    @staticmethod
    def _desmooth_at_sample(
        s: np.ndarray,
        samples: dict[str, np.ndarray],
        i: int,
        kind: SmoothingModelKind,
        smoother: SmoothingModel,
    ) -> np.ndarray:
        """Apply the desmoothing transform at the i-th posterior sample."""
        if kind is SmoothingModelKind.AR1_BAYESIAN:
            lam = float(samples["lambda"][i])
            return smoother.desmooth(s, np.array([lam]))
        if kind is SmoothingModelKind.MA_GLM:
            theta = samples["theta"][i]
            return smoother.desmooth(s, np.asarray(theta, dtype=float))
        raise ValueError(f"unsupported sampling kind {kind!r}")

    def _run_sanity(
        self,
        *,
        name: str,
        cfg: AssetClassConfig,
        disagg: DisaggregationResult,
        handler: FallbackHandler,
    ) -> dict[str, SanityAnchorResult]:
        if self.public_proxies is None or name not in self.public_proxies:
            return {}
        proxy = self.public_proxies[name]
        if cfg.frequency is Frequency.QUARTERLY:
            # Disaggregated output is at monthly (ratio 3); proxy assumed monthly
            periods = 12
        else:
            # Output is daily (ratio 21); proxy assumed daily
            periods = 252
        anchors = run_all_anchors(
            disagg.high_frequency,
            proxy,
            periods_per_year=periods,
        )
        for anchor in anchors.values():
            if not anchor.passed and not anchor.metadata.get("underpowered", False):
                handler.warn(
                    f"sanity anchor {anchor.name!r} failed: {anchor.message}",
                    source=f"{name}/sanity",
                )
        return anchors


__all__ = [
    "FrequencyPipeline",
    "PipelineResult",
    "StrategyResult",
    "UncertaintyBands",
]
