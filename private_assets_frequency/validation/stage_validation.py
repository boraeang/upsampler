r"""
Inter-stage validation gates.

Per Critical Implementation Note 7 of the spec, every stage must validate
its outputs before passing them downstream:

    * **Stage 1 → Stage 2** — desmoothed σ must exceed observed σ; otherwise
      the λ estimate is suspect (the smoothing transform should *increase*
      volatility because it undoes the smoothing).
    * **Stage 2 → Stage 3** — residuals must be approximately uncorrelated
      with the factor returns; otherwise the factor model is misspecified
      and the disaggregation will lean on a bad indicator.
    * **Stage 3 → Stage 4** — high-frequency returns must compound to
      low-frequency within tolerance ``1e-10`` (the multiplicative
      aggregation constraint).

Each function returns a :class:`StageValidationResult` so that the pipeline
runner can aggregate them and the :class:`FallbackHandler` can react to
failures uniformly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.protocols import (
    AggregationType,
    StageValidationResult,
    ValidationStatus,
)
from ..utils.returns import (
    aggregate_returns,
    aggregate_returns_by_blocks,
    realised_volatility,
)


def validate_stage_1_to_2(
    observed: pd.Series,
    desmoothed: pd.Series,
    *,
    tolerance: float = 0.95,
) -> StageValidationResult:
    """Desmoothed volatility should exceed observed volatility.

    Parameters
    ----------
    observed
        Smoothed observed return series (input to Stage 1).
    desmoothed
        Desmoothed return series (output of Stage 1).
    tolerance
        Allow the desmoothed σ to be no smaller than ``tolerance × σ_obs``
        before flagging. Default 0.95 absorbs small finite-sample noise.
    """
    sigma_obs = float(realised_volatility(observed))
    sigma_des = float(realised_volatility(desmoothed))
    increased = sigma_des > tolerance * sigma_obs
    status = ValidationStatus.PASS if increased else ValidationStatus.WARN
    msgs: list[str] = []
    if not increased:
        msgs.append(
            f"Stage 1 → 2: desmoothed σ {sigma_des:.4f} ≤ {tolerance:.0%} × observed σ "
            f"{sigma_obs:.4f}; λ estimate suspect."
        )
    return StageValidationResult(
        stage="Stage 1 → 2 (volatility increase)",
        status=status,
        checks={"vol_increased": increased},
        messages=msgs,
        metadata={
            "sigma_observed": sigma_obs,
            "sigma_desmoothed": sigma_des,
            "vol_ratio": sigma_des / sigma_obs if sigma_obs > 0 else float("nan"),
        },
    )


def validate_stage_2_to_3(
    residuals: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    correlation_threshold: float = 0.20,
) -> StageValidationResult:
    """Stage-2 residuals must be approximately uncorrelated with the factors.

    For OLS / WLS this is a tautology in-sample; for factor models that
    were fit *outside* the residual extraction (e.g. desmoother's joint
    fit), the residual / factor correlation is a real check.

    Parameters
    ----------
    residuals
        Stage 2 residual series.
    factor_returns
        Factor returns aligned to ``residuals.index``.
    correlation_threshold
        Maximum absolute correlation allowed before warning. Default 0.20
        — generous because OLS guarantees orthogonality only in-sample.
    """
    if not isinstance(residuals, pd.Series):
        raise TypeError("residuals must be a Series.")
    if not isinstance(factor_returns, pd.DataFrame):
        raise TypeError("factor_returns must be a DataFrame.")
    factor_returns = factor_returns.reindex(residuals.index)
    if factor_returns.isna().any().any():
        raise ValueError("factor_returns has NaN after alignment to residuals.")
    correlations: dict[str, float] = {}
    over_threshold: list[str] = []
    for col in factor_returns.columns:
        corr = float(np.corrcoef(residuals.to_numpy(), factor_returns[col].to_numpy())[0, 1])
        correlations[col] = corr
        if abs(corr) > correlation_threshold:
            over_threshold.append(f"{col} (|ρ|={abs(corr):.3f})")
    passed = not over_threshold
    msgs = (
        []
        if passed
        else [
            f"Stage 2 → 3: residuals correlated with factor(s) "
            f"{over_threshold!r} above {correlation_threshold:.2f} threshold; "
            "factor model may be misspecified."
        ]
    )
    return StageValidationResult(
        stage="Stage 2 → 3 (residual orthogonality)",
        status=ValidationStatus.PASS if passed else ValidationStatus.WARN,
        checks={"residuals_orthogonal": passed},
        messages=msgs,
        metadata={"correlations": correlations, "threshold": correlation_threshold},
    )


def validate_stage_3_to_4(
    low_frequency: pd.Series,
    high_frequency: pd.Series,
    ratio: int | None = None,
    *,
    block_sizes: np.ndarray | list[int] | None = None,
    aggregation: AggregationType | str = AggregationType.MULTIPLICATIVE,
    tolerance: float = 1e-10,
) -> StageValidationResult:
    """High-frequency series must aggregate to low-frequency within ``tolerance``.

    This is the spec's tightest gate: a failure here means downstream Stage 4
    cannot be trusted to preserve the input low-frequency cumulative return.

    Provide exactly one of ``ratio`` (uniform blocks, e.g. 3 for
    quarterly→monthly) or ``block_sizes`` (irregular blocks, e.g. business days
    per month for monthly→daily).
    """
    if (ratio is None) == (block_sizes is None):
        raise ValueError("provide exactly one of `ratio` or `block_sizes`.")
    if block_sizes is not None:
        aggregated = aggregate_returns_by_blocks(
            high_frequency, block_sizes, method=aggregation
        )
    else:
        aggregated = aggregate_returns(high_frequency, ratio=ratio, method=aggregation)
    if isinstance(aggregated, pd.Series):
        agg_arr = aggregated.to_numpy()
    else:
        agg_arr = np.asarray(aggregated)
    lf_arr = low_frequency.to_numpy()
    if agg_arr.shape != lf_arr.shape:
        return StageValidationResult(
            stage="Stage 3 → 4 (round-trip aggregation)",
            status=ValidationStatus.FAIL,
            checks={"shape_match": False},
            messages=[
                f"Stage 3 → 4: shape mismatch — aggregated has shape {agg_arr.shape}, "
                f"low-freq has shape {lf_arr.shape}."
            ],
        )
    err = float(np.max(np.abs(agg_arr - lf_arr)))
    passed = err < tolerance
    status = ValidationStatus.PASS if passed else ValidationStatus.FAIL
    msgs = (
        []
        if passed
        else [
            f"Stage 3 → 4: max |round-trip error| = {err:.3e} > tolerance {tolerance:.1e}."
        ]
    )
    return StageValidationResult(
        stage="Stage 3 → 4 (round-trip aggregation)",
        status=status,
        checks={"round_trip_within_tol": passed},
        messages=msgs,
        metadata={
            "max_error": err,
            "tolerance": tolerance,
            "ratio": int(ratio) if ratio is not None else None,
            "block_sizes": (
                [int(s) for s in np.asarray(block_sizes, dtype=int)]
                if block_sizes is not None
                else None
            ),
        },
    )


__all__ = [
    "validate_stage_1_to_2",
    "validate_stage_2_to_3",
    "validate_stage_3_to_4",
]
