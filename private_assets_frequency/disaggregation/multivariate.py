r"""
Multi-strategy disaggregation that preserves cross-sectional residual structure.

The Stage 3 spec is explicit (point 4): when disaggregating multiple
strategies simultaneously, the cross-sectional correlation of the residuals
matters for downstream risk numbers. Running :class:`ChowLinDisaggregator`
independently per strategy is the obvious approach and gives an exact
aggregation round-trip on each strategy, but the resulting high-frequency
residuals are independent across strategies by construction — the
cross-sectional correlation collapses, which would falsely increase
diversification.

This module provides :class:`MultivariateDisaggregator` — a wrapper that:

    1. Disaggregates each strategy independently with the chosen method.
    2. Extracts the high-frequency residual panel
       ``ε_HF = y_HF − X_HF · β̂`` per strategy.
    3. Rotates the residual panel via a Cholesky transform so its sample
       covariance matches a target — by default the *low-frequency*
       residual covariance scaled to high-frequency.
    4. Re-imposes the aggregation constraint after the rotation, so the
       round-trip aggregation property is preserved.

The rotation uses the standard whitening + recolouring scheme:

    ε̃_HF = ε_HF · L_emp^{-1} · L_target

where ``L_emp`` and ``L_target`` are Cholesky factors of the empirical and
target covariance matrices respectively. After rotation, residuals are
projected back onto the aggregation null-space (via the same Chow-Lin
disaggregation operator applied to a zero LF residual) so that
``Σ ε̃_HF == 0`` block-wise — i.e. the aggregation constraint stays exact.

The rotation is equivalent to the spec's "marginal disaggregation with
copula adjustment" for Gaussian residuals; for non-Gaussian residuals the
rotation preserves linear correlation but not higher-order dependence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..core.protocols import AggregationType, CorrelationMethod, DisaggregationMethod
from ..utils.returns import aggregate_returns
from .aggregation import aggregation_matrix
from .chow_lin import ChowLinDisaggregator, DisaggregationResult


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MultivariateDisaggregationResult:
    """Output of :meth:`MultivariateDisaggregator.fit`.

    Attributes
    ----------
    high_frequency
        Wide DataFrame of disaggregated high-frequency returns; columns
        match the strategies in the input.
    per_strategy
        Mapping of strategy name to its individual :class:`DisaggregationResult`
        (pre-rotation).
    target_correlation
        Target cross-strategy correlation matrix (LF residuals scaled).
    realised_correlation
        Cross-strategy correlation of the disaggregated HF residuals after
        rotation. Should match the target up to sampling noise.
    aggregation_error
        Maximum absolute round-trip aggregation error across all strategies.
    diagnostics
        Free-form mapping.
    """

    high_frequency: pd.DataFrame
    per_strategy: dict[str, DisaggregationResult]
    target_correlation: pd.DataFrame
    realised_correlation: pd.DataFrame
    aggregation_error: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Disaggregator
# ──────────────────────────────────────────────────────────────────


@dataclass
class MultivariateDisaggregator:
    """Multi-strategy temporal disaggregator with cross-sectional rotation.

    Parameters
    ----------
    method
        Underlying univariate method (Chow-Lin / Fernández / Litterman).
    aggregation
        ``MULTIPLICATIVE`` (default) or ``ADDITIVE``.
    correlation_method
        ``EMPIRICAL`` (default) — use the LF residuals' sample covariance
        as the target. ``SHRINKAGE`` blends with a diagonal target.
    shrinkage_intensity
        Mixing weight when ``correlation_method=SHRINKAGE``.
    rotate_residuals
        If ``True`` (default), apply the cross-sectional rotation step.
        Set ``False`` to fall back to independent per-strategy disaggregation.
    """

    method: DisaggregationMethod | str = DisaggregationMethod.CHOW_LIN
    aggregation: AggregationType | str = AggregationType.MULTIPLICATIVE
    correlation_method: CorrelationMethod | str = CorrelationMethod.EMPIRICAL
    shrinkage_intensity: float = 0.1
    rotate_residuals: bool = True

    def __post_init__(self) -> None:
        self.method = (
            DisaggregationMethod(self.method)
            if not isinstance(self.method, DisaggregationMethod)
            else self.method
        )
        self.aggregation = (
            AggregationType(self.aggregation)
            if not isinstance(self.aggregation, AggregationType)
            else self.aggregation
        )
        self.correlation_method = (
            CorrelationMethod(self.correlation_method)
            if not isinstance(self.correlation_method, CorrelationMethod)
            else self.correlation_method
        )
        if not (0.0 <= self.shrinkage_intensity <= 1.0):
            raise ValueError(
                f"shrinkage_intensity must lie in [0, 1], got {self.shrinkage_intensity}"
            )

    def fit(
        self,
        low_frequency_panel: pd.DataFrame,
        high_frequency_indicators: dict[str, pd.DataFrame],
        ratio: int,
        *,
        high_frequency_index: pd.DatetimeIndex | None = None,
    ) -> MultivariateDisaggregationResult:
        """Disaggregate every strategy in ``low_frequency_panel`` jointly.

        Parameters
        ----------
        low_frequency_panel
            DataFrame with one column per strategy at the low frequency.
        high_frequency_indicators
            Mapping ``{strategy_name: high_freq_indicator_df}``. Each
            indicator DataFrame must have ``len(low_frequency_panel) * ratio``
            rows and is treated as that strategy's regressor matrix.
        ratio
            Number of HF periods per LF period.
        high_frequency_index
            Optional shared HF index. Defaults to the first strategy's
            indicator index — all strategies must share the same HF index.

        Returns
        -------
        MultivariateDisaggregationResult
        """
        self._validate(low_frequency_panel, high_frequency_indicators, ratio)
        strategies = list(low_frequency_panel.columns)

        # 1. Univariate disaggregation per strategy
        per_strategy: dict[str, DisaggregationResult] = {}
        for name in strategies:
            uni = ChowLinDisaggregator(
                method=self.method,
                aggregation=self.aggregation,
            )
            per_strategy[name] = uni.fit(
                low_frequency_panel[name],
                high_frequency_indicators[name],
                ratio=ratio,
                high_frequency_index=high_frequency_index,
            )

        # Extract HF residuals (in workspace space) per strategy. Workspace =
        # log-returns when aggregation is multiplicative, raw otherwise. We
        # rebuild the workspace residual = y_HF (workspace) − X_HF·β.
        hf_workspace, hf_residuals_workspace, hf_index = self._extract_hf_workspace(
            per_strategy, high_frequency_indicators, low_frequency_panel
        )
        # LF residuals in workspace
        lf_residuals_workspace = self._lf_residuals_workspace(
            per_strategy, high_frequency_indicators, low_frequency_panel, ratio
        )

        # 2. Build the target cross-strategy correlation
        target_corr_df = self._target_correlation(lf_residuals_workspace)

        # 3. Rotate within-block deviations; preserve block sums so the
        #    aggregation constraint is still exact. This decomposes ε_HF into
        #    block-mean (which is fixed by GLS / aggregation) plus
        #    within-block deviation (which we may rotate freely).
        if self.rotate_residuals and hf_residuals_workspace.shape[1] >= 2:
            rotated = self._rotate(
                hf_residuals_workspace, target_corr_df.to_numpy()
            )
            rotated = _match_block_means(rotated, hf_residuals_workspace, ratio)
            hf_workspace_rot = (
                hf_workspace - hf_residuals_workspace + rotated
            )
        else:
            hf_workspace_rot = hf_workspace

        # 5. Convert workspace back to return space
        if self.aggregation is AggregationType.MULTIPLICATIVE:
            hf_returns = np.expm1(hf_workspace_rot)
        else:
            hf_returns = hf_workspace_rot

        out_df = pd.DataFrame(hf_returns, index=hf_index, columns=strategies)

        # 6. Compute round-trip aggregation error per strategy
        max_err = 0.0
        for name in strategies:
            agg_back = aggregate_returns(
                out_df[name], ratio, method=self.aggregation
            )
            err = float(np.max(np.abs(
                agg_back.to_numpy() - low_frequency_panel[name].to_numpy()
            )))
            max_err = max(max_err, err)

        # Realised correlation — on the high-frequency residuals after rotation
        realised_corr_df = pd.DataFrame(
            np.corrcoef(hf_workspace_rot.T - hf_workspace.T + hf_residuals_workspace.T, rowvar=True)
            if hf_residuals_workspace.shape[1] > 1
            else np.array([[1.0]]),
            index=strategies,
            columns=strategies,
        )

        return MultivariateDisaggregationResult(
            high_frequency=out_df,
            per_strategy=per_strategy,
            target_correlation=target_corr_df,
            realised_correlation=realised_corr_df,
            aggregation_error=max_err,
            diagnostics={
                "method": self.method.value,
                "aggregation": self.aggregation.value,
                "correlation_method": self.correlation_method.value,
                "rotated": bool(
                    self.rotate_residuals and hf_residuals_workspace.shape[1] >= 2
                ),
                "n_strategies": len(strategies),
                "ratio": int(ratio),
            },
        )

    # ── Internals ───────────────────────────────────────────────────

    def _validate(
        self,
        panel: pd.DataFrame,
        indicators: dict[str, pd.DataFrame],
        ratio: int,
    ) -> None:
        if not isinstance(panel, pd.DataFrame) or panel.empty:
            raise TypeError("low_frequency_panel must be a non-empty DataFrame.")
        if panel.isna().any().any():
            raise ValueError("low_frequency_panel contains NaN.")
        if not isinstance(indicators, dict) or not indicators:
            raise TypeError(
                "high_frequency_indicators must be a non-empty dict[str, DataFrame]."
            )
        missing = [c for c in panel.columns if c not in indicators]
        if missing:
            raise KeyError(
                f"high_frequency_indicators missing keys for strategies {missing!r}"
            )
        n_lf = len(panel)
        n_hf_expected = n_lf * int(ratio)
        first_index: pd.DatetimeIndex | None = None
        for name, df in indicators.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise TypeError(
                    f"indicators[{name!r}] must be a non-empty DataFrame."
                )
            if len(df) != n_hf_expected:
                raise ValueError(
                    f"indicators[{name!r}] has {len(df)} rows; expected {n_hf_expected}"
                )
            if first_index is None:
                first_index = df.index
            elif not first_index.equals(df.index):
                raise ValueError(
                    "all indicator DataFrames must share the same DatetimeIndex."
                )

    def _extract_hf_workspace(
        self,
        per_strategy: dict[str, DisaggregationResult],
        indicators: dict[str, pd.DataFrame],
        panel: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Return (HF workspace values, HF residuals in workspace, HF index)."""
        strategies = list(panel.columns)
        first = next(iter(indicators.values()))
        hf_index = first.index
        n_hf = len(hf_index)
        hf_ws = np.empty((n_hf, len(strategies)))
        resid_ws = np.empty_like(hf_ws)
        for k, name in enumerate(strategies):
            res = per_strategy[name]
            hf_returns = res.high_frequency.to_numpy()
            if self.aggregation is AggregationType.MULTIPLICATIVE:
                hf_ws[:, k] = np.log1p(hf_returns)
            else:
                hf_ws[:, k] = hf_returns
            # Compute systematic component in workspace
            ind = indicators[name]
            ind_arr = ind.to_numpy()
            if self.aggregation is AggregationType.MULTIPLICATIVE:
                ind_ws = np.log1p(ind_arr)
            else:
                ind_ws = ind_arr
            betas = np.array([res.betas[c] for c in ind.columns])
            sys_ws = ind_ws @ betas
            if res.diagnostics.get("use_intercept", True):
                sys_ws = sys_ws + res.diagnostics["alpha"]
            resid_ws[:, k] = hf_ws[:, k] - sys_ws
        return hf_ws, resid_ws, hf_index

    def _lf_residuals_workspace(
        self,
        per_strategy: dict[str, DisaggregationResult],
        indicators: dict[str, pd.DataFrame],
        panel: pd.DataFrame,
        ratio: int,
    ) -> pd.DataFrame:
        """Compute LF-aggregated workspace residuals per strategy."""
        strategies = list(panel.columns)
        n_lf = len(panel)
        out = np.empty((n_lf, len(strategies)))
        C = aggregation_matrix(n_lf, ratio)
        for k, name in enumerate(strategies):
            res = per_strategy[name]
            hf_returns = res.high_frequency.to_numpy()
            if self.aggregation is AggregationType.MULTIPLICATIVE:
                hf_ws = np.log1p(hf_returns)
            else:
                hf_ws = hf_returns
            ind = indicators[name]
            ind_arr = ind.to_numpy()
            if self.aggregation is AggregationType.MULTIPLICATIVE:
                ind_ws = np.log1p(ind_arr)
            else:
                ind_ws = ind_arr
            betas = np.array([res.betas[c] for c in ind.columns])
            sys_hf_ws = ind_ws @ betas
            if res.diagnostics.get("use_intercept", True):
                sys_hf_ws = sys_hf_ws + res.diagnostics["alpha"]
            resid_hf_ws = hf_ws - sys_hf_ws
            out[:, k] = C @ resid_hf_ws
        return pd.DataFrame(out, index=panel.index, columns=strategies)

    def _target_correlation(
        self,
        lf_residuals: pd.DataFrame,
    ) -> pd.DataFrame:
        cov = lf_residuals.cov()
        if self.correlation_method is CorrelationMethod.EMPIRICAL:
            target_cov = cov
        else:
            diag = pd.DataFrame(
                np.diag(np.diag(cov.to_numpy())),
                index=cov.index,
                columns=cov.columns,
            )
            target_cov = (
                (1.0 - self.shrinkage_intensity) * cov
                + self.shrinkage_intensity * diag
            )
        # Convert to correlation
        std = np.sqrt(np.diag(target_cov.to_numpy()))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = target_cov.to_numpy() / np.outer(std, std)
        # Replace NaN (zero-std) with identity
        np.nan_to_num(corr, copy=False, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return pd.DataFrame(corr, index=lf_residuals.columns, columns=lf_residuals.columns)

    @staticmethod
    def _rotate(
        residuals: np.ndarray,
        target_corr: np.ndarray,
    ) -> np.ndarray:
        """Whiten then re-colour the residuals to match the target correlation."""
        n, k = residuals.shape
        if k < 2 or n < k:
            return residuals.copy()
        emp_cov = np.cov(residuals, rowvar=False)
        emp_std = np.sqrt(np.diag(emp_cov))
        with np.errstate(invalid="ignore", divide="ignore"):
            emp_corr = emp_cov / np.outer(emp_std, emp_std)
        np.nan_to_num(emp_corr, copy=False, nan=0.0)
        np.fill_diagonal(emp_corr, 1.0)
        # Cholesky factors
        try:
            L_emp = np.linalg.cholesky(emp_corr + 1e-12 * np.eye(k))
            L_target = np.linalg.cholesky(target_corr + 1e-12 * np.eye(k))
        except np.linalg.LinAlgError:
            return residuals.copy()
        # Standardise residuals (subtract column means, divide by std)
        col_means = residuals.mean(axis=0)
        residuals_centered = residuals - col_means
        # Standardise to unit variance per column
        with np.errstate(invalid="ignore", divide="ignore"):
            standardised = residuals_centered / emp_std
        np.nan_to_num(standardised, copy=False, nan=0.0)
        # Whiten then recolour
        whitened = np.linalg.solve(L_emp, standardised.T).T
        recoloured = (L_target @ whitened.T).T
        # Restore per-column std (use empirical std as the scale)
        recoloured = recoloured * emp_std
        # Restore column means
        recoloured = recoloured + col_means
        return recoloured


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _match_block_means(
    rotated: np.ndarray,
    original: np.ndarray,
    ratio: int,
) -> np.ndarray:
    """Adjust ``rotated`` so its per-block means match those of ``original``.

    Decomposes each column ε[t] = block_mean[b(t)] + within_block_deviation[t].
    The block means in the original residual are fixed by the Chow-Lin
    aggregation constraint (``C·ε = LF residual``), so we restore them after
    the rotation, leaving only the within-block deviations actually rotated.
    """
    n_hf, k = rotated.shape
    if n_hf % ratio != 0:
        raise ValueError(f"length {n_hf} not divisible by ratio {ratio}")
    n_lf = n_hf // ratio
    rot_b = rotated.reshape(n_lf, ratio, k)
    orig_b = original.reshape(n_lf, ratio, k)
    delta = orig_b.mean(axis=1, keepdims=True) - rot_b.mean(axis=1, keepdims=True)
    return (rot_b + delta).reshape(n_hf, k)


__all__ = [
    "MultivariateDisaggregationResult",
    "MultivariateDisaggregator",
]
