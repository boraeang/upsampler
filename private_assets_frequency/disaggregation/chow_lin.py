r"""
Stage 3 temporal disaggregation: Chow-Lin / Fernández / Litterman.

Theoretical foundation
----------------------
Given a low-frequency series :math:`y_{LF}` (length :math:`n_{LF}`) and a
high-frequency indicator matrix :math:`X_{HF}` (shape
:math:`(n_{HF}, K)` with :math:`n_{HF} = r \cdot n_{LF}`), find a
high-frequency series :math:`y_{HF}` such that

    1. **Aggregation constraint** holds: :math:`y_{LF} = C \, y_{HF}`
       where ``C`` sums each block of ``r`` consecutive entries.
    2. ``y_{HF}`` is related to ``X_{HF}`` via a regression with
       autocorrelated residuals: :math:`y_{HF} = X_{HF} \beta + u_{HF}`
       with :math:`u_{HF} \sim \mathcal{N}(0, \sigma^2 V)`.

The GLS solution is::

    y_HF = X_HF · β̂ + V · C' · (C V C')^{-1} · (y_LF − C X_HF · β̂)
    β̂   = (X_LF' (C V C')^{-1} X_LF)^{-1} · X_LF' (C V C')^{-1} y_LF
    X_LF = C · X_HF

The three variants differ only in the choice of ``V`` (the high-frequency
residual covariance up to scale):

    * **Chow-Lin**  — AR(1) residuals, ``ρ`` estimated by maximising the
      profile log-likelihood on a grid.
    * **Fernández** — random-walk residuals (``ρ = 1`` in the underlying
      AR(1)). No ρ to estimate.
    * **Litterman** — AR(1) on the *differenced* residuals; ``ρ`` again
      estimated by profile-likelihood grid.

Multiplicative aggregation (the default for return data) runs the whole
pipeline in log-return workspace and exponentiates at the boundary, so the
exact compounding constraint :math:`(1 + y_{LF}) = \prod (1 + y_{HF})`
holds to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.protocols import AggregationType, DisaggregationMethod
from ..utils.returns import aggregate_returns, aggregate_returns_by_blocks
from .aggregation import (
    aggregation_matrix,
    ar1_covariance,
    from_workspace,
    irregular_aggregation_matrix,
    litterman_covariance,
    random_walk_covariance,
    to_workspace,
)


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DisaggregationResult:
    """Output of :meth:`ChowLinDisaggregator.fit`.

    Attributes
    ----------
    high_frequency
        Disaggregated high-frequency return series.
    low_frequency_input
        The original low-frequency input (echo, for round-trip checks).
    method
        ``'chow_lin'`` | ``'fernandez'`` | ``'litterman'``.
    aggregation
        ``'multiplicative'`` (default) or ``'additive'``.
    rho
        Estimated AR(1) coefficient for Chow-Lin / Litterman, ``None`` for
        Fernández.
    betas
        Mapping ``{indicator_name: β_k}`` from the GLS step.
    sigma2
        Estimated GLS noise variance.
    log_likelihood
        Profile log-likelihood at the chosen ``ρ``.
    aggregation_error
        Maximum absolute round-trip aggregation error in the workspace
        (additive arithmetic). For multiplicative output, exponentiation
        gives the original input back to machine precision.
    diagnostics
        Free-form mapping for downstream consumers.
    """

    high_frequency: pd.Series
    low_frequency_input: pd.Series
    method: str
    aggregation: str
    rho: float | None
    betas: dict[str, float]
    sigma2: float
    log_likelihood: float
    aggregation_error: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Disaggregator
# ──────────────────────────────────────────────────────────────────


@dataclass
class ChowLinDisaggregator:
    """Stage 3 disaggregator with three pluggable methods.

    Parameters
    ----------
    method
        ``DisaggregationMethod.CHOW_LIN`` (default), ``FERNANDEZ``, or
        ``LITTERMAN``. Strings are also accepted.
    aggregation
        ``AggregationType.MULTIPLICATIVE`` (default) or ``ADDITIVE``.
    rho_grid_size
        Number of grid points for the Chow-Lin / Litterman ρ search.
        Default 41.
    rho_lower, rho_upper
        Bounds on ρ. Defaults ``(-0.99, 0.99)`` keep
        ``ar1_covariance(n, ρ)`` finite.
    use_intercept
        If ``True`` (default), the indicator matrix is augmented with a
        column of ones so β includes an intercept.
    """

    method: DisaggregationMethod | str = DisaggregationMethod.CHOW_LIN
    aggregation: AggregationType | str = AggregationType.MULTIPLICATIVE
    rho_grid_size: int = 41
    rho_lower: float = -0.99
    rho_upper: float = 0.99
    use_intercept: bool = True
    rho_grid: np.ndarray = field(init=False)

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
        if not (-1.0 < self.rho_lower < self.rho_upper < 1.0):
            raise ValueError(
                f"require -1 < rho_lower ({self.rho_lower}) < rho_upper ({self.rho_upper}) < 1."
            )
        if self.rho_grid_size < 5:
            raise ValueError(f"rho_grid_size must be >= 5, got {self.rho_grid_size}")
        self.rho_grid = np.linspace(self.rho_lower, self.rho_upper, self.rho_grid_size)

    # ── Public API ─────────────────────────────────────────────────

    def fit(
        self,
        low_frequency: pd.Series,
        high_frequency_indicators: pd.DataFrame,
        ratio: int | None = None,
        *,
        block_sizes: np.ndarray | list[int] | None = None,
        high_frequency_index: pd.DatetimeIndex | None = None,
    ) -> DisaggregationResult:
        """Disaggregate ``low_frequency`` to the high frequency of ``indicators``.

        Exactly one of ``ratio`` (uniform blocks) or ``block_sizes`` (irregular
        blocks) must be supplied.

        Parameters
        ----------
        low_frequency
            Low-frequency series — typically a desmoothed quarterly PE return.
        high_frequency_indicators
            Indicator (factor) returns at the target high frequency. The number
            of rows must equal ``len(low_frequency) * ratio`` (uniform) or
            ``sum(block_sizes)`` (irregular).
        ratio
            Number of high-frequency periods per low-frequency period (e.g. 3
            for quarterly→monthly). Mutually exclusive with ``block_sizes``.
        block_sizes
            Per-low-frequency-period count of high-frequency observations, for
            irregular calendars (e.g. business days per month in monthly→daily).
            Length must equal ``len(low_frequency)``. Mutually exclusive with
            ``ratio``.
        high_frequency_index
            Optional explicit index for the output. Defaults to
            ``high_frequency_indicators.index``.

        Returns
        -------
        DisaggregationResult
        """
        (
            y_lf_pd,
            X_hf,
            hf_index,
            indicator_names,
            C,
            block_sizes_norm,
        ) = self._validate_and_prepare(
            low_frequency,
            high_frequency_indicators,
            ratio,
            block_sizes,
            high_frequency_index,
        )
        n_lf = len(y_lf_pd)
        n_hf = X_hf.shape[0]
        K = X_hf.shape[1]

        # Workspace (additive vs multiplicative)
        y_lf_w = to_workspace(y_lf_pd, self.aggregation)
        X_hf_w = X_hf.copy()
        if self.aggregation is AggregationType.MULTIPLICATIVE:
            # Indicators are returns too — convert to log-space so the linear
            # relationship r_HF = X_HF · β + ε holds in workspace.
            if (X_hf_w <= -1.0).any():
                raise ValueError("multiplicative workspace requires indicators > -1.")
            X_hf_w = np.log1p(X_hf_w)

        if self.use_intercept:
            X_hf_w = np.hstack([X_hf_w, np.ones((n_hf, 1))])

        X_lf_w = C @ X_hf_w  # aggregated indicator at LF

        # Method dispatch
        method_str = self.method.value
        if method_str == DisaggregationMethod.CHOW_LIN.value:
            rho_hat, log_lik, V = self._fit_chow_lin(y_lf_w, X_hf_w, X_lf_w, C, n_hf)
        elif method_str == DisaggregationMethod.FERNANDEZ.value:
            rho_hat = None
            V = random_walk_covariance(n_hf)
            log_lik = self._profile_log_likelihood(y_lf_w, X_hf_w, X_lf_w, C, V)
        elif method_str == DisaggregationMethod.LITTERMAN.value:
            rho_hat, log_lik, V = self._fit_litterman(y_lf_w, X_hf_w, X_lf_w, C, n_hf)
        else:  # pragma: no cover — guarded by enum
            raise AssertionError(f"unknown method {self.method!r}")

        # Final GLS β̂ at the chosen V
        beta_hat, sigma2 = self._gls_beta(y_lf_w, X_lf_w, C, V)
        # Disaggregated solution in workspace
        Vct = V @ C.T
        CVCt = C @ Vct
        residual_lf = y_lf_w - X_lf_w @ beta_hat
        u_hf = Vct @ np.linalg.solve(CVCt, residual_lf)
        y_hf_w = X_hf_w @ beta_hat + u_hf

        # Convert back to return-space
        y_hf_arr = from_workspace(y_hf_w, self.aggregation)

        # Round-trip check (in workspace, where the constraint is exact)
        agg_back = C @ y_hf_w
        agg_err_workspace = float(np.max(np.abs(agg_back - y_lf_w)))

        # Round-trip in *return* space — even tighter check for the user
        if block_sizes_norm is None:
            agg_back_returns = aggregate_returns(
                y_hf_arr, int(ratio), method=self.aggregation
            )
        else:
            agg_back_returns = aggregate_returns_by_blocks(
                y_hf_arr, block_sizes_norm, method=self.aggregation
            )
        agg_err_return = float(np.max(np.abs(agg_back_returns - y_lf_pd.to_numpy())))

        # Build betas dict
        if self.use_intercept:
            betas = {name: float(beta_hat[k]) for k, name in enumerate(indicator_names)}
            alpha = float(beta_hat[-1])
        else:
            betas = {name: float(beta_hat[k]) for k, name in enumerate(indicator_names)}
            alpha = 0.0
        diagnostics: dict[str, Any] = {
            "alpha": alpha,
            "n_low_frequency": n_lf,
            "n_high_frequency": n_hf,
            "ratio": int(ratio) if block_sizes_norm is None else None,
            "aggregation_error_workspace": agg_err_workspace,
            "aggregation_error_return_space": agg_err_return,
            "indicator_names": indicator_names,
            "use_intercept": self.use_intercept,
        }
        if block_sizes_norm is not None:
            diagnostics["block_sizes"] = [int(s) for s in block_sizes_norm]
        if rho_hat is not None:
            diagnostics["rho"] = float(rho_hat)

        out = pd.Series(y_hf_arr, index=hf_index, name=low_frequency.name)
        return DisaggregationResult(
            high_frequency=out,
            low_frequency_input=low_frequency.copy(),
            method=method_str,
            aggregation=self.aggregation.value,
            rho=float(rho_hat) if rho_hat is not None else None,
            betas=betas,
            sigma2=float(sigma2),
            log_likelihood=float(log_lik),
            aggregation_error=agg_err_return,
            diagnostics=diagnostics,
        )

    # ── Estimators ─────────────────────────────────────────────────

    def _fit_chow_lin(
        self,
        y_lf_w: np.ndarray,
        X_hf_w: np.ndarray,
        X_lf_w: np.ndarray,
        C: np.ndarray,
        n_hf: int,
    ) -> tuple[float, float, np.ndarray]:
        log_lik_grid = np.empty(self.rho_grid_size)
        for i, rho in enumerate(self.rho_grid):
            V = ar1_covariance(n_hf, float(rho))
            log_lik_grid[i] = self._profile_log_likelihood(y_lf_w, X_hf_w, X_lf_w, C, V)
        i_star = int(np.argmax(log_lik_grid))
        rho_hat = float(self.rho_grid[i_star])
        V = ar1_covariance(n_hf, rho_hat)
        return rho_hat, float(log_lik_grid[i_star]), V

    def _fit_litterman(
        self,
        y_lf_w: np.ndarray,
        X_hf_w: np.ndarray,
        X_lf_w: np.ndarray,
        C: np.ndarray,
        n_hf: int,
    ) -> tuple[float, float, np.ndarray]:
        log_lik_grid = np.empty(self.rho_grid_size)
        for i, rho in enumerate(self.rho_grid):
            V = litterman_covariance(n_hf, float(rho))
            log_lik_grid[i] = self._profile_log_likelihood(y_lf_w, X_hf_w, X_lf_w, C, V)
        i_star = int(np.argmax(log_lik_grid))
        rho_hat = float(self.rho_grid[i_star])
        V = litterman_covariance(n_hf, rho_hat)
        return rho_hat, float(log_lik_grid[i_star]), V

    def _profile_log_likelihood(
        self,
        y_lf_w: np.ndarray,
        X_hf_w: np.ndarray,
        X_lf_w: np.ndarray,
        C: np.ndarray,
        V: np.ndarray,
    ) -> float:
        """Profile log-likelihood at the GLS β̂ and σ̂² for a given V."""
        beta_hat, sigma2 = self._gls_beta(y_lf_w, X_lf_w, C, V)
        n_lf = len(y_lf_w)
        # log|σ²·V_LF| = n_lf log σ² + log|V_LF|
        V_lf = C @ V @ C.T
        sign, log_det = np.linalg.slogdet(V_lf)
        if sign <= 0.0 or sigma2 <= 0.0:
            return -np.inf
        ll = (
            -0.5 * n_lf * np.log(2.0 * np.pi)
            - 0.5 * n_lf * np.log(sigma2)
            - 0.5 * log_det
            - 0.5 * n_lf  # the (e' V_LF^{-1} e) / σ² term equals n_lf at MLE σ²
        )
        return float(ll)

    @staticmethod
    def _gls_beta(
        y_lf_w: np.ndarray,
        X_lf_w: np.ndarray,
        C: np.ndarray,
        V: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Compute GLS β̂ and the MLE σ̂² for the LF model y_LF = X_LF·β + ε_LF."""
        V_lf = C @ V @ C.T
        # Solve V_lf · A = X_lf and V_lf · b = y_lf
        VinvX = np.linalg.solve(V_lf, X_lf_w)
        Vinvy = np.linalg.solve(V_lf, y_lf_w)
        XtVinvX = X_lf_w.T @ VinvX
        XtVinvy = X_lf_w.T @ Vinvy
        try:
            beta_hat = np.linalg.solve(XtVinvX, XtVinvy)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"GLS normal equations failed: {exc}") from exc
        residual = y_lf_w - X_lf_w @ beta_hat
        n_lf = len(y_lf_w)
        sigma2 = float(residual @ np.linalg.solve(V_lf, residual) / n_lf)
        return beta_hat, sigma2

    # ── Validation ─────────────────────────────────────────────────

    def _validate_and_prepare(
        self,
        low_frequency: pd.Series,
        high_frequency_indicators: pd.DataFrame,
        ratio: int | None,
        block_sizes: np.ndarray | list[int] | None,
        high_frequency_index: pd.DatetimeIndex | None,
    ) -> tuple[pd.Series, np.ndarray, pd.DatetimeIndex, list[str], np.ndarray, np.ndarray | None]:
        if not isinstance(low_frequency, pd.Series):
            raise TypeError(
                f"low_frequency must be a Series, got {type(low_frequency).__name__}"
            )
        if not isinstance(high_frequency_indicators, pd.DataFrame) or high_frequency_indicators.empty:
            raise TypeError("high_frequency_indicators must be a non-empty DataFrame.")
        if low_frequency.isna().any():
            raise ValueError("low_frequency contains NaN.")
        if high_frequency_indicators.isna().any().any():
            raise ValueError("high_frequency_indicators contains NaN.")
        if (ratio is None) == (block_sizes is None):
            raise ValueError("provide exactly one of `ratio` or `block_sizes`.")

        n_lf = len(low_frequency)
        if block_sizes is None:
            if not isinstance(ratio, (int, np.integer)) or ratio < 2:
                raise ValueError(f"ratio must be an integer >= 2, got {ratio!r}")
            n_hf_expected = n_lf * int(ratio)
            C = aggregation_matrix(n_lf, int(ratio))
            block_sizes_norm: np.ndarray | None = None
        else:
            sizes = np.asarray(block_sizes, dtype=int)
            if sizes.ndim != 1 or sizes.size != n_lf:
                raise ValueError(
                    f"block_sizes must be 1-D of length n_lf={n_lf}, got shape {sizes.shape}."
                )
            if (sizes < 1).any():
                raise ValueError("block_sizes entries must all be >= 1.")
            n_hf_expected = int(sizes.sum())
            C = irregular_aggregation_matrix(sizes)
            block_sizes_norm = sizes

        if len(high_frequency_indicators) != n_hf_expected:
            raise ValueError(
                f"high_frequency_indicators has {len(high_frequency_indicators)} rows, "
                f"expected {n_hf_expected}."
            )
        hf_index = (
            high_frequency_index
            if high_frequency_index is not None
            else high_frequency_indicators.index
        )
        if not isinstance(hf_index, pd.DatetimeIndex):
            raise TypeError("high_frequency_index must be a DatetimeIndex.")
        if len(hf_index) != n_hf_expected:
            raise ValueError(
                f"high_frequency_index length {len(hf_index)} != expected {n_hf_expected}"
            )
        X_hf = high_frequency_indicators.to_numpy(dtype=float)
        return (
            low_frequency,
            X_hf,
            hf_index,
            list(high_frequency_indicators.columns),
            C,
            block_sizes_norm,
        )


__all__ = ["ChowLinDisaggregator", "DisaggregationResult"]
