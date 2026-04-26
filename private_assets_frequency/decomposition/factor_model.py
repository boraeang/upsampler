r"""
Stage 2 factor regression: decompose desmoothed returns into systematic + idiosyncratic components.

Per the spec, Stage 2 operates at the *native* frequency of the desmoothed
return series (quarterly for PE / RE / infrastructure / credit, monthly for
hedge funds). The resulting :class:`FactorRegressionResult` captures the
factor loadings and the residual series; downstream stages need both — the
loadings drive the systematic disaggregated component (Stage 3 indicators),
the residuals drive the idiosyncratic component (Stage 4 distribution).

Two estimators are supported:

    * **OLS** — uniform weighting. The default.
    * **Weighted-LS with exponential decay** — recent observations get more
      weight, parametrised by ``halflife`` in periods (Stage 2 spec point 1
      "WLS with optional exponential decay weighting"). Useful when factor
      loadings have drifted (e.g. PE leverage trending down).

Edge cases:

    * Number of factors >= number of observations → raise (singular system).
    * NaN in inputs → raise.
    * Zero-variance factor column → raise (perfectly collinear with intercept).

Return values are pandas-aware: the residual series carries the same index
and name as the input return series; ``betas`` is keyed by factor name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FactorRegressionResult:
    """Output of :meth:`FactorModel.fit`.

    Attributes
    ----------
    betas
        Mapping ``{factor_name: β}`` of factor loadings (excluding intercept).
    alpha
        Estimated intercept (zero if the model was fit without intercept).
    residuals
        Per-period residual ``ε_t = r_t - Σ β_k F_k,t - α``, indexed like
        the input return series.
    fitted
        Per-period systematic component ``Σ β_k F_k,t + α``.
    r_squared
        Coefficient of determination on the *weighted* fit.
    sigma_eps
        Residual standard deviation (sqrt of unbiased mean squared error).
    n_obs
        Number of observations used.
    weights
        Sample weights used in the fit (uniform 1's for OLS).
    method
        ``'ols'`` or ``'wls_expdecay'``.
    halflife
        Halflife in periods for exponential-decay weighting; ``None`` for OLS.
    diagnostics
        Free-form diagnostic mapping (collinearity, leverage, etc.).
    """

    betas: dict[str, float]
    alpha: float
    residuals: pd.Series
    fitted: pd.Series
    r_squared: float
    sigma_eps: float
    n_obs: int
    weights: np.ndarray
    method: str
    halflife: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict(self, factor_returns: pd.DataFrame) -> pd.Series:
        """Predict systematic returns for a new ``factor_returns`` DataFrame.

        Parameters
        ----------
        factor_returns
            Factor returns with at least the columns named in ``self.betas``.

        Returns
        -------
        pd.Series
            ``Σ β_k F_k,t + α`` evaluated at each row of ``factor_returns``.
        """
        missing = [k for k in self.betas if k not in factor_returns.columns]
        if missing:
            raise KeyError(f"factor_returns missing columns {missing!r}")
        sys = factor_returns[list(self.betas)].mul(
            pd.Series(self.betas), axis=1
        ).sum(axis=1)
        return sys + self.alpha


# ──────────────────────────────────────────────────────────────────
# FactorModel
# ──────────────────────────────────────────────────────────────────


@dataclass
class FactorModel:
    """Stage 2 factor regression with optional exponential-decay weighting.

    Parameters
    ----------
    method
        ``'ols'`` (default) or ``'wls_expdecay'``.
    halflife
        Required when ``method='wls_expdecay'``: weight at observation
        ``T - halflife`` is half the weight at the most recent observation.
        Must be ``> 0``.
    use_intercept
        If ``True`` (default), include an intercept column. Set to ``False``
        when fitting strategies that have already had alpha removed elsewhere.
    """

    method: Literal["ols", "wls_expdecay"] = "ols"
    halflife: float | None = None
    use_intercept: bool = True

    def __post_init__(self) -> None:
        if self.method not in ("ols", "wls_expdecay"):
            raise ValueError(
                f"method must be 'ols' or 'wls_expdecay', got {self.method!r}"
            )
        if self.method == "wls_expdecay":
            if self.halflife is None or self.halflife <= 0.0:
                raise ValueError(
                    "halflife must be a positive number for method='wls_expdecay'."
                )

    def fit(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> FactorRegressionResult:
        """Run the regression and return a :class:`FactorRegressionResult`."""
        if not isinstance(returns, pd.Series):
            raise TypeError(
                f"returns must be a Series, got {type(returns).__name__}"
            )
        if not isinstance(factor_returns, pd.DataFrame) or factor_returns.empty:
            raise TypeError(
                "factor_returns must be a non-empty DataFrame."
            )
        if returns.isna().any():
            raise ValueError("returns contains NaN.")
        factor_returns = factor_returns.reindex(returns.index)
        if factor_returns.isna().any().any():
            raise ValueError("factor_returns has NaN after alignment to returns.")

        n = len(returns)
        K = factor_returns.shape[1]
        d = K + (1 if self.use_intercept else 0)
        if n <= d:
            raise ValueError(
                f"need n > {d} observations for a stable fit, got n={n}."
            )

        # Detect zero-variance factor columns (collinear with intercept)
        var_cols = factor_returns.var(ddof=0)
        if self.use_intercept and (var_cols == 0).any():
            zero = list(var_cols.index[var_cols == 0])
            raise ValueError(
                f"factor column(s) {zero!r} have zero variance — collinear with intercept."
            )

        # Build design matrix
        X = factor_returns.to_numpy(dtype=float)
        if self.use_intercept:
            X = np.hstack([X, np.ones((n, 1))])
        y = returns.to_numpy(dtype=float)

        # Compute weights
        weights = self._compute_weights(n)
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, None]
        yw = y * sqrt_w

        # Solve weighted normal equations (more stable than direct lstsq for d ≤ ~50)
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        try:
            coefs = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"weighted normal equations failed: {exc}") from exc

        beta_arr = coefs[:K]
        alpha = float(coefs[K]) if self.use_intercept else 0.0
        betas = {col: float(b) for col, b in zip(factor_returns.columns, beta_arr)}

        # Residuals and fitted (in original space, not weighted)
        fitted_arr = X @ coefs
        resid_arr = y - fitted_arr
        residuals = pd.Series(resid_arr, index=returns.index, name=returns.name)
        fitted = pd.Series(fitted_arr, index=returns.index, name="fitted")

        # σ_ε via unbiased estimator on weighted residuals
        wse = float(np.sum(weights * resid_arr**2))
        df = max(n - d, 1)
        sigma_eps = float(np.sqrt(wse / df))

        # Weighted R²
        ybar_w = float(np.sum(weights * y) / weights.sum())
        ss_tot = float(np.sum(weights * (y - ybar_w) ** 2))
        ss_res = wse
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

        diagnostics = {
            "condition_number": float(np.linalg.cond(XtX)),
            "factor_names": list(factor_returns.columns),
        }

        return FactorRegressionResult(
            betas=betas,
            alpha=alpha,
            residuals=residuals,
            fitted=fitted,
            r_squared=float(r_squared),
            sigma_eps=sigma_eps,
            n_obs=n,
            weights=weights,
            method=self.method,
            halflife=self.halflife,
            diagnostics=diagnostics,
        )

    def _compute_weights(self, n: int) -> np.ndarray:
        if self.method == "ols":
            return np.ones(n)
        # Exponential decay: weight at obs i (0 = oldest, n-1 = newest)
        # is exp(-(n-1-i) · ln(2) / halflife). Most recent obs gets w=1,
        # halflife observations ago gets w=0.5.
        ages = np.arange(n - 1, -1, -1, dtype=float)
        w = np.exp(-ages * np.log(2.0) / float(self.halflife))  # type: ignore[arg-type]
        return w


__all__ = ["FactorModel", "FactorRegressionResult"]
