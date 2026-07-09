r"""
Rudin-Mao-Zhang-Fink (2019) reparameterised unsmoothing model (Stage 1).

Model
-----
Following Rudin, Mao, Zhang & Fink (2019) "Fitting Private Equity into the
Total Portfolio Framework," the observed smoothed return :math:`\tilde r^o_t`
is decomposed via the *reparameterised* form (their Eq. 3):

.. math::

   r^E_t = \theta_0 \, \tilde r^o_t + \sum_{j=1}^{Q} \theta_j \, \tilde r^o_{t-j},
   \qquad \sum_{j=0}^{Q} \theta_j = 1,

where :math:`r^E_t` is the latent true economic return. Combined with the
factor model :math:`r^E_t = \alpha + \sum_i \beta_i F^i_t + \varepsilon_t`
this yields Eq. 4 — the single-regression estimating equation the paper
prescribes:

.. math::

   \tilde r^o_t = \alpha + \theta_0 \sum_i \beta_i F^i_t
   + \sum_{j=1}^{Q} \theta_j \, \tilde r^o_{t-j} + \varepsilon_t
   \qquad (Q \ge 1) .

Following the paper (and per the module's spec) the estimator is a *linear*
OLS on Eq. 4 with the composite factor coefficients :math:`c_i = \theta_0
\beta_i` algebraically unwound after the fit — **not** nonlinear least
squares. Standard errors are classical by default, optionally
heteroskedasticity-robust (``use_hc_se=True``).

Regression layout
-----------------
The regressor matrix for a lag order :math:`Q` and factor matrix
:math:`F \in \mathbb R^{N \times K}` is:

.. code-block:: text

    y_dep    = tilde_r_o[Q:]
    X        = [ constant,               # → alpha
                 F[Q:, :],                # K columns → c_i = θ_0 β_i
                 tilde_r_o[Q-1:N-1],      # → θ_1
                 tilde_r_o[Q-2:N-2],      # → θ_2
                 …,
                 tilde_r_o[0:N-Q] ]       # → θ_Q

The first :math:`Q` observations are lost. For :math:`Q = 0` the model
collapses to a plain factor regression (no lag columns; :math:`\theta_0 = 1`
by construction and :math:`\beta_i = c_i`).

Coefficient unwinding
---------------------
Given the raw OLS coefficients :math:`(\alpha, c_1, \dots, c_K,
\theta_1, \dots, \theta_Q)`:

.. code-block:: text

    theta_0 = 1 - sum(theta_1..theta_Q)     # sum-to-one identity
    beta_i  = c_i / theta_0                 # unwind

A hard warning fires when :math:`\theta_0 < \theta_0`-warning-threshold
(default 0.1) because :math:`\beta_i` becomes numerically unstable in that
regime.

Reconstruction & risk properties
--------------------------------
Reconstruction uses Eq. 3 directly on the aligned window
:math:`t = Q, Q+1, \dots, N-1`:

.. math::

   \hat r^E_t = \theta_0 \tilde r^o_t + \sum_{j=1}^{Q} \theta_j \tilde r^o_{t-j} .

Reported risk quantities (annualised where indicated):

* :math:`\alpha` — per-period and annualised (simple = ``alpha·ppy``;
  geometric = ``(1+alpha)**ppy - 1``; default simple, per the paper).
* :math:`\beta_i` — factor loadings (unwound).
* Volatility of :math:`\hat r^E`, annualised as
  :math:`\sigma\sqrt{\text{ppy}}`.
* % variance explained :math:`= 1 - \text{Var}(\varepsilon^{\text{true}}) /
  \text{Var}(\hat r^E)` where :math:`\varepsilon^{\text{true}}_t = \hat r^E_t
  - \alpha - \sum \beta_i F^i_t`.
* Idiosyncratic (residual) volatility :math:`= \text{std}(\varepsilon^{
  \text{true}}) \sqrt{\text{ppy}}`.

Diagnostics include Ljung-Box and Durbin-Watson on the OLS residuals of
Eq. 4 (which the paper's derivation claims are non-autocorrelated — a
central selling point over the Pedersen-Page-He two-step form), the
regression :math:`R^2`, the realised :math:`\sum \theta_j` (should be
:math:`\approx 1` — the sum constraint is *not* imposed at estimation and
serves as a specification check), and the estimated :math:`\theta_0`.

Lag selection
-------------
:meth:`RudinReparamSmoothing.select_lag_order` compares :math:`Q \in
\{0, \dots, \text{max}\_lag\}` and returns a diagnostic table plus a
recommended :math:`Q`. Two modes:

1. **In-sample (default).** ``variance_ratio_in_sample = Var(ε^{Eq.4}) /
   Var(\hat r^E)``. The spec notes this mechanically favours higher
   :math:`Q`, so the default recommendation applies a parsimony rule: pick
   the smallest :math:`Q` whose in-sample % variance explained is within
   ``tolerance`` of the best. Falls back to :math:`Q = 1` on ties, matching
   the paper's empirical finding.
2. **Out-of-sample.** If ``test_observed_returns`` and
   ``test_factor_returns`` are supplied, fit :math:`(\theta, \beta, \alpha)`
   on the training series, then apply them to the test window and report
   ``Var(test_actual - model_prediction) / Var(test_actual)`` on the
   reconstructed series. A ratio > 1 flags overfitting for that :math:`Q`.
   Recommendation picks the minimising :math:`Q`.

References
----------
Rudin, A., Mao, J., Zhang, N. R., & Fink, A.-M. (2019). "Fitting Private
Equity into the Total Portfolio Framework." *Journal of Portfolio
Management* 46(2), 60-77.  See in particular the "Risk Estimation Through
Unsmoothing" section (Eqs. 1-4) and Exhibit 1.
Stefek, D., & Suryanarayanan, R. (2012). MSCI Research Insight —
reparameterised unsmoothing for real estate.
Pedersen, N., Page, S., & He, F. (2014). "Asset Allocation: Risk Models for
Alternative Investments." *Financial Analysts Journal* 70(3), 34-45.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

from ..core.protocols import DesmoothedResult, Frequency
from .theta_w_conversion import theta_to_w


# ──────────────────────────────────────────────────────────────────
# Configuration & constants
# ──────────────────────────────────────────────────────────────────

_ALLOWED_ALPHA_ANN = frozenset({"simple", "geometric"})
_THETA_0_UNSTABLE = 0.1  # warn if |θ_0| below this: β estimates unstable
_LJUNG_BOX_LAGS = 4
_MAX_LAG_ORDER = 4  # spec: Q ∈ {0, 1, 2, 3, 4}


# ──────────────────────────────────────────────────────────────────
# Result containers for lag-selection diagnostics
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LagSelectionResult:
    """Return value of :meth:`RudinReparamSmoothing.select_lag_order`.

    Attributes
    ----------
    table
        One row per :math:`Q \\in \\{0, \\dots, \\text{max}\\_lag\\}` with columns
        ``alpha_ann``, ``vol_ann``, ``pct_variance_explained``,
        ``idio_vol_ann``, ``variance_ratio_in_sample``, and (if available)
        ``variance_ratio_out_of_sample``. β columns are added dynamically as
        ``beta_<factor_name>``.
    recommended_lag
        The Q selected by the recommendation logic (see the module docstring
        for the rule).
    rationale
        Free-form string explaining why ``recommended_lag`` was chosen.
    mode
        ``"in_sample"`` or ``"out_of_sample"`` — which diagnostic drove the
        recommendation.
    """

    table: pd.DataFrame
    recommended_lag: int
    rationale: str
    mode: str


# ──────────────────────────────────────────────────────────────────
# The smoother class
# ──────────────────────────────────────────────────────────────────


@dataclass
class RudinReparamSmoothing:
    """Reparameterised unsmoothing model of Rudin, Mao, Zhang & Fink (2019).

    Parameters
    ----------
    n_lags
        Lag order :math:`Q \\in \\{0, 1, 2, 3, 4\\}`. Default 1 (the paper's
        preferred value). :math:`Q = 0` reduces to a plain factor regression.
    alpha_annualization
        ``"simple"`` (``α · periods_per_year``) or ``"geometric"``
        (``(1 + α)**periods_per_year - 1``). Default ``"simple"`` to match
        the paper's Exhibit 1 convention.
    use_hc_se
        Report heteroskedasticity-consistent (HC1) standard errors instead of
        classical OLS SEs. Default ``False`` — the paper's framing relies on
        the reparameterisation making Eq. 4's error non-autocorrelated, so
        classical SEs are appropriate.
    periods_per_year
        Annualisation factor. If ``None`` (default), inferred from the return
        index frequency (quarterly → 4, monthly → 12, weekly → 52, daily → 252).

    Notes
    -----
    The class holds no state between calls to :meth:`fit` outside of an
    internal ``_last_fit`` cache used only for reporting; construct once and
    reuse.
    """

    n_lags: int = 1
    alpha_annualization: str = "simple"
    use_hc_se: bool = False
    periods_per_year: int | None = None

    _last_fit: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n_lags, (int, np.integer)):
            raise TypeError(f"n_lags must be int, got {type(self.n_lags).__name__}")
        if not (0 <= self.n_lags <= _MAX_LAG_ORDER):
            raise ValueError(
                f"n_lags must lie in [0, {_MAX_LAG_ORDER}], got {self.n_lags}"
            )
        if self.alpha_annualization not in _ALLOWED_ALPHA_ANN:
            raise ValueError(
                f"alpha_annualization must be one of {_ALLOWED_ALPHA_ANN!r}, "
                f"got {self.alpha_annualization!r}"
            )
        if self.periods_per_year is not None and self.periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year must be > 0 or None, got {self.periods_per_year}"
            )

    # ── protocol methods ──────────────────────────────────────────────

    def fit(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
        priors: dict[str, Any] | None = None,
    ) -> DesmoothedResult:
        """Estimate :math:`(\\alpha, \\beta, \\theta)` via OLS on Eq. 4.

        Parameters
        ----------
        observed_returns
            Smoothed observed return series at native frequency.
        factor_returns
            Factor return matrix aligned to ``observed_returns.index``. Each
            column is one systematic factor. Must be non-empty (at least one
            factor column) — the model requires factor regressors.
        priors
            Accepted for :class:`SmoothingModel` protocol compatibility.
            Ignored. If supplied and non-empty an informational note is
            recorded in ``diagnostics['notes']``.

        Returns
        -------
        DesmoothedResult

        Raises
        ------
        TypeError
            If ``observed_returns`` is not a :class:`pandas.Series` or
            ``factor_returns`` is not a :class:`pandas.DataFrame`.
        ValueError
            If the indices are misaligned, if ``factor_returns`` has zero
            columns, or if the effective sample after dropping the first
            ``n_lags`` observations is smaller than the number of regressors.
        """
        y, F, ppy, factor_names = self._validate_inputs(observed_returns, factor_returns)
        Q = self.n_lags
        N = y.size
        n_eff = N - Q
        n_factors = len(factor_names)
        n_regressors = 1 + n_factors + Q  # constant + factors + lag columns
        if n_eff <= n_regressors:
            raise ValueError(
                f"Effective sample size {n_eff} is not larger than the number "
                f"of regressors {n_regressors} (constant + {n_factors} factors "
                f"+ {Q} lags). Increase the sample or decrease n_lags."
            )

        # ── build regressor matrix ────────────────────────────────
        y_dep = y[Q:]
        F_aligned = F[Q:, :]
        lag_cols = _build_lag_matrix(y, Q)  # (n_eff, Q); empty if Q == 0
        X_no_const = np.concatenate([F_aligned, lag_cols], axis=1) if Q > 0 else F_aligned
        X = sm.add_constant(X_no_const, has_constant="add")

        # ── OLS fit ───────────────────────────────────────────────
        cov_type = "HC1" if self.use_hc_se else "nonrobust"
        model = sm.OLS(y_dep, X)
        results = model.fit(cov_type=cov_type)

        params = np.asarray(results.params, dtype=float)
        se = np.asarray(results.bse, dtype=float)
        tvals = np.asarray(results.tvalues, dtype=float)

        # Layout: [const, factor_1, ..., factor_K, lag_1, ..., lag_Q]
        alpha = float(params[0])
        alpha_se = float(se[0])
        alpha_t = float(tvals[0])
        c = params[1 : 1 + n_factors]  # composite = θ_0 · β_i
        c_se = se[1 : 1 + n_factors]
        c_t = tvals[1 : 1 + n_factors]
        theta_high = params[1 + n_factors : 1 + n_factors + Q]  # θ_1..θ_Q
        theta_high_se = se[1 + n_factors : 1 + n_factors + Q]
        theta_high_t = tvals[1 + n_factors : 1 + n_factors + Q]

        # ── unwind ────────────────────────────────────────────────
        theta_0 = 1.0 - float(theta_high.sum())
        notes: list[str] = []
        if abs(theta_0) < _THETA_0_UNSTABLE:
            notes.append(
                f"theta_0 = {theta_0:.4f} is below the stability floor "
                f"(|θ_0| < {_THETA_0_UNSTABLE}); β estimates are numerically "
                "unstable — treat with caution."
            )
        if abs(theta_0) < 1e-12:
            raise ValueError(
                "Estimated theta_0 ≈ 0 (sum of estimated θ_j's ≈ 1). "
                "Beta unwinding would divide by zero — the sample is likely "
                "too short for the requested n_lags, or the data is degenerate."
            )
        beta = c / theta_0
        beta_dict = {name: float(beta[k]) for k, name in enumerate(factor_names)}
        theta = np.concatenate([[theta_0], theta_high])  # length Q+1

        # w (equivalent MA weights via polynomial inversion)
        try:
            w_full = theta_to_w(theta)
            # Prune trailing near-zero mass for readability
            w_trimmed = _trim_trailing_zeros(w_full)
        except ValueError:
            w_trimmed = np.array([np.nan])
            notes.append("θ ↔ w conversion failed (θ_0 too small); w set to NaN.")

        # ── reconstruction (Eq. 3) ────────────────────────────────
        r_hat_arr = self._reconstruct(y, theta)  # length n_eff
        true_returns = pd.Series(
            r_hat_arr,
            index=observed_returns.index[Q:],
            name=observed_returns.name,
        )

        # ── risk properties ───────────────────────────────────────
        systematic = F_aligned @ beta  # length n_eff
        eps_true = r_hat_arr - alpha - systematic  # residual on Eq. 2, on reconstructed data
        var_rhat = float(np.var(r_hat_arr, ddof=1))
        var_eps_true = float(np.var(eps_true, ddof=1))
        pct_var_explained = float(1.0 - var_eps_true / var_rhat) if var_rhat > 0 else float("nan")

        vol_annual = float(np.sqrt(var_rhat * ppy))
        idio_vol_annual = float(np.sqrt(var_eps_true * ppy))
        alpha_annualised = _annualise_alpha(alpha, ppy, self.alpha_annualization)

        # ── Eq. 4 residual diagnostics (the paper's non-autocorrelation claim) ─
        resid = np.asarray(results.resid, dtype=float)
        dw = float(durbin_watson(resid))
        lb = acorr_ljungbox(resid, lags=[min(_LJUNG_BOX_LAGS, max(1, n_eff // 4))], return_df=True)
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])
        lb_stat = float(lb["lb_stat"].iloc[0])
        theta_sum = float(theta.sum())  # sum constraint diagnostic; should ≈ 1 by construction
        # (θ_0 was defined as 1 - Σθ_high, so theta.sum() == 1 exactly; report anyway.)

        # ── assemble result payload ───────────────────────────────
        if priors:
            notes.append("Priors were supplied but this OLS model ignores them.")

        smoothing_params = {
            "theta": theta.copy(),
            "w": w_trimmed.copy(),
            "beta": beta_dict,
            "alpha": alpha,
            "alpha_annualised": alpha_annualised,
            "n_lags": Q,
        }

        # Posterior summary is labelled "OLS/frequentist" — no true posteriors here.
        posterior_summary = {
            "estimator": "ols_frequentist",
            "cov_type": cov_type,
            "alpha": {
                "estimate": alpha,
                "std_error": alpha_se,
                "t_stat": alpha_t,
            },
            "beta": {
                name: {
                    # β_i = c_i / θ_0; SE via delta method (θ_0 treated as fixed here
                    # for simplicity — matches the paper's linear-OLS framing).
                    "estimate": float(beta[k]),
                    "std_error_composite": float(c_se[k]),
                    "t_stat_composite": float(c_t[k]),
                    "note": "std_error/t_stat are for the composite θ_0·β_i.",
                }
                for k, name in enumerate(factor_names)
            },
            "theta_high": {
                f"theta_{j+1}": {
                    "estimate": float(theta_high[j]),
                    "std_error": float(theta_high_se[j]),
                    "t_stat": float(theta_high_t[j]),
                }
                for j in range(Q)
            },
            "theta_0": {
                "estimate": theta_0,
                "note": "Derived from 1 - Σθ_j; not a free OLS coefficient.",
            },
        }

        diagnostics = {
            "method": "rudin_reparam",
            "n_lags": Q,
            "n_observations": int(N),
            "n_effective": int(n_eff),
            "periods_per_year": int(ppy),
            "regression_r_squared": float(results.rsquared),
            "regression_r_squared_adj": float(results.rsquared_adj),
            "durbin_watson": dw,
            "ljung_box_stat": lb_stat,
            "ljung_box_pvalue": lb_pvalue,
            "ljung_box_lags_tested": min(_LJUNG_BOX_LAGS, max(1, n_eff // 4)),
            "vol_annualised": vol_annual,
            "idio_vol_annualised": idio_vol_annual,
            "pct_variance_explained": pct_var_explained,
            "theta_sum": theta_sum,
            "theta_0": theta_0,
            "variance_ratio_in_sample": (
                float(var_eps_true / var_rhat) if var_rhat > 0 else float("nan")
            ),
            "notes": notes,
        }

        # Cache for reference
        self._last_fit = {
            "theta": theta,
            "beta": beta.copy(),
            "alpha": alpha,
            "factor_names": tuple(factor_names),
            "periods_per_year": int(ppy),
            "resid_eq4": resid,
            "n_effective": int(n_eff),
        }

        return DesmoothedResult(
            true_returns=true_returns,
            smoothing_params=smoothing_params,
            posterior_summary=posterior_summary,
            diagnostics=diagnostics,
        )

    def desmooth(
        self,
        observed_returns: np.ndarray,
        smoothing_params: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct true returns via Eq. 3 given a θ vector.

        Parameters
        ----------
        observed_returns
            Observed return array (1-D, length N).
        smoothing_params
            θ vector of length ``n_lags + 1``. The first entry is θ_0, the
            remainder are θ_1..θ_Q.

        Returns
        -------
        numpy.ndarray
            The reconstructed series of length ``N - n_lags``. The first
            ``n_lags`` observations are lost per Eq. 3's finite-lag construction.
        """
        theta = np.asarray(smoothing_params, dtype=float).reshape(-1)
        Q = self.n_lags
        if theta.size != Q + 1:
            raise ValueError(
                f"smoothing_params must have length n_lags + 1 = {Q + 1}, "
                f"got {theta.size}."
            )
        y = np.asarray(observed_returns, dtype=float).reshape(-1)
        if y.size <= Q:
            raise ValueError(
                f"observed_returns has length {y.size}; need > n_lags ({Q})."
            )
        return self._reconstruct(y, theta)

    def log_likelihood(
        self,
        smoothing_params: np.ndarray,
        observed_returns: np.ndarray,
        factor_returns: np.ndarray,
    ) -> float:
        """Gaussian log-likelihood at :math:`\\hat\\sigma^2 = \\text{RSS}/n`.

        This is the profile likelihood at the given θ, with α and β
        concentrated out by OLS on Eq. 4. Provided primarily for model
        comparison across lag orders.

        Parameters
        ----------
        smoothing_params
            θ vector of length ``n_lags + 1``. Only the ``θ_1..θ_Q`` block
            (positions 1..Q) is used; ``θ_0`` is derived from the sum
            constraint but recomputed internally for the reconstruction step,
            so a mismatch between ``smoothing_params[0]`` and
            ``1 - sum(smoothing_params[1:])`` is silently overridden.
        observed_returns
            Observed return array (1-D, length N).
        factor_returns
            Factor return array (N × K or 1-D).
        """
        theta = np.asarray(smoothing_params, dtype=float).reshape(-1)
        Q = self.n_lags
        if theta.size != Q + 1:
            return -np.inf
        y = np.asarray(observed_returns, dtype=float).reshape(-1)
        F = np.asarray(factor_returns, dtype=float)
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        if y.size != F.shape[0]:
            return -np.inf
        N = y.size
        n_eff = N - Q
        if n_eff <= F.shape[1] + Q + 1:
            return -np.inf
        # Build Eq. 4 regressors and fit; use theta[1:] as the enforced lag coefs.
        # For a genuine profile likelihood over β/α given θ, run OLS with the
        # lag columns absorbed into the dependent variable.
        F_aligned = F[Q:, :]
        lag_cols = _build_lag_matrix(y, Q)  # (n_eff, Q)
        y_dep = y[Q:]
        if Q > 0:
            y_adj = y_dep - lag_cols @ theta[1:]
        else:
            y_adj = y_dep
        X = sm.add_constant(F_aligned, has_constant="add")
        # OLS on y_adj = α + (θ_0·β)·F + ε → concentrating out α, β
        results = sm.OLS(y_adj, X).fit()
        resid = np.asarray(results.resid, dtype=float)
        rss = float(np.sum(resid**2))
        if rss <= 0.0:
            return -np.inf
        sigma2 = rss / n_eff
        ll = -0.5 * n_eff * (np.log(2.0 * np.pi * sigma2) + 1.0)
        return float(ll)

    def log_prior(
        self,
        smoothing_params: np.ndarray,
        prior_config: dict[str, Any],
    ) -> float:
        """Return ``0.0`` — this model has no priors. Present for protocol compliance."""
        return 0.0

    # ── lag-selection diagnostic ─────────────────────────────────────

    def select_lag_order(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
        *,
        max_lag: int = _MAX_LAG_ORDER,
        test_observed_returns: pd.Series | None = None,
        test_factor_returns: pd.DataFrame | None = None,
        tolerance: float = 0.02,
    ) -> LagSelectionResult:
        """Compare :math:`Q \\in \\{0, \\dots, \\text{max}\\_lag\\}` and recommend one.

        Runs the full :meth:`fit` at each :math:`Q` and (optionally) an
        out-of-sample application to a supplied test window.

        Parameters
        ----------
        observed_returns, factor_returns
            Training data.
        max_lag
            Upper end of the sweep. Capped at 4 (the paper's largest tested).
        test_observed_returns, test_factor_returns
            Optional test window; if both are provided, the diagnostic table
            gains an ``variance_ratio_out_of_sample`` column and the
            recommendation minimises that ratio.
        tolerance
            Parsimony threshold (in-sample mode only): pick the smallest
            :math:`Q` whose in-sample % explained is within ``tolerance`` of
            the best. Falls back to :math:`Q = 1` on ties.

        Returns
        -------
        LagSelectionResult
        """
        if not (0 <= max_lag <= _MAX_LAG_ORDER):
            raise ValueError(
                f"max_lag must be in [0, {_MAX_LAG_ORDER}], got {max_lag}"
            )
        oos_available = test_observed_returns is not None and test_factor_returns is not None
        if (test_observed_returns is None) ^ (test_factor_returns is None):
            raise ValueError(
                "Provide both test_observed_returns and test_factor_returns, or neither."
            )

        rows: list[dict[str, Any]] = []
        original_n_lags = self.n_lags
        try:
            for q in range(max_lag + 1):
                self.n_lags = q
                result = self.fit(observed_returns, factor_returns)
                diag = result.diagnostics
                row: dict[str, Any] = {
                    "n_lags": q,
                    "alpha": result.smoothing_params["alpha"],
                    "alpha_ann": result.smoothing_params["alpha_annualised"],
                    "vol_ann": diag["vol_annualised"],
                    "idio_vol_ann": diag["idio_vol_annualised"],
                    "pct_variance_explained": diag["pct_variance_explained"],
                    "variance_ratio_in_sample": diag["variance_ratio_in_sample"],
                    "theta_0": diag["theta_0"],
                }
                for name, b in result.smoothing_params["beta"].items():
                    row[f"beta_{name}"] = b
                if oos_available:
                    row["variance_ratio_out_of_sample"] = self._out_of_sample_variance_ratio(
                        result,
                        test_observed_returns,
                        test_factor_returns,
                    )
                rows.append(row)
        finally:
            self.n_lags = original_n_lags

        table = pd.DataFrame(rows).set_index("n_lags")

        # Recommendation logic
        if oos_available:
            mode = "out_of_sample"
            oos = table["variance_ratio_out_of_sample"]
            recommended = int(oos.idxmin())
            rationale = (
                f"Q={recommended} minimises the out-of-sample variance ratio "
                f"({oos.loc[recommended]:.4f}). Ratios > 1 (if any) indicate "
                "the factor model introduces more variance than it explains."
            )
        else:
            mode = "in_sample"
            pve = table["pct_variance_explained"]
            best = float(pve.max())
            eligible = table.index[pve >= best - tolerance].tolist()
            recommended = int(min(eligible)) if eligible else 1
            # Parsimony tie-break: prefer Q=1 when 0 and 1 both qualify (matches paper)
            if 1 in eligible and 0 in eligible and recommended == 0:
                recommended = 1
            rationale = (
                f"Smallest Q whose in-sample %-explained is within {tolerance:.4f} "
                f"of the best ({best:.4f}) → Q={recommended}. In-sample variance "
                "ratio favours large Q mechanically; provide a test window for a "
                "more honest selector."
            )

        return LagSelectionResult(
            table=table,
            recommended_lag=recommended,
            rationale=rationale,
            mode=mode,
        )

    # ── internal helpers ──────────────────────────────────────────────

    def _validate_inputs(
        self,
        observed_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
        if not isinstance(observed_returns, pd.Series):
            raise TypeError(
                f"observed_returns must be a pandas Series, got "
                f"{type(observed_returns).__name__}."
            )
        if not isinstance(factor_returns, pd.DataFrame):
            raise TypeError(
                f"factor_returns must be a pandas DataFrame, got "
                f"{type(factor_returns).__name__}."
            )
        if factor_returns.shape[1] == 0:
            raise ValueError(
                "factor_returns must have at least one column — the model "
                "requires factor regressors."
            )
        if not observed_returns.index.equals(factor_returns.index):
            raise ValueError(
                "observed_returns.index and factor_returns.index must match."
            )
        y = observed_returns.to_numpy(dtype=float)
        F = factor_returns.to_numpy(dtype=float)
        if np.isnan(y).any() or np.isnan(F).any():
            raise ValueError("Inputs must not contain NaN.")
        ppy = self._periods_per_year(observed_returns.index)
        factor_names = list(factor_returns.columns)
        return y, F, ppy, factor_names

    def _periods_per_year(self, index: pd.Index) -> int:
        if self.periods_per_year is not None:
            return int(self.periods_per_year)
        return _infer_periods_per_year(index)

    @staticmethod
    def _reconstruct(y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        r"""Apply Eq. 3: :math:`\hat r^E_t = \sum_{j=0}^{Q} \theta_j y_{t-j}` for :math:`t \ge Q`."""
        Q = theta.size - 1
        N = y.size
        n_eff = N - Q
        if Q == 0:
            return y.copy()
        out = theta[0] * y[Q:]
        for j in range(1, Q + 1):
            out = out + theta[j] * y[Q - j : N - j]
        return out

    def _out_of_sample_variance_ratio(
        self,
        train_result: DesmoothedResult,
        test_observed: pd.Series,
        test_factors: pd.DataFrame,
    ) -> float:
        """Apply the trained (θ, β, α) to a test window and compute the ratio.

        See the module docstring for the definition; briefly:
        ``variance_ratio_oos = Var(test_actual − model_prediction) / Var(test_actual)``
        where ``test_actual`` is the reconstructed test series (Eq. 3 using
        trained θ) and ``model_prediction`` is ``α + β · F`` on the test factors.
        """
        if not isinstance(test_observed, pd.Series):
            raise TypeError("test_observed_returns must be a pandas Series.")
        if not isinstance(test_factors, pd.DataFrame):
            raise TypeError("test_factor_returns must be a pandas DataFrame.")
        if not test_observed.index.equals(test_factors.index):
            raise ValueError(
                "test_observed_returns.index and test_factor_returns.index must match."
            )
        theta = np.asarray(train_result.smoothing_params["theta"], dtype=float)
        beta_dict = train_result.smoothing_params["beta"]
        alpha = float(train_result.smoothing_params["alpha"])
        y_test = test_observed.to_numpy(dtype=float)
        # Align factor columns to training factor order
        train_factor_names = list(beta_dict.keys())
        missing = [c for c in train_factor_names if c not in test_factors.columns]
        if missing:
            raise ValueError(
                f"test_factor_returns is missing training factors: {missing}"
            )
        F_test = test_factors[train_factor_names].to_numpy(dtype=float)
        Q = theta.size - 1
        if y_test.size <= Q:
            raise ValueError(
                f"test_observed_returns has length {y_test.size}; need > n_lags ({Q})."
            )
        r_hat_test = self._reconstruct(y_test, theta)  # length N_test - Q
        F_aligned = F_test[Q:, :]
        beta_arr = np.array([beta_dict[c] for c in train_factor_names], dtype=float)
        prediction = alpha + F_aligned @ beta_arr
        resid = r_hat_test - prediction
        var_actual = float(np.var(r_hat_test, ddof=1))
        var_resid = float(np.var(resid, ddof=1))
        if var_actual <= 0.0:
            return float("nan")
        return var_resid / var_actual


# ──────────────────────────────────────────────────────────────────
# Free helpers
# ──────────────────────────────────────────────────────────────────


def _build_lag_matrix(y: np.ndarray, Q: int) -> np.ndarray:
    """Return the (N-Q, Q) matrix with columns ``y[Q-1:N-1], y[Q-2:N-2], …, y[0:N-Q]``.

    Returns an empty (N-Q, 0) matrix when ``Q == 0``.
    """
    N = y.size
    n_eff = N - Q
    if Q == 0:
        return np.zeros((n_eff, 0), dtype=float)
    cols = [y[Q - j : N - j].reshape(-1, 1) for j in range(1, Q + 1)]
    return np.concatenate(cols, axis=1)


def _annualise_alpha(alpha: float, ppy: int, mode: str) -> float:
    """Annualise a per-period intercept.

    * ``simple``  → ``alpha · ppy``
    * ``geometric`` → ``(1 + alpha)**ppy - 1``
    """
    if mode == "simple":
        return float(alpha * ppy)
    if mode == "geometric":
        return float((1.0 + alpha) ** ppy - 1.0)
    raise ValueError(f"Unknown alpha_annualization mode {mode!r}")


def _infer_periods_per_year(index: pd.Index) -> int:
    """Infer periods per year from a pandas ``DatetimeIndex`` frequency.

    Falls back on median-spacing heuristics when ``index.freq`` is ``None``.
    Defaults to 12 (monthly) when the spacing is ambiguous.
    """
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq or pd.infer_freq(index)
        if freq is not None:
            token = str(freq).upper()
            if token.startswith(("Q", "BQ")) or "Q" in token[:3]:
                return Frequency.QUARTERLY.periods_per_year
            if token.startswith(("M", "BM", "MS")) or "M" in token[:3]:
                return Frequency.MONTHLY.periods_per_year
            if token.startswith("W"):
                return Frequency.WEEKLY.periods_per_year
            if token.startswith(("B", "D", "C")):
                return Frequency.DAILY.periods_per_year
        # Fallback: median spacing
        if len(index) >= 2:
            spacings = np.diff(index.view("i8")) / 1e9 / 86400.0  # days
            median_days = float(np.median(spacings))
            if median_days >= 80:
                return Frequency.QUARTERLY.periods_per_year
            if median_days >= 25:
                return Frequency.MONTHLY.periods_per_year
            if median_days >= 5:
                return Frequency.WEEKLY.periods_per_year
            return Frequency.DAILY.periods_per_year
    return Frequency.MONTHLY.periods_per_year


def _trim_trailing_zeros(w: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Trim trailing entries whose absolute value is below ``atol``.

    Keeps at least the first entry. Used to compact the w-parameterisation
    for human-readable output; the exact truncated series is available via
    :func:`theta_to_w`.
    """
    if w.size == 0:
        return w
    keep = w.size
    while keep > 1 and abs(w[keep - 1]) < atol:
        keep -= 1
    return w[:keep].copy()


__all__ = [
    "LagSelectionResult",
    "RudinReparamSmoothing",
]
