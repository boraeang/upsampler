"""
Nowcasting of lagged private markets index returns.

Produces provisional estimates of private markets index returns that have not
yet been published, using public market data that is already available. See
``DESIGN_NOTES.md`` in this package for the ways such a module can silently
produce over-optimistic validation results, and what is done about each.

The entry point is :func:`build_information_set`, which defines what is knowable
at a given ``as_of`` date. Every nowcasting model consumes that object and
nothing else, which is what makes the no-leakage guarantee structural rather
than a matter of discipline.

Every model imports and fits with no optional dependencies: Model 3
(:class:`~.models.signature.SignatureNowcaster`) needs ``scikit-learn`` only for
``l1_ratio > 0``, since its default ridge path is pure numpy. Install the extra
with ``pip install private_assets_frequency[nowcast]``.

Examples
--------
>>> from private_assets_frequency.nowcast import build_information_set
>>> from private_assets_frequency.nowcast.models import (            # doctest: +SKIP
...     NaiveCarryForward, SmoothingRegressionNowcaster,
... )
>>> info = build_information_set(                                    # doctest: +SKIP
...     reported=pe_index, public_factors=daily_factors,
...     as_of="2026-10-07", vintages=pe_vintages,
... )
>>> model = SmoothingRegressionNowcaster().fit(info)                 # doctest: +SKIP
>>> model.predict(info, info.unpublished_quarters())                # doctest: +SKIP
"""

from __future__ import annotations

from .base import (
    INTERVAL_LEVELS,
    Nowcaster,
    NowcastResult,
    NowcastTarget,
    make_nowcast_result,
)
from .evaluation import (
    EvaluationData,
    EvaluationResult,
    NestedSelectionError,
    diebold_mariano,
    rolling_origin_evaluation,
)
from .information_set import (
    DEFAULT_PUBLICATION_LAG_DAYS,
    InformationSet,
    NowcastDataModeWarning,
    build_information_set,
    first_print_values,
    latest_values,
    quarterly_from_high_frequency,
    resolve_factor_columns,
)
from .integration import (
    extend_with_nowcasts,
    provisional_markers,
    provisional_quarters,
    reconcile,
    reconciliation_log,
    to_pipeline_returns,
)
from .models import (
    NaiveCarryForward,
    SignatureNowcaster,
    SmoothingRegressionNowcaster,
    StructuralSmoothingNowcaster,
)
from .signatures import (
    FactorPCA,
    PathSpec,
    available_backends,
    build_path,
    compute_signature,
    linear_term_mask,
    path_signature,
    signature_dimension,
    signature_numpy,
)
from .uncertainty import (
    DEFAULT_MIN_OOS,
    ResidualPool,
    ResidualPoolWarning,
    build_residual_pool,
    rolling_origin_backtest,
)

__all__ = [
    "to_pipeline_returns",
    "reconciliation_log",
    "reconcile",
    "provisional_quarters",
    "provisional_markers",
    "extend_with_nowcasts",
    "rolling_origin_evaluation",
    "diebold_mariano",
    "NestedSelectionError",
    "EvaluationResult",
    "EvaluationData",
    "SignatureNowcaster",
    "signature_numpy",
    "signature_dimension",
    "path_signature",
    "linear_term_mask",
    "compute_signature",
    "build_path",
    "available_backends",
    "PathSpec",
    "FactorPCA",
    "DEFAULT_MIN_OOS",
    "DEFAULT_PUBLICATION_LAG_DAYS",
    "INTERVAL_LEVELS",
    "InformationSet",
    "NaiveCarryForward",
    "NowcastDataModeWarning",
    "NowcastResult",
    "NowcastTarget",
    "Nowcaster",
    "ResidualPool",
    "ResidualPoolWarning",
    "SmoothingRegressionNowcaster",
    "StructuralSmoothingNowcaster",
    "build_information_set",
    "build_residual_pool",
    "first_print_values",
    "latest_values",
    "make_nowcast_result",
    "quarterly_from_high_frequency",
    "resolve_factor_columns",
    "rolling_origin_backtest",
]
