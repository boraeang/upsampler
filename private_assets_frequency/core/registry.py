"""
Preset registry for ``private_assets_frequency``.

The registry is a thin name → :class:`AssetClassConfig` lookup. Built-in
asset-class presets are *not* defined here — they live in
``pipeline/presets.py`` and call :func:`register_preset` at import time. The
separation lets users introspect the registry (``list_presets()``,
``filter_presets(asset_class='private_equity')``) without importing the full
preset table.

Examples
--------
>>> from private_assets_frequency.core.config import (
...     AssetClassConfig, BetaDist, NormalPrior, InverseGammaPrior,
... )
>>> from private_assets_frequency.core.registry import (
...     register_preset, get_preset, list_presets,
... )
>>> cfg = AssetClassConfig(
...     asset_class='private_equity',
...     smoothing_model='ar1_bayesian',
...     native_frequency='quarterly',
...     lambda_prior=BetaDist(2, 2),
...     beta_priors={'equity_market': NormalPrior(1.0, 0.5)},
...     alpha_prior=NormalPrior(0.0, 0.05),
...     sigma_eps_prior=InverseGammaPrior(3.0, 0.02),
... )
>>> register_preset('toy_buyout', cfg)
>>> get_preset('toy_buyout') is cfg
True
"""

from __future__ import annotations

from typing import Iterable

from .config import AssetClassConfig


class PresetRegistry:
    """In-memory preset store. Not thread-safe; intended for single-process use."""

    def __init__(self) -> None:
        self._presets: dict[str, AssetClassConfig] = {}

    # ── mutation ────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        config: AssetClassConfig,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register ``config`` under ``name``.

        Parameters
        ----------
        name
            Preset key (e.g. ``'us_large_buyout'``). Must be a non-empty string.
        config
            The :class:`AssetClassConfig` to store.
        overwrite
            If ``False`` (default) and ``name`` is already present, raise
            ``KeyError``. Pass ``overwrite=True`` to replace.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"Preset name must be a non-empty string, got {name!r}.")
        if not isinstance(config, AssetClassConfig):
            raise TypeError(
                f"config must be an AssetClassConfig, got {type(config).__name__}."
            )
        if name in self._presets and not overwrite:
            raise KeyError(
                f"Preset {name!r} already registered; pass overwrite=True to replace."
            )
        self._presets[name] = config

    def unregister(self, name: str) -> None:
        """Remove ``name`` from the registry. Raises ``KeyError`` if absent."""
        del self._presets[name]

    def clear(self) -> None:
        """Remove all presets. Primarily useful in tests."""
        self._presets.clear()

    # ── lookup ──────────────────────────────────────────────────────

    def get(self, name: str) -> AssetClassConfig:
        """Return the preset registered under ``name``.

        Raises
        ------
        KeyError
            If ``name`` is not registered. The exception message lists the
            closest known names.
        """
        if name in self._presets:
            return self._presets[name]
        # Build a helpful error message
        known = sorted(self._presets)
        if not known:
            raise KeyError(f"No preset {name!r} registered (registry is empty).")
        suggestions = _closest_names(name, known, k=3)
        suffix = f" Did you mean: {suggestions!r}?" if suggestions else ""
        raise KeyError(f"No preset {name!r} registered.{suffix}")

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._presets

    def __len__(self) -> int:
        return len(self._presets)

    def __iter__(self) -> Iterable[str]:
        return iter(sorted(self._presets))

    def list(self) -> list[str]:
        """Return a sorted list of all registered preset names."""
        return sorted(self._presets)

    def filter(
        self,
        *,
        asset_class: str | None = None,
        smoothing_model: str | None = None,
        native_frequency: str | None = None,
    ) -> dict[str, AssetClassConfig]:
        """Return presets matching all supplied filters."""
        out: dict[str, AssetClassConfig] = {}
        for name, cfg in self._presets.items():
            if asset_class is not None and cfg.asset_class != asset_class:
                continue
            if smoothing_model is not None and cfg.smoothing_model != smoothing_model:
                continue
            if native_frequency is not None and cfg.native_frequency != native_frequency:
                continue
            out[name] = cfg
        return out

    def items(self) -> list[tuple[str, AssetClassConfig]]:
        return [(k, self._presets[k]) for k in sorted(self._presets)]


# ──────────────────────────────────────────────────────────────────
# Module-level default registry
# ──────────────────────────────────────────────────────────────────

_DEFAULT_REGISTRY = PresetRegistry()


def get_preset(name: str) -> AssetClassConfig:
    """Return the preset ``name`` from the default registry."""
    return _DEFAULT_REGISTRY.get(name)


def register_preset(
    name: str,
    config: AssetClassConfig,
    *,
    overwrite: bool = False,
) -> None:
    """Register ``config`` under ``name`` in the default registry."""
    _DEFAULT_REGISTRY.register(name, config, overwrite=overwrite)


def unregister_preset(name: str) -> None:
    """Remove ``name`` from the default registry."""
    _DEFAULT_REGISTRY.unregister(name)


def list_presets() -> list[str]:
    """Return a sorted list of all preset names in the default registry."""
    return _DEFAULT_REGISTRY.list()


def filter_presets(
    *,
    asset_class: str | None = None,
    smoothing_model: str | None = None,
    native_frequency: str | None = None,
) -> dict[str, AssetClassConfig]:
    """Return presets in the default registry matching all supplied filters."""
    return _DEFAULT_REGISTRY.filter(
        asset_class=asset_class,
        smoothing_model=smoothing_model,
        native_frequency=native_frequency,
    )


def default_registry() -> PresetRegistry:
    """Return the module-level default registry (for inspection / tests)."""
    return _DEFAULT_REGISTRY


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _closest_names(query: str, candidates: list[str], k: int) -> list[str]:
    """Return up to ``k`` candidates closest to ``query`` by Levenshtein distance."""
    scored = sorted(((_levenshtein(query, c), c) for c in candidates))
    return [c for _, c in scored[:k]]


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein distance, O(len(a) * len(b)) time and O(len(b)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


__all__ = [
    "PresetRegistry",
    "default_registry",
    "filter_presets",
    "get_preset",
    "list_presets",
    "register_preset",
    "unregister_preset",
]
