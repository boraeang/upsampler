"""
Nowcasting models, in increasing order of ambition.

* :class:`~.naive.NaiveCarryForward` — Model 0, the bar to clear.
* :class:`~.smoothing_regression.SmoothingRegressionNowcaster` — Model 1, the
  reduced form of the library's AR(1)/Rudin smoothing models and the benchmark
  every other model is measured against.
* :class:`~.structural.StructuralSmoothingNowcaster` — Model 2, reusing an
  already-fitted desmoothing model's parameters instead of re-estimating.
* :class:`~.signature.SignatureNowcaster` — Model 3, the challenger: regression
  on path-signature features of the public market path.

Models 0–2 import and run with no optional dependencies. Model 3 needs
``scikit-learn`` (the ``nowcast`` extra) and optionally ``iisignature`` / ``esig``.
"""

from __future__ import annotations

from .naive import NaiveCarryForward
from .signature import SignatureNowcaster
from .smoothing_regression import SmoothingRegressionNowcaster
from .structural import StructuralSmoothingNowcaster

__all__ = [
    "NaiveCarryForward",
    "SignatureNowcaster",
    "SmoothingRegressionNowcaster",
    "StructuralSmoothingNowcaster",
]
