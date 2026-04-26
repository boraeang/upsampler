# CLAUDE.md

## Project Specification

The full specification for this project is at:
`docs/claude_code_prompt_private_assets_frequency_v2.md`

Read it in full before implementing anything. It contains:
- Architecture (5-stage pipeline)
- Four pluggable smoothing models (AR(1), MA(q), Threshold AR(1), NoSmoothing)
- Asset class presets (PE, Infrastructure, Credit, Hedge Funds, Real Estate)
- API design, testing strategy, and build order
- Decision priorities, fallback mechanisms, and sanity anchors

## Decision Priorities (always apply)

1. Statistical validity > architectural purity
2. Numerical stability > theoretical elegance
3. Robustness to small samples > asymptotic correctness
4. Interpretability > marginal accuracy gains

## Rules

- Python 3.10+, type hints throughout
- Dependencies: numpy, scipy, pandas, statsmodels only (no PyMC/Stan/MCMC)
- NumPy-style docstrings on every public function
- Write and run tests before moving to the next module
- Every stage must validate its outputs before passing downstream
- Use `uncertainty_mode` and `FallbackPolicy` as specified in the prompt

## Current Build Phase

Phase: [UPDATE THIS AS YOU PROGRESS]
Next module to build: [UPDATE THIS]