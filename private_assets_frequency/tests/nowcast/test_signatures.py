r"""
Tests for ``nowcast/signatures.py``.

Covers module-spec test 3 — the numpy backend matches ``iisignature``/``esig`` to
1e-10 for random piecewise-linear paths at levels 1-3, and Chen's identity holds
(signature of a concatenation equals the tensor product) — plus the term
selection and path construction everything downstream depends on.

Neither signature library installs in this environment (``iisignature`` fails to
build; ``esig`` needs ``numpy >= 2``, which breaks the pinned ``pandas 2.1.4``),
so the cross-check skips and the numpy backend's correctness rests on the
algebraic identities tested directly: the closed form for a straight line, the
shuffle identity :math:`S^{(i)}S^{(j)} = S^{(i,j)} + S^{(j,i)}`,
reparametrisation invariance, and Chen's identity. Those pin the computation
without reference to any library.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from private_assets_frequency.nowcast import build_information_set
from private_assets_frequency.nowcast.models.signature import feature_budget
from private_assets_frequency.nowcast.signatures import (
    MAX_SIGNATURE_LEVEL,
    FactorPCA,
    PathSpec,
    available_backends,
    build_path,
    chen_product,
    compute_signature,
    linear_term_mask,
    path_signature,
    signature_dimension,
    signature_feature_names,
    signature_numpy,
    signature_words,
    tensor_exponential,
)

AS_OF = pd.Timestamp("2026-10-07")
Q2 = pd.Period("2026Q2", freq="Q")

_LIBRARY_BACKENDS = [b for b in available_backends() if b != "numpy"]
_needs_library = pytest.mark.skipif(
    not _LIBRARY_BACKENDS,
    reason=(
        "no signature library importable (iisignature does not build here; "
        "esig requires numpy>=2, which breaks pandas 2.1.4)"
    ),
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _info(ds, **kwargs):
    params = dict(
        reported=ds.reported,
        public_factors=ds.daily_factors,
        as_of=AS_OF,
        vintages=ds.vintages,
    )
    params.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_information_set(**params)


def _random_path(n_points: int, n_channels: int, seed: int) -> np.ndarray:
    """A random piecewise-linear path starting at the origin."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_points - 1, n_channels))
    return np.vstack([np.zeros(n_channels), np.cumsum(steps, axis=0)])


def _split_levels(flat: np.ndarray, d: int, level: int) -> list[np.ndarray]:
    """Split a flat signature back into per-level blocks."""
    out: list[np.ndarray] = []
    start = 0
    for k in range(1, level + 1):
        size = d**k
        out.append(flat[start : start + size])
        start += size
    return out


# ──────────────────────────────────────────────────────────────────
# Word enumeration and the linear-term rule
# ──────────────────────────────────────────────────────────────────


class TestSignatureWords:
    def test_canonical_order(self):
        assert signature_words(2, 2) == [
            (0,),
            (1,),
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]

    def test_level_blocks_are_contiguous_and_lexicographic(self):
        words = signature_words(3, 3)
        assert len(words) == 3 + 9 + 27
        assert words[:3] == [(0,), (1,), (2,)]
        assert words[3:6] == [(0, 0), (0, 1), (0, 2)]
        assert words[12] == (0, 0, 0)
        lengths = [len(w) for w in words]
        assert lengths == sorted(lengths), "levels must be ascending"

    @pytest.mark.parametrize("bad", [0, -1])
    def test_bad_channel_count(self, bad):
        with pytest.raises(ValueError, match="n_channels must be >= 1"):
            signature_words(bad, 2)

    @pytest.mark.parametrize("bad", [0, MAX_SIGNATURE_LEVEL + 1])
    def test_bad_level(self, bad):
        with pytest.raises(ValueError, match="level must be in"):
            signature_words(2, bad)


class TestLinearTermMask:
    r"""``keep_sigs='linear'``: each non-time channel at most once."""

    def test_level_two_two_channels_drops_only_the_squared_term(self):
        words = signature_words(2, 2)
        mask = linear_term_mask(2, 2)
        kept = [words[i] for i in range(len(words)) if mask[i]]
        dropped = [words[i] for i in range(len(words)) if not mask[i]]
        assert kept == [(0,), (1,), (0, 0), (0, 1), (1, 0)]
        assert dropped == [(1, 1)]

    def test_time_repeats_are_exempt(self):
        r"""``(0,0)`` survives: repeated *time* is a polynomial weight, not a product.

        ``(0,1)`` is :math:`\int s\,dX_s` and ``(1,0)`` is
        :math:`\int X_s\,ds` — both linear in the path. Only ``(1,1)``,
        which equals :math:`\tfrac12 X_1^2`, is not.
        """
        mask = linear_term_mask(2, 3)
        words = signature_words(2, 3)
        kept = {words[i] for i in range(len(words)) if mask[i]}
        assert (0, 0) in kept
        assert (0, 0, 0) in kept
        assert (1, 1) not in kept
        assert (0, 1, 1) not in kept
        assert (1, 0, 1) not in kept
        assert (0, 1, 0) in kept

    def test_two_factor_channels_each_allowed_once(self):
        """With time + 2 factors, the cross term survives but the squares do not."""
        mask = linear_term_mask(3, 2)
        words = signature_words(3, 2)
        dropped = {words[i] for i in range(len(words)) if not mask[i]}
        assert dropped == {(1, 1), (2, 2)}
        kept = {words[i] for i in range(len(words)) if mask[i]}
        assert (1, 2) in kept and (2, 1) in kept

    def test_time_channel_none_applies_the_rule_everywhere(self):
        mask = linear_term_mask(2, 2, time_channel=None)
        words = signature_words(2, 2)
        dropped = {words[i] for i in range(len(words)) if not mask[i]}
        assert dropped == {(0, 0), (1, 1)}

    @pytest.mark.parametrize(
        "d,level,n_all,n_linear",
        [(2, 1, 2, 2), (2, 2, 6, 5), (2, 3, 14, 9), (2, 4, 30, 14), (3, 2, 12, 10), (3, 3, 39, 23)],
    )
    def test_dimension_counts(self, d, level, n_all, n_linear):
        """The counts that decide whether a configuration fits the feature budget."""
        assert signature_dimension(d, level, "all") == n_all
        assert signature_dimension(d, level, "linear") == n_linear

    def test_feature_budget_consequence(self):
        """Where the ``n_train/5`` budget actually bites, at ~90 quarters.

        Computed with :func:`feature_budget` — the function the guard itself calls
        — rather than by reconstructing the arithmetic from
        :func:`signature_dimension`. The two differ: the guard **excludes** the
        structurally constant time-only terms, because
        :class:`~...models.signature.SignatureNowcaster` drops them from the
        regressors. Rebuilding the sum by hand here is how an earlier version of
        this test came to assert a boundary one channel too tight.

        .. code-block:: text

            budget = 90/5 = 18 features (incl. s_lag1 and the intercept)

            d=2 (time + 1 factor)   linear L2 =  5   ok
            d=3 (time + 2 factors)  linear L2 = 10   ok
            d=4 (time + 3 factors)  linear L2 = 17   ok     <- 3 channels still fits
            d=5 (time + 4 factors)  linear L2 = 26   over   <- 4 is the cap at L2
            d=3 (time + 2 factors)  linear L3 = 22   over   <- 1 channel at L3
            d=2 (time + 1 factor)   all    L4 = 28   over

        The module spec's "time + 1-2 channels at level 2" is therefore
        conservative: three factor channels do fit at level 2 on a ~90-quarter
        sample. Level 3 is the tighter constraint — it admits one factor channel
        and no more.
        """
        def n_features(d, level, keep):
            return feature_budget(
                90, n_channels=d, level=level, keep_sigs=keep
            )["n_features"]

        def within(d, level, keep):
            return feature_budget(
                90, n_channels=d, level=level, keep_sigs=keep
            )["within_budget"]

        assert n_features(2, 2, "linear") == 5 and within(2, 2, "linear")
        assert n_features(3, 2, "linear") == 10 and within(3, 2, "linear")
        assert n_features(4, 2, "linear") == 17 and within(4, 2, "linear")
        # A fourth factor channel at level 2 is out of budget ...
        assert n_features(5, 2, "linear") == 26 and not within(5, 2, "linear")
        # ... level 3 is out with only two factor channels ...
        assert n_features(3, 3, "linear") == 22 and not within(3, 3, "linear")
        # ... and so is level 4 without the linear restriction.
        assert n_features(2, 4, "all") == 28 and not within(2, 4, "all")

    def test_linear_restriction_is_what_buys_the_headroom(self):
        """At level 3 with two factors, dropping non-linear terms halves the count."""
        assert signature_dimension(3, 3, "all") == 39
        assert signature_dimension(3, 3, "linear") == 23

    def test_feature_names_match_the_mask(self):
        names = signature_feature_names(["t", "x"], 2, "linear")
        assert names == ["sig[t]", "sig[x]", "sig[t,t]", "sig[t,x]", "sig[x,t]"]
        assert len(signature_feature_names(["t", "x"], 2, "all")) == 6

    def test_bad_keep_sigs(self):
        with pytest.raises(ValueError, match="keep_sigs must be"):
            signature_dimension(2, 2, "some")
        with pytest.raises(ValueError, match="keep_sigs must be"):
            signature_feature_names(["t", "x"], 2, "some")


# ──────────────────────────────────────────────────────────────────
# The numpy backend
# ──────────────────────────────────────────────────────────────────


class TestNumpyBackendClosedForms:
    def test_straight_line_level_two(self):
        r"""A line to :math:`(1, 2)`: :math:`S^k = \Delta^{\otimes k}/k!`."""
        sig = signature_numpy(np.array([[0.0, 0.0], [1.0, 2.0]]), 2)
        np.testing.assert_allclose(sig, [1.0, 2.0, 0.5, 1.0, 1.0, 2.0], atol=1e-14)

    def test_straight_line_level_three(self):
        delta = np.array([1.0, 2.0])
        sig = signature_numpy(np.array([[0.0, 0.0], delta]), 3)
        expected = np.concatenate(
            [
                delta,
                np.outer(delta, delta).reshape(-1) / 2.0,
                np.einsum("i,j,k->ijk", delta, delta, delta).reshape(-1) / 6.0,
            ]
        )
        np.testing.assert_allclose(sig, expected, atol=1e-14)

    def test_tensor_exponential_matches_a_one_segment_signature(self):
        delta = np.array([0.3, -0.7, 1.1])
        blocks = tensor_exponential(delta, 3)
        direct = signature_numpy(np.array([np.zeros(3), delta]), 3)
        np.testing.assert_allclose(np.concatenate(blocks), direct, atol=1e-14)

    def test_level_one_is_the_total_increment(self):
        """Level 1 depends only on the endpoints, whatever happens in between."""
        path = _random_path(25, 3, seed=1)
        sig = signature_numpy(path, 2)
        np.testing.assert_allclose(sig[:3], path[-1] - path[0], atol=1e-13)

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_shuffle_identity(self, d):
        r""":math:`S^{(i)} S^{(j)} = S^{(i,j)} + S^{(j,i)}` — integration by parts."""
        path = _random_path(20, d, seed=d)
        sig = signature_numpy(path, 2)
        level1 = sig[:d]
        level2 = sig[d : d + d * d].reshape(d, d)
        np.testing.assert_allclose(
            np.outer(level1, level1), level2 + level2.T, atol=1e-12
        )

    def test_reparametrisation_invariance(self):
        """Inserting points along existing segments must not change the signature."""
        path = _random_path(8, 3, seed=11)
        dense = [path[0]]
        for i in range(1, path.shape[0]):
            dense.append(0.5 * (path[i - 1] + path[i]))  # collinear midpoint
            dense.append(path[i])
        dense_arr = np.asarray(dense)
        for level in (1, 2, 3):
            np.testing.assert_allclose(
                signature_numpy(path, level),
                signature_numpy(dense_arr, level),
                atol=1e-12,
            )

    def test_repeated_points_are_ignored(self):
        """A zero increment multiplies by the identity, so duplicates are free."""
        path = _random_path(10, 2, seed=5)
        with_dupes = np.repeat(path, 3, axis=0)
        for level in (1, 2, 3):
            np.testing.assert_allclose(
                signature_numpy(path, level),
                signature_numpy(with_dupes, level),
                atol=1e-13,
            )

    def test_single_point_path_is_all_zeros(self):
        sig = signature_numpy(np.zeros((1, 3)), 2)
        assert sig.shape == (3 + 9,)
        assert not sig.any()

    def test_constant_path_is_all_zeros(self):
        sig = signature_numpy(np.full((12, 2), 1.7), 3)
        assert not sig.any()

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_output_length(self, level):
        d = 3
        sig = signature_numpy(_random_path(6, d, seed=level), level)
        assert sig.size == sum(d**k for k in range(1, level + 1))
        assert sig.size == len(signature_words(d, level))

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            signature_numpy(np.zeros(5), 2)
        with pytest.raises(ValueError, match="non-finite"):
            signature_numpy(np.array([[0.0, 0.0], [np.nan, 1.0]]), 2)
        with pytest.raises(ValueError, match="level must be in"):
            signature_numpy(np.zeros((3, 2)), 0)
        with pytest.raises(ValueError, match="level must be in"):
            signature_numpy(np.zeros((3, 2)), MAX_SIGNATURE_LEVEL + 1)


class TestChenIdentity:
    """Spec test 3, second half: signature of a concatenation = tensor product."""

    @pytest.mark.parametrize("level", [1, 2, 3])
    @pytest.mark.parametrize("d", [2, 3])
    def test_concatenation_equals_the_product(self, level, d):
        path = _random_path(16, d, seed=100 + level * 10 + d)
        split = 9
        left, right = path[: split + 1], path[split:]
        assert np.array_equal(left[-1], right[0]), "segments must share a point"

        full = signature_numpy(path, level)
        product = np.concatenate(
            chen_product(
                _split_levels(signature_numpy(left, level), d, level),
                _split_levels(signature_numpy(right, level), d, level),
                level,
            )
        )
        np.testing.assert_allclose(full, product, atol=1e-12)

    def test_three_way_associativity(self):
        d, level = 2, 3
        path = _random_path(19, d, seed=77)
        a, b, c = path[:7], path[6:13], path[12:]
        sa = _split_levels(signature_numpy(a, level), d, level)
        sb = _split_levels(signature_numpy(b, level), d, level)
        sc = _split_levels(signature_numpy(c, level), d, level)
        left_first = chen_product(chen_product(sa, sb, level), sc, level)
        right_first = chen_product(sa, chen_product(sb, sc, level), level)
        np.testing.assert_allclose(
            np.concatenate(left_first), np.concatenate(right_first), atol=1e-12
        )
        np.testing.assert_allclose(
            np.concatenate(left_first), signature_numpy(path, level), atol=1e-12
        )

    def test_identity_element(self):
        """A degenerate segment leaves the signature unchanged."""
        d, level = 3, 2
        path = _random_path(10, d, seed=3)
        sig = _split_levels(signature_numpy(path, level), d, level)
        identity = _split_levels(
            signature_numpy(np.zeros((2, d)), level), d, level
        )
        np.testing.assert_allclose(
            np.concatenate(chen_product(sig, identity, level)),
            np.concatenate(sig),
            atol=1e-14,
        )

    def test_level_mismatch_rejected(self):
        a = _split_levels(signature_numpy(_random_path(5, 2, 1), 2), 2, 2)
        b = _split_levels(signature_numpy(_random_path(5, 2, 2), 3), 2, 3)
        with pytest.raises(ValueError, match="must carry 3 levels"):
            chen_product(a, b, 3)


class TestLibraryCrossCheck:
    """Spec test 3, first half — skipped when no library is importable."""

    @_needs_library
    @pytest.mark.parametrize("level", [1, 2, 3])
    @pytest.mark.parametrize("d", [2, 3])
    @pytest.mark.parametrize("backend", _LIBRARY_BACKENDS)
    def test_numpy_matches_the_library(self, backend, d, level):
        for seed in range(5):
            path = _random_path(14, d, seed=1000 + seed)
            np.testing.assert_allclose(
                compute_signature(path, level, backend="numpy"),
                compute_signature(path, level, backend=backend),
                atol=1e-10,
                rtol=0,
            )

    def test_numpy_is_always_available(self):
        assert "numpy" in available_backends()
        assert available_backends()[-1] == "numpy"

    def test_auto_backend_produces_a_result(self):
        path = _random_path(10, 2, seed=9)
        auto = compute_signature(path, 2)
        explicit = compute_signature(path, 2, backend="numpy")
        if available_backends() == ("numpy",):
            np.testing.assert_array_equal(auto, explicit)
        else:
            np.testing.assert_allclose(auto, explicit, atol=1e-10)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="unknown signature backend"):
            compute_signature(_random_path(5, 2, 1), 2, backend="sigkernel")

    def test_keep_sigs_filtering_matches_the_mask(self):
        path = _random_path(12, 3, seed=4)
        full = compute_signature(path, 2, backend="numpy", keep_sigs="all")
        linear = compute_signature(path, 2, backend="numpy", keep_sigs="linear")
        mask = linear_term_mask(3, 2)
        np.testing.assert_array_equal(linear, full[mask])
        assert linear.size == signature_dimension(3, 2, "linear")


# ──────────────────────────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────────────────────────


class TestFactorPCA:
    @pytest.fixture
    def panel(self):
        rng = np.random.default_rng(0)
        common = rng.standard_normal(500)
        return pd.DataFrame(
            {
                "a": common + 0.1 * rng.standard_normal(500),
                "b": 0.8 * common + 0.1 * rng.standard_normal(500),
                "c": rng.standard_normal(500),
            }
        )

    def test_first_component_captures_the_common_factor(self, panel):
        pca = FactorPCA(n_components=1).fit(panel)
        assert pca.explained_variance_ratio_[0] > 0.5
        projected = pca.transform(panel)
        assert list(projected.columns) == ["pc1"]
        assert abs(np.corrcoef(projected["pc1"], panel["a"])[0, 1]) > 0.8

    def test_components_are_orthonormal(self, panel):
        pca = FactorPCA(n_components=3).fit(panel)
        gram = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-12)

    def test_transform_is_deterministic_and_centred(self, panel):
        pca = FactorPCA(n_components=2).fit(panel)
        first = pca.transform(panel)
        second = pca.transform(panel)
        pd.testing.assert_frame_equal(first, second)
        np.testing.assert_allclose(first.mean().to_numpy(), 0.0, atol=1e-12)

    def test_rotation_fitted_on_a_subset_ignores_later_rows(self, panel):
        """The leakage property: a training-only fit must not move with the tail."""
        train = panel.iloc[:200]
        pca = FactorPCA(n_components=2).fit(train)
        components = pca.components_.copy()
        mean = pca.mean_.copy()

        perturbed = panel.copy()
        perturbed.iloc[200:] *= 50.0
        pca_again = FactorPCA(n_components=2).fit(perturbed.iloc[:200])
        np.testing.assert_array_equal(pca_again.components_, components)
        np.testing.assert_array_equal(pca_again.mean_, mean)

    def test_transform_before_fit_raises(self, panel):
        with pytest.raises(RuntimeError, match="call fit\\(\\) before transform"):
            FactorPCA(n_components=1).transform(panel)

    def test_missing_column_at_transform_raises(self, panel):
        pca = FactorPCA(n_components=1).fit(panel)
        with pytest.raises(KeyError, match="missing columns"):
            pca.transform(panel.drop(columns="b"))

    def test_too_many_components_rejected(self, panel):
        with pytest.raises(ValueError, match="n_components must be in 1..3"):
            FactorPCA(n_components=4).fit(panel)

    def test_non_frame_rejected(self):
        with pytest.raises(TypeError, match="must be a DataFrame"):
            FactorPCA(n_components=1).fit(np.zeros((10, 2)))

    def test_too_few_rows_rejected(self, panel):
        with pytest.raises(ValueError, match="at least 2 rows"):
            FactorPCA(n_components=1).fit(panel.iloc[:1])


# ──────────────────────────────────────────────────────────────────
# PathSpec
# ──────────────────────────────────────────────────────────────────


class TestPathSpec:
    def test_defaults_match_the_spec(self):
        spec = PathSpec()
        assert spec.lookback == 1
        assert spec.path_frequency == "daily"
        assert spec.level_channel == "log"
        assert spec.include_time is True
        assert spec.basepoint is True
        assert spec.missing == "ffill"
        assert spec.n_components is None

    def test_channel_names_and_count(self):
        spec = PathSpec()
        assert spec.channel_names(["eq", "cr"]) == ["time", "eq", "cr"]
        assert spec.n_channels(2) == 3
        assert spec.n_channels(2, n_early_signals=1) == 4
        no_time = PathSpec(include_time=False)
        assert no_time.channel_names(["eq"]) == ["eq"]
        assert no_time.n_channels(1) == 1

    def test_pca_replaces_factor_channel_names(self):
        spec = PathSpec(n_components=2)
        assert spec.channel_names(["eq", "cr", "rate"]) == ["time", "pc1", "pc2"]
        assert spec.n_channels(3) == 3

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"lookback": 0}, "lookback must be >= 1"),
            ({"path_frequency": "weekly"}, "path_frequency must be"),
            ({"level_channel": "pct"}, "level_channel must be"),
            ({"missing": "drop"}, "missing must be"),
            ({"n_components": 0}, "n_components must be >= 1"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PathSpec(**kwargs)


# ──────────────────────────────────────────────────────────────────
# Path construction
# ──────────────────────────────────────────────────────────────────


class TestBuildPath:
    @pytest.mark.parametrize("lookback,expected_start", [(1, "2026-04-01"), (2, "2026-01-01"), (4, "2025-07-01")])
    def test_window_spans_the_requested_quarters(
        self, motivating_dataset, lookback, expected_start
    ):
        info = _info(motivating_dataset)
        _, _, diag = build_path(info, Q2, PathSpec(lookback=lookback))
        assert diag["window_start"] == pd.Timestamp(expected_start)
        assert diag["window_end"].normalize() == pd.Timestamp("2026-06-30")
        assert not diag["clipped_by_as_of"]

    def test_window_never_reaches_past_as_of(self, motivating_dataset):
        """The open quarter's window is clipped at ``as_of``, not at quarter end."""
        info = _info(motivating_dataset)
        q4 = pd.Period("2026Q4", freq="Q")
        path, _, diag = build_path(info, q4, PathSpec())
        assert diag["window_end"] <= AS_OF
        assert diag["clipped_by_as_of"]
        assert diag["n_path_observations"] < 10
        assert np.all(np.isfinite(path))

    def test_path_shape_and_channels(self, motivating_dataset):
        info = _info(motivating_dataset)
        path, names, diag = build_path(info, Q2, PathSpec())
        assert names == ["time", "equity_market"]
        # Basepoint plus one point per business day in the quarter.
        assert path.shape == (diag["n_path_observations"] + 1, 2)
        assert path[0].tolist() == [0.0, 0.0]

    def test_time_channel_is_rescaled_to_unit_interval(self, motivating_dataset):
        info = _info(motivating_dataset)
        path, _, _ = build_path(info, Q2, PathSpec())
        time = path[:, 0]
        assert time[0] == 0.0
        assert time[-1] == pytest.approx(1.0)
        assert np.all(np.diff(time) >= 0.0)

    def test_simple_level_channel_gives_the_quarter_return_exactly(
        self, motivating_dataset
    ):
        r"""The property the nesting test rests on: level-1 term :math:`= F_Q`."""
        info = _info(motivating_dataset)
        expected = float(info.factor_row(Q2)["equity_market"])
        features, names, _ = path_signature(
            info, Q2, PathSpec(level_channel="simple"), level=1, keep_sigs="all"
        )
        value = features[names.index("sig[equity_market]")]
        assert value == expected, "must be exact, not approximate"

    def test_log_level_channel_gives_the_log_quarter_return_exactly(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        expected = float(np.log1p(info.factor_row(Q2)["equity_market"]))
        features, names, _ = path_signature(
            info, Q2, PathSpec(level_channel="log"), level=1, keep_sigs="all"
        )
        assert features[names.index("sig[equity_market]")] == pytest.approx(
            expected, abs=1e-15
        )

    def test_basepoint_is_what_makes_the_level_one_term_complete(
        self, motivating_dataset
    ):
        """Without it the path starts at day one's level and misses that return."""
        info = _info(motivating_dataset)
        f_q = float(info.factor_row(Q2)["equity_market"])
        with_bp, names, _ = path_signature(
            info, Q2, PathSpec(level_channel="simple"), level=1, keep_sigs="all"
        )
        without_bp, _, _ = path_signature(
            info,
            Q2,
            PathSpec(level_channel="simple", basepoint=False),
            level=1,
            keep_sigs="all",
        )
        idx = names.index("sig[equity_market]")
        assert with_bp[idx] == f_q
        assert without_bp[idx] != pytest.approx(f_q, abs=1e-12)

    def test_monthly_path_compounds_to_the_same_quarter_return(
        self, motivating_dataset
    ):
        """Changing the path resolution must not change the window's total return."""
        info = _info(motivating_dataset)
        daily, names, d_diag = path_signature(
            info, Q2, PathSpec(level_channel="simple"), level=1, keep_sigs="all"
        )
        monthly, _, m_diag = path_signature(
            info,
            Q2,
            PathSpec(level_channel="simple", path_frequency="monthly"),
            level=1,
            keep_sigs="all",
        )
        idx = names.index("sig[equity_market]")
        assert monthly[idx] == pytest.approx(daily[idx], rel=1e-12)
        assert m_diag["n_path_observations"] == 3
        assert d_diag["n_path_observations"] > 50

    def test_monthly_and_daily_differ_at_level_two(self, motivating_dataset):
        """Resolution does change the *path*, which is the point of level 2."""
        info = _info(motivating_dataset)
        daily, names, _ = path_signature(info, Q2, PathSpec(), level=2)
        monthly, _, _ = path_signature(
            info, Q2, PathSpec(path_frequency="monthly"), level=2
        )
        idx = names.index("sig[equity_market,time]")
        assert daily[idx] != pytest.approx(monthly[idx], rel=1e-6)

    def test_rectilinear_doubles_the_points_and_moves_one_axis_at_a_time(
        self, motivating_dataset
    ):
        info = _info(motivating_dataset)
        plain, _, _ = build_path(info, Q2, PathSpec())
        recti, _, _ = build_path(info, Q2, PathSpec(missing="rectilinear"))
        assert recti.shape[0] == 2 * plain.shape[0] - 1
        increments = np.diff(recti, axis=0)
        time_moves = increments[:, 0] != 0.0
        data_moves = np.abs(increments[:, 1:]).sum(axis=1) != 0.0
        assert not np.any(time_moves & data_moves), (
            "a rectilinear path never moves time and data in the same segment"
        )

    def test_rectilinear_preserves_the_total_increment(self, motivating_dataset):
        info = _info(motivating_dataset)
        plain, _, _ = build_path(info, Q2, PathSpec(level_channel="simple"))
        recti, _, _ = build_path(
            info, Q2, PathSpec(level_channel="simple", missing="rectilinear")
        )
        np.testing.assert_allclose(
            recti[-1] - recti[0], plain[-1] - plain[0], atol=1e-14
        )

    def test_no_time_channel(self, motivating_dataset):
        info = _info(motivating_dataset)
        path, names, _ = build_path(info, Q2, PathSpec(include_time=False))
        assert names == ["equity_market"]
        assert path.shape[1] == 1

    def test_two_factor_panel_gives_three_channels(
        self, linear_dataset_two_factor
    ):
        info = _info(
            linear_dataset_two_factor,
            as_of=linear_dataset_two_factor.reported.index[-1].end_time
            + pd.Timedelta(days=200),
        )
        quarter = pd.PeriodIndex(info.reported.index, freq="Q")[-1]
        path, names, _ = build_path(info, quarter, PathSpec())
        assert names == ["time", "equity_market", "credit_proxy"]
        assert path.shape[1] == 3

    def test_pca_channels_require_a_fitted_rotation(self, linear_dataset_two_factor):
        info = _info(
            linear_dataset_two_factor,
            as_of=linear_dataset_two_factor.reported.index[-1].end_time
            + pd.Timedelta(days=200),
        )
        quarter = pd.PeriodIndex(info.reported.index, freq="Q")[-1]
        with pytest.raises(ValueError, match="needs a FactorPCA that has already"):
            build_path(info, quarter, PathSpec(n_components=1))
        pca = FactorPCA(n_components=1).fit(info.public_factors.iloc[:2000])
        path, names, diag = build_path(
            info, quarter, PathSpec(n_components=1), pca=pca
        )
        assert names == ["time", "pc1"]
        assert path.shape[1] == 2
        assert diag["n_components"] == 1

    def test_early_signal_channel_is_added(self, early_signal_dataset):
        as_of = early_signal_dataset.reported.index[-1].end_time + pd.Timedelta(
            days=200
        )
        info = _info(
            early_signal_dataset,
            as_of=as_of,
            early_signals=early_signal_dataset.early_signals,
        )
        quarter = pd.PeriodIndex(info.reported.index, freq="Q")[-1]
        path, names, diag = build_path(info, quarter, PathSpec())
        assert names == ["time", "equity_market", "listed_pe_nav"]
        assert diag["early_signal_columns"] == ("listed_pe_nav",)
        assert path.shape[1] == 3

    def test_early_signals_can_be_switched_off(self, early_signal_dataset):
        as_of = early_signal_dataset.reported.index[-1].end_time + pd.Timedelta(
            days=200
        )
        info = _info(
            early_signal_dataset,
            as_of=as_of,
            early_signals=early_signal_dataset.early_signals,
        )
        quarter = pd.PeriodIndex(info.reported.index, freq="Q")[-1]
        _, names, _ = build_path(
            info, quarter, PathSpec(early_signal_columns=())
        )
        assert names == ["time", "equity_market"]

    def test_empty_window_raises(self, motivating_dataset):
        info = _info(motivating_dataset)
        ancient = pd.Period("1995Q1", freq="Q")
        with pytest.raises(ValueError, match="no public factor observations"):
            build_path(info, ancient, PathSpec())

    def test_path_is_independent_of_post_as_of_data(self, motivating_dataset):
        """The leakage property, at the path level."""
        base_info = _info(motivating_dataset)
        base_path, _, _ = build_path(base_info, Q2, PathSpec())

        perturbed = motivating_dataset.daily_factors.copy()
        perturbed.loc[perturbed.index > AS_OF] += 0.25
        other_info = _info(motivating_dataset, public_factors=perturbed)
        other_path, _, _ = build_path(other_info, Q2, PathSpec())
        np.testing.assert_array_equal(base_path, other_path)


class TestPathSignature:
    def test_names_align_with_values(self, motivating_dataset):
        info = _info(motivating_dataset)
        features, names, diag = path_signature(info, Q2, PathSpec(), level=2)
        assert len(names) == features.size == diag["n_features"]
        assert names == signature_feature_names(["time", "equity_market"], 2, "linear")
        assert diag["level"] == 2
        assert diag["keep_sigs"] == "linear"
        assert diag["backend"] in available_backends()

    def test_linear_is_the_default(self, motivating_dataset):
        info = _info(motivating_dataset)
        linear, _, _ = path_signature(info, Q2, PathSpec(), level=2)
        every, _, _ = path_signature(info, Q2, PathSpec(), level=2, keep_sigs="all")
        assert linear.size == 5 and every.size == 6

    def test_time_level_one_term_is_one_for_a_complete_window(
        self, motivating_dataset
    ):
        """Hence it is collinear with an intercept, and gets dropped downstream."""
        info = _info(motivating_dataset)
        features, names, _ = path_signature(info, Q2, PathSpec(), level=1)
        assert features[names.index("sig[time]")] == pytest.approx(1.0)

    def test_no_time_channel_drops_the_rule_exemption(self, motivating_dataset):
        info = _info(motivating_dataset)
        features, names, _ = path_signature(
            info, Q2, PathSpec(include_time=False), level=2, keep_sigs="linear"
        )
        # One channel, each allowed once: only sig[equity_market] survives.
        assert names == ["sig[equity_market]"]
        assert features.size == 1


class TestPathInformationContent:
    r"""The within-quarter path really does carry what level 1 cannot see.

    Pre-registers the premise of spec test 6: if these correlations were not
    ordered this way, a signature model failing to beat the smoothing regression
    on the path-dependent DGP would say nothing about the model.
    """

    def test_level_two_term_tracks_the_average_level_statistic(self, path_dataset):
        as_of = path_dataset.reported.index[-1].end_time + pd.Timedelta(days=200)
        info = _info(path_dataset, as_of=as_of)
        statistic = path_dataset.params["path_statistic"]["equity_market"]

        quarters = pd.PeriodIndex(info.reported.index, freq="Q")[-40:]
        rows = []
        for quarter in quarters:
            features, names, _ = path_signature(
                info, quarter, PathSpec(), level=2, keep_sigs="linear"
            )
            lookup = {names[i]: features[i] for i in range(len(names))}
            rows.append(
                (
                    float(np.log1p(statistic.loc[quarter])),
                    lookup["sig[equity_market,time]"],
                    lookup["sig[equity_market]"],
                )
            )
        arr = np.array(rows)
        corr_level2 = abs(np.corrcoef(arr[:, 1], arr[:, 0])[0, 1])
        corr_level1 = abs(np.corrcoef(arr[:, 2], arr[:, 0])[0, 1])
        assert corr_level2 > 0.99, corr_level2
        assert corr_level2 > corr_level1 + 0.05, (corr_level2, corr_level1)

    def test_on_the_linear_dgp_level_one_already_suffices(self, motivating_dataset):
        """The mirror image: no extra path signal to find, so no edge to expect."""
        info = _info(motivating_dataset)
        quarters = pd.PeriodIndex(info.reported.index, freq="Q")[-40:]
        rows = []
        for quarter in quarters:
            features, names, _ = path_signature(
                info, quarter, PathSpec(), level=2, keep_sigs="linear"
            )
            lookup = {names[i]: features[i] for i in range(len(names))}
            rows.append(
                (
                    float(np.log1p(motivating_dataset.quarterly_factors.loc[quarter, "equity_market"])),
                    lookup["sig[equity_market]"],
                )
            )
        arr = np.array(rows)
        assert abs(np.corrcoef(arr[:, 1], arr[:, 0])[0, 1]) > 0.999
