"""Tests for :mod:`src.edge_matching` (task 5)."""

import numpy as np

from src import contour_extraction as ce
from src import edge_matching as em
from src import evaluation as ev
from src import piece_description as pd
from src import segmentation as seg


def _described(rows=3, cols=4, seed=21):
    src = ev.synthetic_source_image(110 * rows, 110 * cols, seed=200 + seed)
    scrambled, gt = ev.generate_puzzle(src, rows=rows, cols=cols,
                                       rotate=True, seed=seed)
    mask = seg.foreground_mask(scrambled, "background", open_radius=1,
                               close_radius=1)
    labels, _ = seg.connected_components(mask)
    labels, stats = seg.filter_components(labels, min_area_ratio=0.3)
    pieces = ce.extract_pieces(scrambled, labels, stats=stats)
    return pieces, pd.describe_pieces(pieces), gt


# --------------------------------------------------------------------------
# the individual terms
# --------------------------------------------------------------------------
def test_admissibility_truth_table():
    T, B, F = pd.SIDE_TAB, pd.SIDE_BLANK, pd.SIDE_FLAT
    assert em.sides_admissible(T, B)
    assert em.sides_admissible(B, T)
    assert not em.sides_admissible(T, T)
    assert not em.sides_admissible(B, B)
    for other in (T, B, F):
        assert not em.sides_admissible(F, other)
        assert not em.sides_admissible(other, F)


def test_shape_distance_is_zero_for_a_perfect_fit():
    t = np.linspace(0, 1, 96)
    p = 0.3 * np.sin(np.pi * t)
    mate = -p[::-1]                     # exactly p_a(t) = -p_b(1-t)
    assert em.shape_distance(p, mate) < 1e-9


def test_shape_distance_grows_with_mismatch():
    t = np.linspace(0, 1, 96)
    p = 0.3 * np.sin(np.pi * t)
    good = -p[::-1]
    bad = -0.3 * np.sin(2 * np.pi * t)[::-1]
    assert em.shape_distance(p, good) < em.shape_distance(p, bad)


def test_colour_distance_is_zero_for_identical_reversed_strips():
    rng = np.random.default_rng(0)
    a = rng.random((64, 3, 3))
    assert em.colour_distance(a, a[::-1]) < 1e-9
    assert em.colour_distance(a, rng.random((64, 3, 3))) > 0.1


def test_length_distance_is_relative():
    assert em.length_distance(100.0, 100.0) == 0.0
    assert abs(em.length_distance(80.0, 100.0) - 0.2) < 1e-12


def test_shift_tolerance_absorbs_a_small_misalignment():
    t = np.linspace(0, 1, 96)
    p = 0.3 * np.sin(np.pi * t)
    mate = -np.roll(p, 3)[::-1]
    assert em.shape_distance(p, mate) < em.shape_distance(p, mate, max_shift=0)


def test_side_compatibility_is_infinite_for_illegal_pairs():
    _, descs, _ = _described(2, 3, 22)
    sides = [s for d in descs for s in d.sides]
    tab = next(s for s in sides if s.type == pd.SIDE_TAB)
    flat = next(s for s in sides if s.type == pd.SIDE_FLAT)
    assert not np.isfinite(em.side_compatibility(tab, flat))
    assert not np.isfinite(em.side_compatibility(tab, tab))


# --------------------------------------------------------------------------
# the full table
# --------------------------------------------------------------------------
def test_table_shape_and_self_pairs_excluded():
    _, descs, _ = _described(2, 3, 23)
    n = len(descs)
    table = em.build_compatibility(descs)
    assert table.cost.shape == (n, 4, n, 4)
    for i in range(n):
        assert not np.isfinite(table.cost[i, :, i, :]).any()


def test_table_matches_the_scalar_formula():
    _, descs, _ = _described(2, 3, 24)
    table = em.build_compatibility(descs)
    n = len(descs)
    checked = 0
    for i in range(n):
        for s in range(4):
            for j in range(n):
                for t in range(4):
                    v = table.cost[i, s, j, t]
                    if not np.isfinite(v):
                        continue
                    ref = em.side_compatibility(descs[i].sides[s],
                                                descs[j].sides[t])
                    assert abs(v - ref) < 1e-8
                    checked += 1
    assert checked > 0


def test_table_is_symmetric():
    _, descs, _ = _described(2, 3, 25)
    table = em.build_compatibility(descs)
    n = len(descs)
    a = table.cost.reshape(n * 4, n * 4)
    finite = np.isfinite(a)
    assert np.array_equal(finite, finite.T)
    assert np.max(np.abs(a[finite] - a.T[finite])) < 1e-8


def test_colour_normalisation_is_invariant_to_lighting():
    """A strip lit brighter must still match the pattern it faces."""
    rng = np.random.default_rng(0)
    strip = rng.random((64, 3, 3))
    # a gain and offset that stay inside [0, 1], so nothing is clipped --
    # clipping is a *non-linear* change and no normalisation can undo it
    lit = strip * 0.6 + 0.2
    a = em._normalise_strip(strip.reshape(1, 64, -1), "meanstd")
    b = em._normalise_strip(lit.reshape(1, 64, -1), "meanstd")
    assert np.max(np.abs(a - b)) < 0.05
    # without it, the same pair is far apart
    raw = em._normalise_strip(strip.reshape(1, 64, -1), "none")
    raw_lit = em._normalise_strip(lit.reshape(1, 64, -1), "none")
    assert np.max(np.abs(raw - raw_lit)) > 0.1


def test_colour_normalisation_keeps_true_pairs_cheapest():
    _, descs, gt = _described(3, 4, 29)
    plain = em.build_compatibility(descs)
    normed = em.build_compatibility(descs, colour_norm="meanstd")
    assoc = ev.associate_with_ground_truth([d.piece for d in descs], gt)
    a = ev.matching_accuracy(plain, descs, gt, assoc)["top1_accuracy"]
    b = ev.matching_accuracy(normed, descs, gt, assoc)["top1_accuracy"]
    # uniform synthetic lighting: normalisation must not make things worse
    assert b >= a - 0.1


def test_gradient_compatibility_rewards_a_continuing_texture():
    """A strip whose texture continues must beat one that does not."""
    m = 64
    t = np.linspace(0, 1, m)[:, None]
    # a ramp that continues smoothly across the seam, plus two impostors
    base = np.stack([t[:, 0], 1 - t[:, 0], 0.5 * np.ones(m)], axis=1)
    depths = 6
    def strip(offset, slope):
        return np.stack([base + offset + slope * d for d in range(depths)],
                        axis=1)                       # (M, D, 3)
    a = strip(0.0, -0.02)
    partner = strip(-0.02, -0.02)[::-1]               # continues a's gradient
    impostor = strip(0.4, +0.05)[::-1]                # jumps and reverses
    cols = np.stack([a, partner, impostor])
    d = em.gradient_compatibility(cols, max_shift=2)
    assert d[0, 1] < d[0, 2]


def test_gradient_compatibility_is_symmetric_and_globally_offset_invariant():
    """Symmetric, and blind to a lighting change affecting the whole scene.

    Note what it is *not*: shifting one piece alone does change its scores,
    because the prediction is compared against the neighbour's actual
    boundary colour and a per-piece offset lands in that residual.  That is
    the standard behaviour of MGC, and it is why the ``mgc+ssd`` metric pairs
    it with the mean/std-normalised SSD term, which is per-piece invariant.
    """
    rng = np.random.default_rng(1)
    cols = rng.random((5, 48, 4, 3))
    d = em.gradient_compatibility(cols, max_shift=2)
    assert np.allclose(d, d.T)
    d2 = em.gradient_compatibility(cols + 0.2, max_shift=2)
    assert np.max(np.abs(d - d2)) < 1e-6


def test_colour_metric_choices_all_produce_valid_tables():
    _, descs, _ = _described(2, 3, 30)
    n = len(descs)
    plain = em.build_compatibility(descs, colour_metric="ssd")
    for metric in ("mgc", "mgc+ssd"):
        t = em.build_compatibility(descs, colour_metric=metric)
        assert t.cost.shape == (n, 4, n, 4)
        # the same pairs stay admissible whatever the photometric term is
        assert np.array_equal(np.isfinite(t.cost), np.isfinite(plain.cost))
        fin = np.isfinite(t.cost)
        assert np.all(t.cost[fin] >= 0)
    with np.testing.assert_raises(ValueError):
        em.build_compatibility(descs, colour_metric="nonsense")


def test_weights_change_the_cost_predictably():
    _, descs, _ = _described(2, 3, 26)
    t1 = em.build_compatibility(descs, em.MatchWeights(1.0, 0.0, 0.0))
    t2 = em.build_compatibility(descs, em.MatchWeights(1.0, 1.0, 0.0))
    fin = np.isfinite(t1.cost)
    assert np.allclose(t1.cost[fin], t1.shape[fin])
    assert np.allclose(t2.cost[fin], (t1.shape + t1.colour)[fin])


def test_true_seams_are_cheaper_than_random_admissible_ones():
    _, descs, gt = _described(3, 4, 27)
    table = em.build_compatibility(descs)
    pieces = [d.piece for d in descs]
    assoc = ev.associate_with_ground_truth(pieces, gt)
    acc = ev.matching_accuracy(table, descs, gt, assoc)
    assert acc["n_true_seams"] > 10
    assert acc["top1_accuracy"] > 0.6
    assert acc["top3_accuracy"] >= acc["top1_accuracy"]


def test_best_buddies_are_mutual_and_mostly_correct():
    _, descs, gt = _described(3, 4, 28)
    table = em.build_compatibility(descs)
    buddies = em.best_buddies(table)
    assert buddies
    flat = table.cost.reshape(len(descs) * 4, -1)
    for (i, s, j, t, cost) in buddies:
        a, b = i * 4 + s, j * 4 + t
        assert int(np.argmin(flat[a])) == b
        assert int(np.argmin(flat[b])) == a

    assoc = ev.associate_with_ground_truth([d.piece for d in descs], gt)
    cells = [tuple(gt.placements[g][:2]) for g in assoc]
    correct = sum(1 for (i, s, j, t, c) in buddies
                  if abs(cells[i][0] - cells[j][0])
                  + abs(cells[i][1] - cells[j][1]) == 1)
    assert correct / len(buddies) > 0.7
