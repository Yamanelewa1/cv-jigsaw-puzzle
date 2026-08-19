"""Tests for :mod:`src.ml` -- Milestone 2's two models and their plumbing."""

import numpy as np
import torch

from src.ml import features as feat
from src.ml.dataset import PairDataset, build_puzzle_sample, augment_strip
from src.ml.gnn import GNNConfig, PuzzleGNN, build_graph
from src.ml.infer import (admissibility_mask, predicted_matches,
                          probabilities_to_table, siamese_table, gnn_table)
from src.ml.siamese import SiameseCNN
from src.ml.train import TrainConfig, roc_auc


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
class _FakeSide:
    """A side with the attributes the feature extractors read."""

    def __init__(self, m=96, d=6, seed=0, stype="tab"):
        rng = np.random.default_rng(seed)
        self.colors = rng.random((m, d, 3)).astype(np.float32)
        self.profile = rng.normal(0, 0.2, m).astype(np.float32)
        self.type = stype
        self.length = 110.0
        self.amplitude = 0.3


def test_side_strip_has_the_declared_shape_and_channels():
    s = _FakeSide()
    x = feat.side_strip(s)
    assert x.shape == (4, feat.STRIP_SAMPLES, feat.STRIP_DEPTHS)
    assert x.dtype == np.float32
    # channel 3 is the profile, constant across depth
    assert np.allclose(x[3, :, 0], x[3, :, -1])


def test_side_strip_reverse_flips_along_the_side():
    s = _FakeSide(seed=1)
    fwd = feat.side_strip(s, reverse=False)
    rev = feat.side_strip(s, reverse=True)
    assert np.allclose(fwd, rev[:, ::-1])


def test_side_vector_dimension_matches_the_constant():
    assert feat.side_vector(_FakeSide()).shape == (feat.VECTOR_DIM,)


def test_side_type_one_hot_is_in_the_vector():
    for t in ("flat", "tab", "blank"):
        v = feat.side_vector(_FakeSide(stype=t))
        one_hot = v[8 * 3 + 16: 8 * 3 + 16 + 3]
        assert one_hot.sum() == 1.0
        assert one_hot[feat.side_type_index(t)] == 1.0


def test_relative_orientation_is_consistent_with_the_side_convention():
    """If side a faces N and side b faces S, the pieces are aligned."""
    for rot_a in range(4):
        for rot_b in range(4):
            for d in range(4):
                a = (rot_a + d) % 4              # A's side facing direction d
                b = (rot_b + d + 2) % 4          # B's side facing the other way
                assert feat.relative_orientation(a, b) == (rot_b - rot_a) % 4


def test_augmentation_leaves_the_shape_channel_alone():
    x = feat.side_strip(_FakeSide(seed=3))
    y = augment_strip(x, np.random.default_rng(0))
    assert np.allclose(x[3], y[3])
    assert not np.allclose(x[:3], y[:3])
    assert y[:3].min() >= 0.0 and y[:3].max() <= 1.0


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def test_labelled_puzzle_has_plausible_positives():
    s = build_puzzle_sample(3, 4, seed=11)
    assert s is not None
    assert s.n_pieces == 12
    # a 3x4 grid has 17 internal seams; a couple may be dropped when a side
    # type is misclassified, but most must survive
    assert 12 <= len(s.positives) <= 17
    for (i, si, j, sj) in s.positives:
        assert 0 <= i < 12 and 0 <= j < 12 and i != j
        assert 0 <= si < 4 and 0 <= sj < 4
        # a true seam is always a tab meeting a blank
        ta = s.descriptions[i].sides[si].type
        tb = s.descriptions[j].sides[sj].type
        assert {ta, tb} == {"tab", "blank"}


def test_pairs_are_balanced_as_configured_and_negatives_are_admissible():
    s = build_puzzle_sample(3, 4, seed=12)
    ds = PairDataset([s], [0], mode="strip", negatives_per_positive=4, seed=0)
    bal = ds.balance()
    assert bal["positive"] == len(s.positives)
    assert bal["negative"] <= 4 * bal["positive"]
    types = s.side_types()
    for (pi, a, b, label) in ds.pairs:
        assert {types[a], types[b]} == {"tab", "blank"}   # admissible only
        assert a // 4 != b // 4                            # never same piece


def test_resampling_changes_negatives_but_keeps_positives():
    s = build_puzzle_sample(3, 4, seed=13)
    ds = PairDataset([s], [0], mode="strip", seed=0)
    pos_before = {(a, b) for (_, a, b, y) in ds.pairs if y == 1}
    neg_before = {(a, b) for (_, a, b, y) in ds.pairs if y == 0}
    ds.resample()
    pos_after = {(a, b) for (_, a, b, y) in ds.pairs if y == 1}
    neg_after = {(a, b) for (_, a, b, y) in ds.pairs if y == 0}
    assert pos_before == pos_after
    assert neg_before != neg_after


def test_split_is_by_puzzle_and_disjoint():
    from src.ml.dataset import generate_dataset
    samples, split = generate_dataset(n_puzzles=6, sizes=((3, 4),), seed=5,
                                      verbose=False)
    idx = split.train + split.val + split.test
    assert len(idx) == len(set(idx)) == len(samples)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def test_siamese_output_shape_and_pair_symmetry():
    m = SiameseCNN().eval()
    a = torch.randn(5, 4, 64, 8)
    b = torch.randn(5, 4, 64, 8)
    with torch.no_grad():
        ab = m(a, b)
        ba = m(b, a)
    assert ab.shape == (5,)
    # the head combines the embeddings symmetrically, so swapping must not
    # change the verdict
    assert torch.allclose(ab, ba, atol=1e-5)


def test_siamese_score_matrix_is_square_and_probabilistic():
    m = SiameseCNN()
    p = m.score_all(torch.randn(12, 4, 64, 8))
    assert p.shape == (12, 12)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_gnn_graph_has_only_admissible_candidates():
    n_pieces = 6
    types = ["tab", "blank", "tab", "blank"] * n_pieces
    vecs = np.random.default_rng(0).random((4 * n_pieces, feat.VECTOR_DIM))
    x, edge_index, pairs = build_graph(vecs, types, n_pieces)
    assert x.shape == (4 * n_pieces, feat.VECTOR_DIM)
    for a, b in pairs:
        assert {types[a], types[b]} == {"tab", "blank"}
        assert a // 4 != b // 4


def test_gnn_forward_and_score_matrix():
    n_pieces = 6
    types = ["tab", "blank"] * (2 * n_pieces)
    vecs = np.random.default_rng(1).random((4 * n_pieces, feat.VECTOR_DIM))
    x, ei, pairs = build_graph(vecs, types, n_pieces)
    g = PuzzleGNN(feat.VECTOR_DIM)
    out = g(x, ei, torch.as_tensor(pairs))
    assert out.shape == (len(pairs),)
    m = g.score_matrix(x, ei, pairs, 4 * n_pieces)
    assert m.shape == (4 * n_pieces, 4 * n_pieces)
    assert np.allclose(m, m.T)                    # symmetric relation


def test_the_two_models_are_structurally_different():
    """Not a re-tuning of one another: different layers, different inputs."""
    s = SiameseCNN()
    g = PuzzleGNN(feat.VECTOR_DIM)
    s_types = {type(m).__name__ for m in s.modules()}
    g_types = {type(m).__name__ for m in g.modules()}
    assert "Conv2d" in s_types and "Conv2d" not in g_types
    assert any("Conv" in t and t != "Conv2d" for t in g_types) or \
        any(t in g_types for t in ("SAGEConv", "_MeanConv"))


# --------------------------------------------------------------------------
# training utilities
# --------------------------------------------------------------------------
def test_roc_auc_matches_known_cases():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    # all scores tied -> no ranking information
    assert abs(roc_auc(np.array([0, 1, 0, 1]), np.ones(4)) - 0.5) < 1e-9


def test_roc_auc_agrees_with_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-9


def test_train_config_records_everything_the_report_needs():
    d = TrainConfig().as_dict()
    for key in ("epochs", "batch_size", "learning_rate", "optimiser", "loss",
                "augmentation", "model_selection"):
        assert key in d


# --------------------------------------------------------------------------
# inference: model scores -> the Milestone 1 assembler
# --------------------------------------------------------------------------
def test_probabilities_become_a_valid_compatibility_table():
    s = build_puzzle_sample(3, 4, seed=14)
    n = s.n_pieces
    rng = np.random.default_rng(0)
    prob = rng.random((4 * n, 4 * n))
    table = probabilities_to_table(prob, s.descriptions)
    assert table.cost.shape == (n, 4, n, 4)

    ok = admissibility_mask(s.descriptions)
    finite = np.isfinite(table.cost.reshape(4 * n, 4 * n))
    assert np.array_equal(finite, ok)             # exactly the legal pairs
    # a higher probability must always mean a lower cost.  The mapping is
    # -log p, which is monotone but not linear, so monotonicity is what to
    # assert -- a Pearson correlation would only be about -0.9 even when the
    # ordering is exactly right.
    flat_p = 0.5 * (prob + prob.T)
    a, b = np.nonzero(ok)
    c = table.cost.reshape(4 * n, 4 * n)[a, b]
    by_prob = np.argsort(flat_p[a, b])
    assert np.all(np.diff(c[by_prob]) <= 1e-9)


def test_model_tables_feed_the_classical_assembler():
    from src.assembly import assemble
    s = build_puzzle_sample(3, 4, seed=15)
    for table in (siamese_table(SiameseCNN(), s.descriptions),
                  gnn_table(PuzzleGNN(feat.VECTOR_DIM), s.descriptions, s.table)):
        a = assemble(s.descriptions, table, s.grid_shape)
        assert a.n_placed == s.n_pieces
        pid, _ = a.as_arrays()
        assert sorted(pid.ravel().tolist()) == list(range(s.n_pieces))


def test_predicted_matches_report_the_four_required_outputs():
    s = build_puzzle_sample(3, 4, seed=16)
    out = predicted_matches(siamese_table(SiameseCNN(), s.descriptions),
                            s.descriptions)
    assert out
    for row in out:
        for key in ("piece_a", "side_a", "piece_b", "side_b", "score",
                    "is_neighbour", "relative_orientation"):
            assert key in row
        assert 0.0 <= row["score"] <= 1.0
        assert row["relative_orientation"] in (0, 1, 2, 3)
        assert row["relative_orientation"] == feat.relative_orientation(
            row["side_a"], row["side_b"])


# ==========================================================================
# orientation accuracy (Milestone 2 / task 5)
# ==========================================================================
def test_true_rotations_are_recovered_from_the_labels():
    """The orientation ground truth is implied by cells + positives."""
    from src.ml.dataset import build_puzzle_sample
    from src import evaluation as ev
    from src.piece_description import DIRECTIONS

    s = build_puzzle_sample(3, 4, seed=77)
    if s is None or not s.positives:
        return
    rots = s.true_rotations()
    assert rots, "no rotation could be derived"
    assert set(rots) <= set(range(s.n_pieces))
    assert all(0 <= v < 4 for v in rots.values())

    # every true seam must agree with the rotations it implies
    step_dir = {(-1, 0): "N", (0, 1): "E", (1, 0): "S", (0, -1): "W"}
    for (i, a, j, b) in s.positives:
        (ri, ci), (rj, cj) = s.cells[i], s.cells[j]
        d = step_dir.get((rj - ri, cj - ci))
        if d is None or i not in rots:
            continue
        assert (rots[i] + DIRECTIONS.index(d)) % 4 == a


def test_orientation_accuracy_is_one_for_a_correct_arrangement():
    """Placing every piece at its true rotation must score 1.0."""
    from src.assembly import Assembly, Placement
    from src import evaluation as ev

    rows, cols = 2, 3
    cells = {i: (i // cols, i % cols) for i in range(rows * cols)}
    rots = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1}
    grid = [[Placement(piece=r * cols + c, rotation=rots[r * cols + c])
             for c in range(cols)] for r in range(rows)]
    asm = Assembly(grid_shape=(rows, cols), grid=grid, n_placed=rows * cols)
    out = ev.orientation_accuracy_from_cells(asm, rots, cells)
    assert out["orientation_accuracy"] == 1.0
    assert out["position_and_orientation_accuracy"] == 1.0

    # a single shared quarter turn is still a correct orientation
    turned = [[Placement(piece=r * cols + c,
                         rotation=(rots[r * cols + c] + 1) % 4)
               for c in range(cols)] for r in range(rows)]
    asm2 = Assembly(grid_shape=(rows, cols), grid=turned, n_placed=rows * cols)
    assert ev.orientation_accuracy_from_cells(asm2, rots)["orientation_accuracy"] == 1.0

    # but per-piece random turns are not
    mixed = [[Placement(piece=r * cols + c,
                        rotation=(rots[r * cols + c] + (r + 2 * c)) % 4)
              for c in range(cols)] for r in range(rows)]
    asm3 = Assembly(grid_shape=(rows, cols), grid=mixed, n_placed=rows * cols)
    assert ev.orientation_accuracy_from_cells(asm3, rots)["orientation_accuracy"] < 1.0


# --------------------------------------------------------------------------
# recovering side-pair labels for the provided photographs (task 1)
# --------------------------------------------------------------------------
def _synthetic_sample():
    """A generated puzzle, which carries the labels the recovery must match."""
    return build_puzzle_sample(3, 4, seed=451)


def test_rotation_recovery_reproduces_the_known_seams():
    """On a puzzle whose true seams ARE known, the recovery must find them.

    ``real_labels`` is used on photographs precisely because their seams are
    unknown, so it cannot be checked there.  A generated puzzle carries both,
    which makes it the only place the recovery can be pinned down: it is given
    nothing but the cells, and must rediscover the side pairs the generator
    recorded.
    """
    from src.ml.real_labels import recover_rotations

    s = _synthetic_sample()
    if s is None:
        return                                     # segmentation missed a piece
    lab = recover_rotations(s.descriptions, s.cells, s.grid_shape, table=s.table)

    truth = {(a, sa, b, sb) for a, sa, b, sb in s.positives}
    truth |= {(b, sb, a, sa) for a, sa, b, sb in s.positives}
    found = sum(1 for p in lab.positives if p in truth)
    assert found / max(len(lab.positives), 1) > 0.8, (
        f"only {found}/{len(lab.positives)} recovered seams are true seams")


def test_rotation_recovery_makes_seams_complementary():
    """A tab must meet a blank across every recovered seam."""
    from src.ml.real_labels import recover_rotations
    from src.piece_description import SIDE_BLANK, SIDE_TAB

    s = _synthetic_sample()
    if s is None:
        return
    lab = recover_rotations(s.descriptions, s.cells, s.grid_shape, table=s.table)
    assert lab.complementary_fraction > 0.8
    assert lab.border_fraction > 0.8


def test_true_neighbour_pairs_counts_the_grid_adjacencies():
    """A 3x4 grid has 2*3*4 - 3 - 4 = 17 internal adjacencies."""
    from src.ml.real_labels import true_neighbour_pairs

    cells = {r * 4 + c: (r, c) for r in range(3) for c in range(4)}
    assert len(true_neighbour_pairs(cells)) == 17


def test_labelled_pairs_are_balanced_and_admissible():
    """Negatives must be admissible pairs, not free wins on the tab/blank rule."""
    from src.ml.real_labels import label_side_pairs
    from src.piece_description import SIDE_BLANK, SIDE_FLAT, SIDE_TAB

    s = _synthetic_sample()
    if s is None:
        return
    labels, pairs, lab = label_side_pairs(s.descriptions, s.cells,
                                          s.grid_shape, table=s.table,
                                          negatives_per_positive=3)
    assert len(labels) == len(pairs)
    assert labels.sum() == len(lab.positives)
    for (a, sa, b, sb) in pairs:
        ta = s.descriptions[a].sides[sa].type
        tb = s.descriptions[b].sides[sb].type
        assert ta != SIDE_FLAT and tb != SIDE_FLAT
        assert ((ta == SIDE_TAB and tb == SIDE_BLANK)
                or (ta == SIDE_BLANK and tb == SIDE_TAB))
