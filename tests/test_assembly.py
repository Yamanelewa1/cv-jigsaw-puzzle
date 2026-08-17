"""Tests for :mod:`src.assembly` and the end-to-end routine (task 6)."""

import numpy as np

from src import PuzzleSolver, solve_puzzle
from src import assembly as asm
from src import edge_matching as em
from src import evaluation as ev
from src import piece_description as pd


def _solved(rows=3, cols=4, seed=31, rotate=True):
    src = ev.synthetic_source_image(110 * rows, 110 * cols, seed=300 + seed)
    scrambled, gt = ev.generate_puzzle(src, rows=rows, cols=cols,
                                       rotate=rotate, seed=seed)
    res = PuzzleSolver(open_radius=1, close_radius=1).solve(scrambled,
                                                            (rows, cols))
    return src, scrambled, gt, res


# --------------------------------------------------------------------------
# grid inference
# --------------------------------------------------------------------------
def test_infer_grid_shape_from_border_counts():
    _, _, gt, res = _solved(3, 4, 32)
    assert asm.infer_grid_shape(res.descriptions) == (3, 4)


def test_infer_grid_shape_on_an_empty_list():
    assert asm.infer_grid_shape([]) == (0, 0)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def test_similarity_from_pairs_maps_both_points():
    src = np.array([[3.0, 4.0], [3.0, 24.0]])
    dst = np.array([[10.0, 10.0], [30.0, 10.0]])       # rotated 90 degrees
    m = asm.similarity_from_pairs(src, dst)
    for s, d in zip(src, dst):
        got = m @ np.array([s[0], s[1], 1.0])
        assert np.allclose(got, d, atol=1e-9)


def test_warp_similarity_moves_a_square():
    img = np.zeros((40, 40, 3))
    mask = np.zeros((40, 40), dtype=bool)
    img[10:20, 10:20] = 1.0
    mask[10:20, 10:20] = True
    m = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]])   # pure translation
    wi, wm = asm.warp_similarity(img, mask, m, (40, 40))
    assert wm[15 + 5, 15 + 7]
    assert not wm[15, 15 - 7]


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def test_assembly_fills_the_grid_and_uses_each_piece_once():
    _, _, gt, res = _solved(3, 4, 33)
    a = res.assembly
    assert a.grid_shape == (3, 4)
    assert a.complete and a.n_placed == 12
    pid, rot = a.as_arrays()
    assert sorted(pid.ravel().tolist()) == list(range(12))
    assert set(rot.ravel().tolist()) <= {0, 1, 2, 3}


def test_border_pieces_show_their_flats_outwards():
    """Flats must face out of the grid, and nothing else may.

    A placement made through a relaxed (dead-end) stage is allowed to break
    the rule -- that is what the relaxation is for -- so those are excluded
    and counted separately.
    """
    _, _, _, res = _solved(3, 4, 34)
    a, descs = res.assembly, res.descriptions
    rows, cols = a.grid_shape
    checked = 0
    for r in range(rows):
        for c in range(cols):
            pl = a.grid[r][c]
            if pl.forced:
                continue
            for d in ("N", "E", "S", "W"):
                dr, dc = {"N": (-1, 0), "E": (0, 1),
                          "S": (1, 0), "W": (0, -1)}[d]
                outside = not (0 <= r + dr < rows and 0 <= c + dc < cols)
                is_flat = descs[pl.piece].side_facing(pl.rotation, d).is_flat
                assert is_flat == outside, (r, c, d)
                checked += 1
    assert checked >= 4 * (rows * cols - 1)


def test_rotated_puzzle_is_reconstructed_exactly():
    _, _, gt, res = _solved(3, 4, 35, rotate=True)
    assoc = ev.associate_with_ground_truth(res.pieces, gt)
    assert ev.neighbour_accuracy(res.assembly, gt, assoc)["neighbour_accuracy"] == 1.0
    assert ev.direct_accuracy(res.assembly, res.descriptions, gt,
                              assoc)["position_accuracy"] == 1.0
    assert ev.rotation_accuracy(res.assembly, res.descriptions, gt,
                                assoc)["rotation_accuracy"] == 1.0


def test_unrotated_puzzle_is_reconstructed_exactly():
    _, _, gt, res = _solved(2, 3, 36, rotate=False)
    assoc = ev.associate_with_ground_truth(res.pieces, gt)
    assert ev.neighbour_accuracy(res.assembly, gt, assoc)["neighbour_accuracy"] == 1.0


def test_assembly_returns_a_full_arrangement_even_with_useless_costs():
    """Dead-end handling: nothing matches, but every piece is still placed."""
    _, _, _, res = _solved(2, 3, 37)
    table = res.table
    broken = em.CompatibilityTable(
        cost=np.full_like(table.cost, np.inf),
        shape=table.shape, colour=table.colour, length=table.length,
        weights=table.weights)
    a = asm.assemble(res.descriptions, broken, (2, 3))
    assert a.n_placed == 6
    assert a.n_forced > 0
    pid, _ = a.as_arrays()
    assert sorted(pid.ravel().tolist()) == list(range(6))


def test_restarts_never_return_a_worse_arrangement():
    _, _, gt, res = _solved(3, 4, 38)
    one = asm.assemble(res.descriptions, res.table, (3, 4), max_restarts=1)
    many = asm.assemble(res.descriptions, res.table, (3, 4), max_restarts=8)
    assert asm._arrangement_key(many) <= asm._arrangement_key(one)


# --------------------------------------------------------------------------
# rendering + end to end
# --------------------------------------------------------------------------
def test_render_assembly_produces_a_populated_canvas():
    src, _, _, res = _solved(3, 4, 39)
    canvas, (pad_y, pad_x, ch, cw) = asm.render_assembly(
        res.descriptions, res.assembly, return_geometry=True)
    rows, cols = res.assembly.grid_shape
    assert canvas.shape[0] >= rows * ch and canvas.shape[1] >= cols * cw
    body = canvas[pad_y:pad_y + rows * ch, pad_x:pad_x + cols * cw]
    assert (body.sum(axis=2) > 10).mean() > 0.97      # almost no holes


def test_end_to_end_reconstruction_resembles_the_original():
    src, scrambled, gt, res = _solved(3, 4, 40)
    m = ev.image_metrics(res.reconstructed_body(), src)
    assert m["psnr_db"] > 18.0
    assert m["ssim"] > 0.6
    # ... and clearly better than against an unrelated picture
    other = ev.synthetic_source_image(*res.reconstructed_body().shape[:2], seed=999)
    assert m["mse"] < ev.image_metrics(res.reconstructed_body(), other)["mse"]


def test_solve_puzzle_wrapper_reports_quality():
    src = ev.synthetic_source_image(220, 330, seed=41)
    scrambled, gt = ev.generate_puzzle(src, rows=2, cols=3, rotate=True, seed=41)
    res = solve_puzzle(scrambled, (2, 3), open_radius=1, close_radius=1)
    assert res.n_pieces == 6
    assert 0.0 <= res.quality["quality"] <= 1.0
    assert res.quality["n_illegal_seams"] == 0
    assert res.reconstruction is not None
    assert "total" in res.timings
    assert res.summary()


def test_seam_quality_prefers_the_correct_arrangement():
    _, _, gt, res = _solved(3, 4, 42)
    good = ev.seam_quality(res.assembly, res.table)
    # shuffle the grid and re-score
    import copy
    bad_asm = copy.deepcopy(res.assembly)
    flat = [bad_asm.grid[r][c] for r in range(3) for c in range(4)]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(flat))
    for k, idx in enumerate(order):
        bad_asm.grid[k // 4][k % 4] = flat[idx]
    bad = ev.seam_quality(bad_asm, res.table)
    assert good["quality"] > bad["quality"]
