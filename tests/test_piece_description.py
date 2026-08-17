"""Tests for :mod:`src.piece_description` (task 4)."""

import numpy as np

from src import contour_extraction as ce
from src import evaluation as ev
from src import piece_description as pd
from src import segmentation as seg


def _pieces(rows=3, cols=4, seed=7):
    src = ev.synthetic_source_image(110 * rows, 110 * cols, seed=100 + seed)
    scrambled, gt = ev.generate_puzzle(src, rows=rows, cols=cols,
                                       rotate=True, seed=seed)
    mask = seg.foreground_mask(scrambled, "background", open_radius=1,
                               close_radius=1)
    labels, _ = seg.connected_components(mask)
    labels, stats = seg.filter_components(labels, min_area_ratio=0.3)
    return ce.extract_pieces(scrambled, labels, stats=stats), gt


# --------------------------------------------------------------------------
# contour signal processing
# --------------------------------------------------------------------------
def test_smooth_contour_keeps_length_and_reduces_jitter():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    clean = np.stack([30 * np.sin(t), 30 * np.cos(t)], axis=1)
    noisy = clean + np.random.default_rng(0).normal(0, 0.7, clean.shape)
    out = pd.smooth_contour(noisy, 7)
    assert out.shape == noisy.shape
    assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean()


def test_turning_angles_of_a_square_peak_at_the_corners():
    side = np.linspace(0, 40, 41)
    top = np.stack([np.zeros_like(side), side], 1)
    right = np.stack([side, np.full_like(side, 40)], 1)
    bottom = np.stack([np.full_like(side, 40), side[::-1]], 1)
    left = np.stack([side[::-1], np.zeros_like(side)], 1)
    square = np.vstack([top, right, bottom, left])
    angle, _ = pd.turning_angles(square, 5)
    assert np.rad2deg(angle.max()) > 80.0


def test_dominant_orientation_makes_the_edges_axis_aligned():
    """The angle it returns must be the one that straightens the rectangle."""
    t = np.linspace(0, 1, 400)
    box = np.vstack([np.stack([np.zeros_like(t), 60 * t], 1),
                     np.stack([40 * t, np.full_like(t, 60)], 1),
                     np.stack([np.full_like(t, 40), 60 * (1 - t)], 1),
                     np.stack([40 * (1 - t), np.zeros_like(t)], 1)])
    for deg in (0.0, 17.0, 40.0, 71.0, 123.0):
        a = np.deg2rad(deg)
        rot = np.stack([box[:, 0] * np.cos(a) - box[:, 1] * np.sin(a),
                        box[:, 0] * np.sin(a) + box[:, 1] * np.cos(a)], 1)
        theta = pd.dominant_orientation(rot)
        assert 0.0 <= theta < np.pi / 2 + 1e-9
        # undo it with the library's own convention and check the extent
        c, s = np.cos(-theta), np.sin(-theta)
        ry = rot[:, 0] * c + rot[:, 1] * s
        rx = -rot[:, 0] * s + rot[:, 1] * c
        h, w = ry.max() - ry.min(), rx.max() - rx.min()
        assert abs(min(h, w) - 40) < 2.0 and abs(max(h, w) - 60) < 2.0, deg


# --------------------------------------------------------------------------
# corners and sides
# --------------------------------------------------------------------------
def _corner_quality(corners):
    sides = [np.hypot(*(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    angles = []
    for i in range(4):
        a = corners[(i - 1) % 4] - corners[i]
        b = corners[(i + 1) % 4] - corners[i]
        cosang = np.dot(a, b) / (np.hypot(*a) * np.hypot(*b) + 1e-12)
        angles.append(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
    return sides, angles


def test_find_corners_always_returns_a_usable_quadrilateral():
    """Whatever happens, four distinct points enclosing a real area."""
    for args in ((3, 4, 7), (2, 3, 8), (3, 4, 10)):
        pieces, _ = _pieces(*args)
        for p in pieces:
            corners = pd.find_corners(p.contour, fallback_mask=p.mask)
            assert corners.shape == (4, 2)
            sides, angles = _corner_quality(corners)
            assert min(sides) > 0.2 * max(sides)
            assert 30 < min(angles) and max(angles) < 150
            from src.contour_extraction import polygon_area
            assert polygon_area(corners) > 0.25 * p.mask.sum()


def test_find_corners_is_accurate_for_almost_every_piece():
    """The body-edge model must give a near-perfect rectangle nearly always.

    A stray piece may fall through to a fallback strategy and come out with a
    skewed quadrilateral; that costs one descriptor, not the reconstruction
    (``test_rotated_puzzle_is_reconstructed_exactly`` covers the outcome), so
    the requirement here is on the rate rather than on every single piece.
    """
    good = total = 0
    for args in ((3, 4, 7), (2, 3, 8), (3, 4, 10), (4, 5, 13)):
        pieces, _ = _pieces(*args)
        for p in pieces:
            sides, angles = _corner_quality(
                pd.find_corners(p.contour, fallback_mask=p.mask))
            total += 1
            if min(sides) > 0.6 * max(sides) and min(angles) > 60 \
                    and max(angles) < 120:
                good += 1
    assert good / total >= 0.9, f"only {good}/{total} pieces well cornered"


def test_split_sides_covers_the_whole_boundary():
    pieces, _ = _pieces(2, 3, 8)
    p = pieces[0]
    corners = pd.find_corners(p.contour, fallback_mask=p.mask)
    sides = pd.split_sides(p.contour.astype(float), corners, 64)
    assert len(sides) == 4
    for i in range(4):
        assert sides[i].shape == (64, 2)
        # consecutive sides share an endpoint
        assert np.hypot(*(sides[i][-1] - sides[(i + 1) % 4][0])) < 3.0


def test_classify_side_flat_tab_and_blank():
    x = np.linspace(0, 100, 96)
    centroid = (30.0, 50.0)             # below the side (larger y)

    flat = np.stack([np.zeros_like(x), x], 1)
    assert pd.classify_side(flat, centroid)[0] == pd.SIDE_FLAT

    bump = -25 * np.exp(-((x - 50) ** 2) / (2 * 12 ** 2))   # outward = -y
    tab = np.stack([bump, x], 1)
    assert pd.classify_side(tab, centroid)[0] == pd.SIDE_TAB

    blank = np.stack([-bump, x], 1)
    assert pd.classify_side(blank, centroid)[0] == pd.SIDE_BLANK


def test_profile_is_scale_invariant():
    x = np.linspace(0, 100, 96)
    centroid = (30.0, 50.0)
    bump = -25 * np.exp(-((x - 50) ** 2) / (2 * 12 ** 2))
    small = np.stack([bump, x], 1)
    big = small * 3.0
    _, p1, _, _ = pd.classify_side(small, centroid)
    _, p2, _, _ = pd.classify_side(big, (90.0, 150.0))
    assert np.max(np.abs(p1 - p2)) < 1e-6


def test_colour_strip_never_samples_the_background():
    pieces, _ = _pieces(2, 3, 9)
    for p in pieces:
        d = pd.describe_piece(p)
        for s in d.sides:
            # a strip taken from a colourful piece must not be black
            assert s.colors.shape[0] == len(s.points)
            assert s.colors.reshape(-1, 3).sum(axis=1).min() > 0.05


def test_side_type_counts_match_the_grid():
    """A 3x4 puzzle has 4 corner, 6 edge and 2 interior pieces."""
    pieces, gt = _pieces(3, 4, 10)
    descs = pd.describe_pieces(pieces)
    counts = {0: 0, 1: 0, 2: 0}
    for d in descs:
        counts[min(d.n_flats, 2)] = counts.get(min(d.n_flats, 2), 0) + 1
    assert counts[2] == 4
    assert counts[1] == 6
    assert counts[0] == 2
    assert sum(1 for d in descs if d.is_corner_piece) == 4


def test_every_piece_has_exactly_four_sides_in_clockwise_order():
    pieces, _ = _pieces(2, 3, 11)
    for d in pd.describe_pieces(pieces):
        assert len(d.sides) == 4
        assert [s.index for s in d.sides] == [0, 1, 2, 3]
        # side i ends where side i+1 begins
        for i in range(4):
            end = np.asarray(d.sides[i].corners[1])
            start = np.asarray(d.sides[(i + 1) % 4].corners[0])
            assert np.hypot(*(end - start)) < 3.0


def test_side_facing_follows_the_rotation():
    pieces, _ = _pieces(2, 3, 12)
    d = pd.describe_pieces(pieces)[0]
    for rot in range(4):
        assert d.side_facing(rot, "N") is d.sides[rot % 4]
        assert d.side_facing(rot, "E") is d.sides[(rot + 1) % 4]
        assert d.side_facing(rot, "S") is d.sides[(rot + 2) % 4]
        assert d.side_facing(rot, "W") is d.sides[(rot + 3) % 4]
