"""Tests for :mod:`src.segmentation` and :mod:`src.contour_extraction` (task 3)."""

import numpy as np

from src import contour_extraction as ce
from src import evaluation as ev
from src import segmentation as seg


# --------------------------------------------------------------------------
# morphology
# --------------------------------------------------------------------------
def test_disk_is_symmetric_and_round():
    d = seg.disk(2)
    assert d.shape == (5, 5)
    assert np.array_equal(d, d[::-1]) and np.array_equal(d, d[:, ::-1])
    assert d[2, 2] and not d[0, 0]


def test_erosion_shrinks_and_dilation_grows_a_square():
    m = np.zeros((21, 21), dtype=bool)
    m[6:15, 6:15] = True
    e = seg.erode(m, np.ones((3, 3), bool))
    d = seg.dilate(m, np.ones((3, 3), bool))
    assert e.sum() == 7 * 7
    assert d.sum() == 11 * 11


def test_opening_removes_speckle_closing_seals_holes():
    m = np.zeros((25, 25), dtype=bool)
    m[8:18, 8:18] = True
    m[2, 2] = True                      # isolated speckle
    m[12, 12] = False                   # pin hole
    assert not seg.opening(m, seg.disk(1))[2, 2]
    assert seg.closing(m, seg.disk(1))[12, 12]


def test_majority_smooth_is_self_dual():
    """A shape and its complement must be smoothed the same way.

    Exactly self-dual wherever the neighbourhood count is odd; right at the
    image border the count can be even and a tie is broken one way, so the
    interior is what is checked.
    """
    rng = np.random.default_rng(0)
    m = rng.random((40, 40)) > 0.5
    a = seg.majority_smooth(m, 2)
    b = ~seg.majority_smooth(~m, 2)
    assert np.array_equal(a[3:-3, 3:-3], b[3:-3, 3:-3])


def test_majority_smooth_rounds_a_noisy_boundary():
    m = np.zeros((40, 40), dtype=bool)
    m[10:30, 10:30] = True
    m[9, 15] = m[9, 21] = True          # single-pixel spikes
    m[20, 10] = False                   # single-pixel dent
    out = seg.majority_smooth(m, 2)
    assert not out[9, 15] and not out[9, 21]
    assert out[20, 10]


def test_fill_holes_fills_a_ring_but_not_the_outside():
    m = np.zeros((30, 30), dtype=bool)
    m[5:25, 5:25] = True
    m[10:20, 10:20] = False
    filled = seg.fill_holes(m)
    assert filled[15, 15]
    assert not filled[1, 1]


def test_reconstruct_keeps_only_seeded_components():
    allowed = np.zeros((10, 20), dtype=bool)
    allowed[2:5, 2:5] = True
    allowed[2:5, 12:15] = True
    seed = np.zeros_like(allowed)
    seed[3, 3] = True
    out = seg.reconstruct(seed, allowed)
    assert out[3, 3] and not out[3, 13]


# --------------------------------------------------------------------------
# connected components
# --------------------------------------------------------------------------
def test_connected_components_counts_and_labels_blobs():
    m = np.zeros((10, 20), dtype=bool)
    m[2:5, 2:6] = True
    m[6:9, 12:18] = True
    labels, n = seg.connected_components(m)
    assert n == 2
    assert labels[3, 3] != labels[7, 14]
    assert labels[0, 0] == 0
    assert len(np.unique(labels[m])) == 2


def test_connectivity_changes_the_diagonal_case():
    m = np.zeros((5, 5), dtype=bool)
    m[1, 1] = m[2, 2] = True
    assert seg.connected_components(m, 4)[1] == 2
    assert seg.connected_components(m, 8)[1] == 1


def test_empty_mask_has_no_components():
    labels, n = seg.connected_components(np.zeros((6, 6), bool))
    assert n == 0 and labels.max() == 0


def test_distance_transform_is_exact_for_a_square():
    m = np.zeros((21, 21), dtype=bool)
    m[5:16, 5:16] = True
    d = seg.distance_transform(m)
    assert not d[m].min() < 1.0
    # the centre of an 11x11 square is 6 px from the outside
    assert abs(d[10, 10] - 6.0) < 1e-9
    # a pixel on the edge is 1 px away
    assert abs(d[5, 10] - 1.0) < 1e-9
    assert d[~m].max() == 0.0


def test_distance_transform_matches_brute_force():
    rng = np.random.default_rng(0)
    m = np.zeros((24, 24), dtype=bool)
    m[4:20, 6:18] = True
    m[10:14, 10:12] = False
    d = seg.distance_transform(m)
    ys, xs = np.nonzero(~m)
    for (y, x) in [(8, 9), (15, 15), (5, 7)]:
        brute = np.hypot(ys - y, xs - x).min()
        assert abs(d[y, x] - brute) < 1e-9


def test_split_touching_separates_two_joined_blobs():
    """Two discs joined by a narrow neck must come apart into two labels."""
    m = np.zeros((60, 110), dtype=bool)
    yy, xx = np.mgrid[0:60, 0:110]
    m |= (yy - 30) ** 2 + (xx - 32) ** 2 <= 22 ** 2
    m |= (yy - 30) ** 2 + (xx - 78) ** 2 <= 22 ** 2
    m[27:33, 30:80] = True                     # the isthmus
    labels, n = seg.connected_components(m)
    assert n == 1                              # they really are one component

    target = float(np.pi * 22 ** 2)
    split, n_split = seg.split_touching(labels, target_area=target)
    assert n_split == 1
    assert int(split.max()) == 2
    a, b = [int((split == k).sum()) for k in (1, 2)]
    assert min(a, b) > 0.3 * max(a, b)         # a fair split, not a sliver
    assert split[30, 32] != split[30, 78]      # one disc each


def test_split_touching_leaves_separate_pieces_alone():
    m = np.zeros((40, 80), dtype=bool)
    m[8:32, 8:32] = True
    m[8:32, 48:72] = True
    labels, n = seg.connected_components(m)
    split, n_split = seg.split_touching(labels)
    assert n_split == 0
    assert int(split.max()) == n == 2


def test_filter_components_upper_area_bound():
    m = np.zeros((60, 120), dtype=bool)
    m[5:25, 5:25] = True                       # normal
    m[5:25, 35:55] = True                      # normal
    m[5:55, 65:115] = True                     # much too big
    labels, _ = seg.connected_components(m)
    _, stats = seg.filter_components(labels, min_area_ratio=0.3,
                                     max_area_ratio=1.7)
    assert len(stats) == 2


def test_reference_area_is_the_plain_median_on_uniform_input():
    areas = np.array([980.0, 1000.0, 1020.0, 1000.0])
    assert abs(seg.reference_area(areas) - 1000.0) < 1e-9


def test_reference_area_ignores_a_swarm_of_slivers():
    """A watershed can shed more slivers than there are pieces.

    The slivers then outnumber the pieces, so a plain median follows them
    instead of the pieces.  ``reference_area`` must still report the piece
    size -- this is the failure that left 3 of 35 pieces on one dataset
    photograph.
    """
    areas = np.array([7700.0] * 35 + [150.0] * 60)
    assert abs(np.median(areas) - 150.0) < 1e-9          # the plain median fails
    assert abs(seg.reference_area(areas) - 7700.0) < 1e-9


def test_reference_area_is_not_dragged_up_by_oversized_blobs():
    """Unresolved merged pieces are huge but few; they must not win either."""
    areas = np.array([1000.0] * 20 + [9000.0] * 3)
    assert abs(seg.reference_area(areas) - 1000.0) < 1e-9


def test_filter_components_survives_sliver_debris():
    """End to end: two real squares plus sliver debris keeps both squares."""
    m = np.zeros((80, 200), dtype=bool)
    m[5:25, 5:25] = True                       # a piece
    m[5:25, 35:55] = True                      # a piece
    for k in range(12):                        # debris, 2x2 each
        m[70:72, 4 * k:4 * k + 2] = True
    labels, _ = seg.connected_components(m)
    _, stats = seg.filter_components(labels, min_area_ratio=0.45,
                                     max_area_ratio=1.7)
    assert len(stats) == 2


def test_component_stats_area_bbox_and_centroid():
    m = np.zeros((12, 12), dtype=bool)
    m[3:7, 4:9] = True
    labels, _ = seg.connected_components(m)
    st = seg.component_stats(labels)[0]
    assert st.area == 4 * 5
    assert st.bbox == (3, 4, 7, 9)
    assert abs(st.centroid[0] - 4.5) < 1e-9
    assert abs(st.centroid[1] - 6.0) < 1e-9


def test_filter_components_drops_small_clutter():
    m = np.zeros((40, 40), dtype=bool)
    m[2:12, 2:12] = True                # big
    m[20:30, 20:30] = True              # big
    m[35, 35] = True                    # tiny
    labels, _ = seg.connected_components(m)
    kept, stats = seg.filter_components(labels, min_area_ratio=0.3)
    assert len(stats) == 2
    assert kept.max() == 2


def test_remove_border_components():
    m = np.zeros((20, 20), dtype=bool)
    m[0:4, 0:4] = True                  # touches the frame
    m[8:14, 8:14] = True
    out = seg.remove_border_components(m)
    assert not out[1, 1] and out[10, 10]


# --------------------------------------------------------------------------
# foreground mask
# --------------------------------------------------------------------------
def test_foreground_mask_separates_pieces_from_the_background():
    src = ev.synthetic_source_image(200, 260, seed=1)
    scrambled, gt = ev.generate_puzzle(src, rows=2, cols=2, rotate=True, seed=2)
    mask = seg.foreground_mask(scrambled, "background", open_radius=1,
                               close_radius=1)
    labels, _ = seg.connected_components(mask)
    _, stats = seg.filter_components(labels, min_area_ratio=0.3)
    assert len(stats) == 4
    assert 0.05 < mask.mean() < 0.7


def test_background_colour_is_estimated_from_the_frame():
    img = np.zeros((40, 40, 3), dtype=np.float64)
    img[15:25, 15:25] = 1.0
    assert np.allclose(seg.background_color(img), 0.0)
    d = seg.background_distance(img)
    assert d[20, 20] > 0.9 and d[0, 0] < 0.1


# --------------------------------------------------------------------------
# contour extraction
# --------------------------------------------------------------------------
def test_trace_boundary_walks_a_square_once():
    m = np.zeros((12, 12), dtype=bool)
    m[3:9, 3:9] = True
    c = ce.trace_boundary(m)
    assert len(c) == 4 * 6 - 4                    # perimeter of a 6x6 square
    assert m[c[:, 0], c[:, 1]].all()
    assert len(set(map(tuple, c))) == len(c)      # no repeats


def test_trace_boundary_handles_a_concave_shape():
    m = np.zeros((14, 14), dtype=bool)
    m[3:11, 3:11] = True
    m[3:7, 6:9] = False                           # a notch open to the top
    c = ce.trace_boundary(m)
    assert len(c) > 20
    assert m[c[:, 0], c[:, 1]].all()


def test_polygon_area_and_perimeter_of_a_square():
    pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
    assert abs(ce.polygon_area(pts) - 100.0) < 1e-9
    assert abs(ce.polygon_perimeter(pts) - 40.0) < 1e-9


def test_resample_contour_gives_even_spacing():
    pts = np.array([[0, 0], [0, 9]], dtype=float)
    out = ce.resample_contour(pts, 10)
    assert out.shape == (10, 2)
    assert np.allclose(np.diff(out[:, 1]), 1.0)


def test_convex_hull_of_a_square_with_interior_points():
    pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [5, 5], [3, 7]],
                   dtype=float)
    hull = ce.convex_hull(pts)
    assert len(hull) == 4
    assert abs(ce.polygon_area(hull) - 100.0) < 1e-9


def test_min_area_rect_recovers_a_rotated_rectangle():
    rng = np.random.default_rng(0)
    a = np.deg2rad(27.0)
    local = np.stack([rng.uniform(-20, 20, 400), rng.uniform(-8, 8, 400)], 1)
    rot = np.stack([local[:, 0] * np.cos(a) - local[:, 1] * np.sin(a),
                    local[:, 0] * np.sin(a) + local[:, 1] * np.cos(a)], 1)
    _, size, _ = ce.min_area_rect(rot)
    long_side, short_side = max(size), min(size)
    assert abs(long_side - 40) < 3 and abs(short_side - 16) < 3


def test_rotate_image_by_360_degrees_is_the_identity():
    rng = np.random.default_rng(1)
    img = rng.random((21, 21))
    out = ce.rotate_image(img, 2 * np.pi)
    assert np.max(np.abs(out[2:-2, 2:-2] - img[2:-2, 2:-2])) < 1e-9


def test_rotate_mask_preserves_area_approximately():
    m = np.zeros((41, 41), dtype=bool)
    m[12:29, 12:29] = True
    r = ce.rotate_mask(m, np.deg2rad(30.0))
    assert abs(int(r.sum()) - int(m.sum())) < 0.05 * m.sum()


def test_extract_pieces_crops_masks_and_contours():
    src = ev.synthetic_source_image(200, 260, seed=3)
    scrambled, _ = ev.generate_puzzle(src, rows=2, cols=2, rotate=True, seed=4)
    mask = seg.foreground_mask(scrambled, "background", open_radius=1,
                               close_radius=1)
    labels, _ = seg.connected_components(mask)
    labels, stats = seg.filter_components(labels, min_area_ratio=0.3)
    pieces = ce.extract_pieces(scrambled, labels, stats=stats)
    assert len(pieces) == 4
    for p in pieces:
        assert p.mask.shape == p.image.shape[:2]
        assert p.mask.any() and len(p.contour) > 20
        assert p.image[~p.mask].max() == 0.0        # background zeroed
        y0, x0, y1, x1 = p.bbox
        assert 0 <= y0 < y1 and 0 <= x0 < x1


def test_normalize_piece_makes_the_body_axis_aligned():
    src = ev.synthetic_source_image(200, 260, seed=5)
    scrambled, _ = ev.generate_puzzle(src, rows=2, cols=2, rotate=True, seed=6)
    mask = seg.foreground_mask(scrambled, "background", open_radius=1,
                               close_radius=1)
    labels, _ = seg.connected_components(mask)
    labels, stats = seg.filter_components(labels, min_area_ratio=0.3)
    for p in ce.extract_pieces(scrambled, labels, stats=stats):
        n = ce.normalize_piece(p)
        residual = abs(np.rad2deg(n.angle)) % 90.0
        assert min(residual, 90.0 - residual) < 5.0
