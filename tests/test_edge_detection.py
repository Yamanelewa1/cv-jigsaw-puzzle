"""Tests for :mod:`src.edge_detection` (task 2)."""

import numpy as np

from src import edge_detection as ed


def _vertical_step(h=32, w=32, at=16):
    img = np.zeros((h, w))
    img[:, at:] = 1.0
    return img


def test_sobel_and_prewitt_kernels_are_zero_sum_and_transposed():
    for kx, ky in ((ed.SOBEL_X, ed.SOBEL_Y), (ed.PREWITT_X, ed.PREWITT_Y)):
        assert abs(kx.sum()) < 1e-12 and abs(ky.sum()) < 1e-12
        # the y kernel is the transpose of the x kernel
        assert np.allclose(kx, ky.T)


def test_gradient_of_a_vertical_edge_is_horizontal():
    img = _vertical_step()
    gx, gy, mag, theta = ed.sobel(img)
    y, x = 16, 15                       # just left of the step
    assert abs(gx[y, x]) > 1.0
    assert abs(gy[y, x]) < 1e-9
    # gradient direction points across the edge (0 or pi)
    assert min(abs(theta[y, x]), abs(abs(theta[y, x]) - np.pi)) < 1e-6


def test_gradient_of_a_horizontal_edge_is_vertical():
    img = _vertical_step().T
    gx, gy, mag, theta = ed.sobel(img)
    assert abs(gy[15, 16]) > 1.0
    assert abs(gx[15, 16]) < 1e-9


def test_prewitt_and_sobel_find_the_same_edge_location():
    img = _vertical_step()
    _, _, ms, _ = ed.sobel(img)
    _, _, mp, _ = ed.prewitt(img)
    assert int(np.argmax(ms[16])) == int(np.argmax(mp[16]))


def test_constant_image_has_no_gradient():
    img = np.full((16, 16), 0.7)
    _, _, mag, _ = ed.sobel(img, normalize=False)
    assert mag.max() < 1e-12


def test_non_maximum_suppression_thins_the_ridge():
    # a slightly blurred step, as any real edge is, gives a ridge several
    # pixels wide that NMS must reduce to its crest
    from src.enhancement import gaussian_blur
    img = gaussian_blur(_vertical_step(48, 48, 24), 2.0)
    _, _, mag, theta = ed.sobel(img)
    nms = ed.non_maximum_suppression(mag, theta)
    assert (nms > 0.1).sum() < (mag > 0.1).sum()
    # a ridge several pixels wide is reduced to its crest; a perfectly
    # symmetric step has two equal crest pixels, which NMS keeps by design
    assert int((mag[24] > 0.1).sum()) > 6
    assert 1 <= int((nms[24] > 0.1).sum()) <= 2


def test_double_threshold_partitions_the_pixels():
    nms = np.array([[0.0, 0.03, 0.09, 0.4]])
    strong, weak = ed.double_threshold(nms, 0.05, 0.15)
    assert np.array_equal(strong, [[False, False, False, True]])
    assert np.array_equal(weak, [[False, False, True, False]])
    assert not (strong & weak).any()


def test_hysteresis_keeps_connected_weak_pixels_only():
    strong = np.zeros((5, 7), dtype=bool)
    weak = np.zeros((5, 7), dtype=bool)
    strong[2, 0] = True
    weak[2, 1] = weak[2, 2] = True      # chained to the strong pixel
    weak[0, 5] = True                   # isolated
    out = ed.hysteresis(strong, weak)
    assert out[2, 0] and out[2, 1] and out[2, 2]
    assert not out[0, 5]


def test_canny_outlines_a_rectangle_without_filling_it():
    img = np.zeros((60, 60))
    img[15:45, 15:45] = 1.0
    edges = ed.canny(img, sigma=1.0, low=0.05, high=0.2)
    assert edges.dtype == np.bool_
    # edges hug the border ...
    assert edges[13:18, 25:35].any() and edges[42:47, 25:35].any()
    # ... and the flat interior stays empty
    assert not edges[25:35, 25:35].any()


def test_canny_sigma_controls_the_amount_of_detail():
    # fine texture on the left, one strong step on the right
    yy, xx = np.mgrid[0:64, 0:64]
    img = np.zeros((64, 64))
    img[:, :32] = 0.25 + 0.5 * (((yy + xx) // 3) % 2)[:, :32]
    img[:, 32:] = 0.85
    fine = ed.canny(img, sigma=0.8, low=0.05, high=0.15)
    coarse = ed.canny(img, sigma=3.0, low=0.05, high=0.15)
    assert fine.sum() > coarse.sum()
    # the strong step survives at both scales
    assert fine[:, 28:36].any() and coarse[:, 28:36].any()


def test_canny_returns_all_stages_on_request():
    img = _vertical_step()
    edges, stages = ed.canny(img, return_stages=True)
    for key in ("smoothed", "magnitude", "orientation", "nms", "strong",
                "weak", "low", "high"):
        assert key in stages
    assert stages["nms"].shape == img.shape
    assert edges.shape == img.shape


def test_canny_percentile_mode_is_exposure_independent():
    img = _vertical_step()
    a = ed.canny(img, 1.0, 80, 95, use_percentiles=True)
    b = ed.canny(0.5 * img, 1.0, 80, 95, use_percentiles=True)
    assert (a == b).mean() > 0.99
