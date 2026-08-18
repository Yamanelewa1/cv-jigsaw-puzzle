"""Tests for :mod:`src.enhancement` (primitives + task 1)."""

import numpy as np

from src import enhancement as enh


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def test_to_float_and_back_roundtrip():
    a = np.array([[0, 128, 255]], dtype=np.uint8)
    f = enh.to_float(a)
    assert f.max() <= 1.0 and f.min() >= 0.0
    assert np.array_equal(enh.to_uint8(f), a)


def test_to_gray_uses_luma_weights():
    rgb = np.zeros((1, 3, 3), dtype=np.float64)
    rgb[0, 0] = (1, 0, 0)
    rgb[0, 1] = (0, 1, 0)
    rgb[0, 2] = (0, 0, 1)
    g = enh.to_gray(rgb)
    assert np.allclose(g[0], [0.299, 0.587, 0.114])


def test_convolution_matches_manual_sum():
    rng = np.random.default_rng(0)
    img = rng.random((9, 11))
    k = rng.random((3, 3))
    out = enh.correlate2d(img, k, mode="constant")
    # interior pixel computed by hand
    y, x = 4, 5
    expected = float((img[y - 1:y + 2, x - 1:x + 2] * k).sum())
    assert abs(out[y, x] - expected) < 1e-12


def test_convolution_is_flipped_correlation():
    rng = np.random.default_rng(1)
    img = rng.random((7, 7))
    k = rng.random((3, 3))
    assert np.allclose(enh.convolve2d(img, k),
                       enh.correlate2d(img, k[::-1, ::-1]))


def test_integral_image_gives_window_sums():
    rng = np.random.default_rng(2)
    img = rng.random((8, 8))
    s = enh.integral_image(img)
    total = s[5, 6] - s[1, 6] - s[5, 2] + s[1, 2]
    assert abs(total - img[1:5, 2:6].sum()) < 1e-12


def test_resize_bilinear_preserves_corners_and_shape():
    img = np.arange(16, dtype=np.float64).reshape(4, 4) / 15.0
    out = enh.resize_bilinear(img, (7, 9))
    assert out.shape == (7, 9)
    assert abs(out[0, 0] - img[0, 0]) < 1e-9
    assert abs(out[-1, -1] - img[-1, -1]) < 1e-9


# --------------------------------------------------------------------------
# noise reduction
# --------------------------------------------------------------------------
def test_mean_filter_equals_the_plain_neighbourhood_average():
    rng = np.random.default_rng(3)
    img = rng.random((24, 30))
    out = enh.mean_filter(img, 3)
    for y in range(1, 23):
        for x in range(1, 29):
            assert abs(out[y, x] - img[y - 1:y + 2, x - 1:x + 2].mean()) < 1e-12


def test_mean_filter_preserves_a_constant_image():
    img = np.full((16, 16), 0.4)
    assert np.allclose(enh.mean_filter(img, 5), 0.4)


def test_mean_filter_reduces_gaussian_noise():
    rng = np.random.default_rng(4)
    clean = np.full((64, 64), 0.5)
    noisy = np.clip(clean + rng.normal(0, 0.1, clean.shape), 0, 1)
    out = enh.mean_filter(noisy, 5)
    assert out.std() < 0.4 * noisy.std()


def test_mean_filter_edge_response_has_corners_the_gaussian_does_not():
    """A box filter turns a step into a hard-cornered ramp.

    The flat kernel's step response is piecewise linear, so its slope jumps
    discontinuously at both ends of the transition; the Gaussian's is smooth
    everywhere.  Comparing the largest second difference across the edge
    measures exactly that, and it is the reason the pipeline smooths with
    `gaussian_blur` rather than with the cheaper box filter.
    """
    step = np.zeros((8, 40))
    step[:, 20:] = 1.0
    box = enh.mean_filter(step, 9)[4]
    gau = enh.gaussian_blur(step, sigma=2.0)[4]
    # matched so that both transitions span a comparable width
    assert np.abs(np.diff(box, 2)).max() > 1.8 * np.abs(np.diff(gau, 2)).max()


def test_gaussian_kernel_normalised_and_symmetric():
    k = enh.gaussian_kernel_1d(sigma=1.7)
    assert abs(k.sum() - 1.0) < 1e-12
    assert np.allclose(k, k[::-1])
    assert len(k) % 2 == 1
    # size is derived from sigma when it is not given
    assert len(k) == 2 * int(np.ceil(3 * 1.7)) + 1
    k2 = enh.gaussian_kernel(sigma=1.7)
    assert abs(k2.sum() - 1.0) < 1e-12
    assert np.allclose(k2, np.outer(k, k) / np.outer(k, k).sum())


def test_gaussian_blur_preserves_constant_image():
    img = np.full((20, 20), 0.4)
    assert np.allclose(enh.gaussian_blur(img, 2.0), 0.4, atol=1e-12)


def test_separable_blur_equals_2d_convolution():
    rng = np.random.default_rng(3)
    img = rng.random((16, 16))
    sep = enh.gaussian_blur(img, 1.3)
    full = enh.correlate2d(img, enh.gaussian_kernel(sigma=1.3))
    assert np.max(np.abs(sep - full)) < 1e-10


def test_median_filter_removes_impulses_gaussian_does_not():
    img = np.full((21, 21), 0.5)
    truth = img.copy()
    img[5, 5] = 1.0            # salt
    img[12, 14] = 0.0          # pepper
    med = enh.median_filter(img, 3)
    gau = enh.gaussian_blur(img, 1.0)
    assert np.max(np.abs(med - truth)) < 1e-9
    assert np.max(np.abs(gau - truth)) > 1e-3


def test_median_filter_handles_colour_images():
    rng = np.random.default_rng(4)
    img = rng.random((12, 13, 3))
    out = enh.median_filter(img, 3)
    assert out.shape == img.shape


# --------------------------------------------------------------------------
# contrast
# --------------------------------------------------------------------------
def test_histogram_counts_every_pixel():
    rng = np.random.default_rng(5)
    img = rng.random((30, 40))
    h = enh.histogram(img, 256)
    assert h.shape == (256,)
    assert int(h.sum()) == 30 * 40


def test_cumulative_histogram_is_monotone_and_ends_at_one():
    h = enh.histogram(np.random.default_rng(6).random((20, 20)))
    c = enh.cumulative_histogram(h)
    assert np.all(np.diff(c) >= -1e-12)
    assert abs(c[-1] - 1.0) < 1e-12


def test_histogram_equalization_flattens_the_histogram():
    rng = np.random.default_rng(7)
    # a low-contrast image concentrated in a narrow band
    img = 0.45 + 0.06 * rng.random((64, 64))
    eq = enh.histogram_equalization(img)
    before = np.std(enh.histogram(img, 32))
    after = np.std(enh.histogram(eq, 32))
    assert after < before
    assert eq.min() >= 0.0 and eq.max() <= 1.0


def test_contrast_stretch_spans_the_full_range():
    rng = np.random.default_rng(8)
    img = 0.4 + 0.1 * rng.random((50, 50))
    out = enh.contrast_stretch(img, 1, 99)
    assert out.min() < 0.05 and out.max() > 0.95


# --------------------------------------------------------------------------
# sharpening
# --------------------------------------------------------------------------
def test_unsharp_mask_increases_edge_contrast():
    img = np.zeros((32, 32))
    img[:, 16:] = 1.0
    soft = enh.gaussian_blur(img, 2.0)
    sharp = enh.unsharp_mask(soft, sigma=2.0, amount=1.5)
    # gradient magnitude across the step must grow
    assert np.abs(np.diff(sharp[16])).max() > np.abs(np.diff(soft[16])).max()


def test_laplacian_kernels_sum_to_zero():
    for diagonal in (True, False):
        assert abs(enh.laplacian_kernel(diagonal).sum()) < 1e-12


def test_laplacian_sharpen_leaves_flat_regions_alone():
    img = np.full((16, 16), 0.3)
    out = enh.laplacian_sharpen(img, alpha=1.0)
    assert np.allclose(out, 0.3, atol=1e-9)


def test_enhance_for_segmentation_returns_same_shape():
    rng = np.random.default_rng(9)
    img = rng.random((40, 50, 3))
    out = enh.enhance_for_segmentation(img)
    assert out.shape == img.shape
    assert 0.0 <= out.min() and out.max() <= 1.0
