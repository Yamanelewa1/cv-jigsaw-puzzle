"""Tests for :mod:`src.thresholding` (task 1d)."""

import numpy as np

from src import thresholding as th


def _bimodal(seed=0, lo=0.2, hi=0.8, size=(64, 64)):
    """Half the pixels around ``lo``, half around ``hi``."""
    rng = np.random.default_rng(seed)
    img = np.empty(size)
    img[:, :size[1] // 2] = lo + 0.05 * rng.standard_normal(
        (size[0], size[1] // 2))
    img[:, size[1] // 2:] = hi + 0.05 * rng.standard_normal(
        (size[0], size[1] - size[1] // 2))
    return np.clip(img, 0, 1)


def test_global_threshold_and_inversion():
    img = np.array([[0.1, 0.9]])
    assert np.array_equal(th.global_threshold(img, 0.5), [[False, True]])
    assert np.array_equal(th.global_threshold(img, 0.5, invert=True),
                          [[True, False]])


def test_otsu_lands_between_the_two_modes():
    img = _bimodal()
    t = th.otsu_threshold(img)
    assert 0.2 < t < 0.8
    mask = th.otsu(img)
    # the split should recover the two halves almost exactly
    assert mask[:, 32:].mean() > 0.98
    assert mask[:, :32].mean() < 0.02


def test_otsu_agrees_with_isodata_on_a_clean_image():
    img = _bimodal(seed=1)
    assert abs(th.otsu_threshold(img) - th.isodata_threshold(img)) < 0.05


def test_otsu_maximises_between_class_variance():
    """Brute-force check of the criterion Otsu is supposed to optimise."""
    img = _bimodal(seed=2)
    t = th.otsu_threshold(img)

    def between_class(level):
        a, b = img[img <= level], img[img > level]
        if a.size == 0 or b.size == 0:
            return -1.0
        w0, w1 = a.size / img.size, b.size / img.size
        return w0 * w1 * (a.mean() - b.mean()) ** 2

    best = max(np.linspace(0.05, 0.95, 91), key=between_class)
    assert abs(between_class(t) - between_class(best)) < 1e-4


def test_adaptive_mean_equals_the_local_box_mean():
    rng = np.random.default_rng(3)
    img = rng.random((21, 21))
    block, c = 5, 0.0
    mask = th.adaptive_threshold(img, block, c, "mean")
    r = block // 2
    y, x = 10, 10
    local = img[y - r:y + r + 1, x - r:x + r + 1].mean()
    assert bool(mask[y, x]) == bool(img[y, x] > local - c)


def test_adaptive_beats_global_under_an_illumination_gradient():
    # a constant-contrast pattern under a strong left-to-right ramp
    yy, xx = np.mgrid[0:64, 0:64]
    pattern = ((xx // 8) % 2 == 0).astype(float) * 0.25 + 0.15
    img = np.clip(pattern + xx / 64.0 * 0.7, 0, 1)
    truth = (xx // 8) % 2 == 0

    glob = th.otsu(img)
    adap = th.adaptive_threshold(img, 21, 0.01, "mean")
    acc_g = max((glob == truth).mean(), (~glob == truth).mean())
    acc_a = max((adap == truth).mean(), (~adap == truth).mean())
    assert acc_a > acc_g


def test_gaussian_and_mean_adaptive_broadly_agree():
    img = _bimodal(seed=4)
    a = th.adaptive_threshold(img, 15, 0.02, "mean")
    b = th.adaptive_threshold(img, 15, 0.02, "gaussian")
    assert (a == b).mean() > 0.9


def test_threshold_image_dispatch():
    img = _bimodal(seed=5)
    for method, kwargs in [("global", {"level": 0.5}), ("isodata", {}),
                           ("otsu", {}), ("adaptive", {"block_size": 15})]:
        out = th.threshold_image(img, method, **kwargs)
        assert out.dtype == np.bool_ and out.shape == img.shape
