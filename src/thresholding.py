"""Milestone 1 / task 1d -- Thresholding, implemented from scratch.

* :func:`global_threshold` -- fixed level, or the classic iterative
  (isodata) level selection.
* :func:`otsu_threshold` / :func:`otsu` -- between-class variance maximisation
  computed directly from the library's own histogram.
* :func:`adaptive_threshold` -- mean or Gaussian local thresholding, with the
  mean variant evaluated in O(1) per pixel through an integral image.
"""

from __future__ import annotations

import numpy as np

from .enhancement import integral_image, to_float, to_gray
from .enhancement import gaussian_blur, histogram

__all__ = [
    "global_threshold",
    "isodata_threshold",
    "otsu_threshold",
    "otsu",
    "adaptive_threshold",
    "threshold_image",
]


# --------------------------------------------------------------------------
# global
# --------------------------------------------------------------------------
def global_threshold(image: np.ndarray, level: float = 0.5,
                     invert: bool = False) -> np.ndarray:
    """Binarise with a single fixed ``level``; returns a boolean mask."""
    gray = to_gray(image)
    mask = gray > level
    return ~mask if invert else mask


def isodata_threshold(image: np.ndarray, bins: int = 256,
                      max_iter: int = 100, tol: float = 1e-5) -> float:
    """Iterative (isodata / Ridler-Calvard) global threshold selection.

    Starts at the mean intensity and repeatedly sets the threshold to the
    average of the two class means until it stops moving.  Cheap, and a good
    sanity check against Otsu on bimodal images.
    """
    gray = to_gray(image)
    t = float(gray.mean())
    for _ in range(max_iter):
        lo = gray[gray <= t]
        hi = gray[gray > t]
        if lo.size == 0 or hi.size == 0:
            break
        t_new = 0.5 * (lo.mean() + hi.mean())
        if abs(t_new - t) < tol:
            t = t_new
            break
        t = t_new
    return float(t)


def otsu_threshold(image: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold: the level maximising between-class variance.

    With ``w0``/``w1`` the class probabilities and ``mu0``/``mu1`` the class
    means, we maximise

        ``sigma_b^2(t) = w0(t) w1(t) (mu0(t) - mu1(t))^2``

    which is equivalent to minimising the intra-class variance but needs only
    cumulative sums of the histogram, so all ``bins`` candidates are evaluated
    at once in a vectorised expression.
    """
    hist = histogram(image, bins)
    p = hist / max(hist.sum(), 1e-12)
    centers = (np.arange(bins) + 0.5) / bins

    w0 = np.cumsum(p)                       # P(class 0) for each split
    w1 = 1.0 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]

    valid = (w0 > 1e-12) & (w1 > 1e-12)
    sigma_b = np.zeros(bins, dtype=np.float64)
    sigma_b[valid] = ((mu_t * w0[valid] - mu[valid]) ** 2
                      / (w0[valid] * w1[valid]))

    # When the two modes are well separated the criterion is *flat* across
    # the whole empty valley between them, so ``argmax`` would return its
    # left edge -- a threshold sitting right against the darker mode.  The
    # midpoint of the maximising plateau is the same optimum but as far from
    # both modes as possible, which is what makes the split noise tolerant.
    peak = float(sigma_b.max())
    plateau = np.flatnonzero(sigma_b >= peak - 1e-12 * max(peak, 1.0))
    return float(centers[plateau].mean())


def otsu(image: np.ndarray, bins: int = 256, invert: bool = False) -> np.ndarray:
    """Binarise ``image`` at the Otsu level; returns a boolean mask."""
    t = otsu_threshold(image, bins)
    return global_threshold(image, t, invert)


# --------------------------------------------------------------------------
# adaptive / local
# --------------------------------------------------------------------------
def adaptive_threshold(image: np.ndarray, block_size: int = 51, c: float = 0.02,
                       method: str = "mean", invert: bool = False,
                       sigma: float | None = None) -> np.ndarray:
    """Local thresholding: ``pixel > local_statistic - c``.

    Parameters
    ----------
    block_size:
        Side of the local window (forced odd).
    c:
        Constant subtracted from the local statistic; raising it makes the
        operator more conservative about calling a pixel foreground.
    method:
        ``"mean"`` uses a box average evaluated with an integral image, so
        the cost does not depend on ``block_size``.  ``"gaussian"`` uses a
        Gaussian-weighted average (``sigma`` defaults to ``block_size / 6``),
        which weights nearby pixels more and produces smoother boundaries.

    Local thresholding is what makes segmentation survive the strong
    illumination gradient across the dataset photographs, where one global
    level clips the darker corners.
    """
    gray = to_gray(image)
    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1
    r = block_size // 2

    if method == "mean":
        integ = integral_image(gray)
        h, w = gray.shape
        ys = np.arange(h)
        xs = np.arange(w)
        y0 = np.clip(ys - r, 0, h)[:, None]
        y1 = np.clip(ys + r + 1, 0, h)[:, None]
        x0 = np.clip(xs - r, 0, w)[None, :]
        x1 = np.clip(xs + r + 1, 0, w)[None, :]
        total = (integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
        count = (y1 - y0) * (x1 - x0)
        local = total / np.maximum(count, 1)
    elif method == "gaussian":
        local = gaussian_blur(gray, sigma if sigma else block_size / 6.0,
                              size=block_size)
    else:
        raise ValueError("method must be 'mean' or 'gaussian'")

    mask = gray > (local - c)
    return ~mask if invert else mask


def threshold_image(image: np.ndarray, method: str = "otsu", **kwargs) -> np.ndarray:
    """Dispatch helper: ``method`` in ``{global, isodata, otsu, adaptive}``."""
    if method == "global":
        return global_threshold(image, **kwargs)
    if method == "isodata":
        invert = kwargs.pop("invert", False)
        return global_threshold(image, isodata_threshold(image, **kwargs), invert)
    if method == "otsu":
        return otsu(image, **kwargs)
    if method == "adaptive":
        return adaptive_threshold(image, **kwargs)
    raise ValueError(f"unknown thresholding method {method!r}")
