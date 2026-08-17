"""Milestone 1 / task 2 -- Edge detection, implemented from scratch.

* :func:`sobel` and :func:`prewitt` return gradient magnitude **and**
  orientation.
* :func:`canny` is the complete detector: Gaussian smoothing, gradient
  computation, non-maximum suppression, double thresholding and
  hysteresis-based edge linking.

Default parameters and their effect are documented on each function and
illustrated by ``main.py --demo edges``.
"""

from __future__ import annotations

import numpy as np

from .enhancement import correlate2d, pad, to_gray
from .enhancement import gaussian_blur

__all__ = [
    "SOBEL_X", "SOBEL_Y", "PREWITT_X", "PREWITT_Y",
    "gradient",
    "sobel",
    "prewitt",
    "non_maximum_suppression",
    "double_threshold",
    "hysteresis",
    "canny",
]

# Both operators are separable smoothing x differencing kernels.  Sobel's
# [1 2 1] smoother weights the centre row twice as much as Prewitt's
# [1 1 1], so Sobel is slightly less noise sensitive and less "boxy".
SOBEL_X = np.array([[-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0]])
SOBEL_Y = np.array([[-1.0, -2.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 1.0]])
PREWITT_X = np.array([[-1.0, 0.0, 1.0],
                      [-1.0, 0.0, 1.0],
                      [-1.0, 0.0, 1.0]])
PREWITT_Y = np.array([[-1.0, -1.0, -1.0],
                      [0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0]])


def gradient(image: np.ndarray, kx: np.ndarray, ky: np.ndarray,
             normalize: bool = True):
    """Apply a pair of derivative kernels and return ``(gx, gy, mag, theta)``.

    ``theta`` is the gradient direction in radians in ``(-pi, pi]``, measured
    with ``x`` to the right and ``y`` downwards (image convention).
    ``mag`` is optionally scaled to ``[0, 1]``.
    """
    gray = to_gray(image)
    gx = correlate2d(gray, kx)
    gy = correlate2d(gray, ky)
    mag = np.hypot(gx, gy)
    if normalize and mag.max() > 0:
        mag = mag / mag.max()
    theta = np.arctan2(gy, gx)
    return gx, gy, mag, theta


def sobel(image: np.ndarray, normalize: bool = True):
    """Sobel gradient. Returns ``(gx, gy, magnitude, orientation)``."""
    return gradient(image, SOBEL_X, SOBEL_Y, normalize)


def prewitt(image: np.ndarray, normalize: bool = True):
    """Prewitt gradient. Returns ``(gx, gy, magnitude, orientation)``."""
    return gradient(image, PREWITT_X, PREWITT_Y, normalize)


# --------------------------------------------------------------------------
# Canny stages
# --------------------------------------------------------------------------
def non_maximum_suppression(mag: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Thin the gradient magnitude to one-pixel ridges.

    The continuous gradient direction is quantised to the 4 available
    discrete orientations (0, 45, 90, 135 degrees); a pixel survives only if
    its magnitude is >= both neighbours along that direction.  Implemented by
    shifting the whole array once per orientation -- no per-pixel loop.
    """
    padded = pad(mag, 1, 1, "constant", 0.0)
    # neighbour pairs for each quantised direction, as (dy, dx)
    offsets = {
        0: ((0, 1), (0, -1)),     # gradient horizontal  -> compare left/right
        1: ((-1, 1), (1, -1)),    # 45 deg
        2: ((-1, 0), (1, 0)),     # gradient vertical    -> compare up/down
        3: ((-1, -1), (1, 1)),    # 135 deg
    }
    # map angle (mod 180 deg) to bucket 0..3
    ang = np.rad2deg(theta) % 180.0
    bucket = np.floor((ang + 22.5) / 45.0).astype(np.int64) % 4

    h, w = mag.shape
    out = np.zeros_like(mag)
    for b, ((dy1, dx1), (dy2, dx2)) in offsets.items():
        sel = bucket == b
        if not sel.any():
            continue
        n1 = padded[1 + dy1:1 + dy1 + h, 1 + dx1:1 + dx1 + w]
        n2 = padded[1 + dy2:1 + dy2 + h, 1 + dx2:1 + dx2 + w]
        keep = sel & (mag >= n1) & (mag >= n2)
        out[keep] = mag[keep]
    return out


def double_threshold(nms: np.ndarray, low: float, high: float):
    """Split the thinned magnitude into strong and weak edge pixels."""
    strong = nms >= high
    weak = (nms >= low) & ~strong
    return strong, weak


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Edge linking: keep weak pixels 8-connected to a strong pixel.

    This is a binary geodesic reconstruction of ``strong`` inside
    ``strong | weak``, i.e. exactly the transitive closure Canny asks for.
    It is evaluated with one connected-component labelling pass
    (:func:`src.segmentation.reconstruct`) instead of an iterative flood, so
    the cost does not depend on how long the edge chains are, and there is no
    recursion-depth limit.
    """
    from .segmentation import reconstruct        # local: avoids import cycle
    return reconstruct(strong, strong | weak, connectivity=8)


def canny(image: np.ndarray, sigma: float = 1.4, low: float = 0.05,
          high: float = 0.15, use_percentiles: bool = False,
          return_stages: bool = False):
    """Full Canny edge detector.

    Parameters
    ----------
    sigma:
        Std-dev of the Gaussian pre-smoothing.  Small values (~1.0) keep fine
        detail such as the printed picture inside a piece; larger values
        (~2.0) leave essentially only the piece silhouette.  ``1.4`` is the
        default because it removes the fabric texture of the dataset's black
        cloth while keeping the tab/blank curvature intact.
    low, high:
        Hysteresis levels on the *normalised* gradient magnitude
        (``0 <= mag <= 1``).  A ratio around ``high = 3 * low`` is the usual
        recommendation and is what the defaults implement.
    use_percentiles:
        Interpret ``low``/``high`` as percentiles (0-100) of the non-zero
        magnitudes instead of absolute levels, which makes the detector
        exposure independent across the dataset.
    return_stages:
        Also return the intermediate arrays for the report figures.

    Returns
    -------
    Boolean edge map, or ``(edges, stages_dict)`` when ``return_stages``.
    """
    smoothed = gaussian_blur(image, sigma)
    gx, gy, mag, theta = sobel(smoothed)
    nms = non_maximum_suppression(mag, theta)

    if use_percentiles:
        nz = nms[nms > 0]
        if nz.size:
            low_v = float(np.percentile(nz, low))
            high_v = float(np.percentile(nz, high))
        else:
            low_v = high_v = 1.0
    else:
        low_v, high_v = float(low), float(high)
    if high_v < low_v:
        low_v, high_v = high_v, low_v

    strong, weak = double_threshold(nms, low_v, high_v)
    edges = hysteresis(strong, weak)

    if return_stages:
        stages = {
            "smoothed": smoothed,
            "gx": gx, "gy": gy,
            "magnitude": mag,
            "orientation": theta,
            "nms": nms,
            "strong": strong,
            "weak": weak,
            "low": low_v,
            "high": high_v,
        }
        return edges, stages
    return edges
