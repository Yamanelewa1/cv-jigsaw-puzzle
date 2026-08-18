"""Milestone 1 / task 1 -- Array primitives and image enhancement.

Everything in this package is written from scratch on top of NumPy only: no
OpenCV / SciPy filtering, morphology, labelling or feature code is used.
NumPy is treated as the "array + arithmetic" layer, not as an
image-processing library.  This module holds

**the shared primitives** every other module is built on

    dtype and colour conversion (:func:`to_float`, :func:`to_uint8`,
    :func:`to_gray`), border padding (:func:`pad`), the sliding-window view
    used by every neighbourhood operator, the 2-D correlation/convolution
    routine (:func:`convolve2d`), bilinear resampling
    (:func:`resize_bilinear`) and the integral image
    (:func:`integral_image`);

**the enhancement operators required by task 1**

    *noise reduction* -- :func:`gaussian_kernel`, :func:`gaussian_blur`,
    :func:`median_filter`;
    *contrast adjustment* -- :func:`histogram`,
    :func:`histogram_equalization`, :func:`contrast_stretch`;
    *sharpening* -- :func:`unsharp_mask`, :func:`laplacian_sharpen`.

All routines accept ``uint8`` or float images, grayscale or RGB, and return
float images in ``[0, 1]`` unless stated otherwise.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    # --- shared primitives -------------------------------------------------
    "to_float",
    "to_uint8",
    "to_gray",
    "pad",
    "sliding_window_view",
    "correlate2d",
    "convolve2d",
    "convolve_separable",
    "resize_bilinear",
    "integral_image",
    # --- noise reduction ---------------------------------------------------
    "mean_filter",
    "gaussian_kernel",
    "gaussian_kernel_1d",
    "gaussian_blur",
    "median_filter",
    # --- contrast adjustment -----------------------------------------------
    "histogram",
    "cumulative_histogram",
    "histogram_equalization",
    "contrast_stretch",
    # --- sharpening --------------------------------------------------------
    "unsharp_mask",
    "laplacian_kernel",
    "laplacian_sharpen",
    "enhance_for_segmentation",
]

# Rec. 601 luma weights (the same ones OpenCV/PIL use for RGB -> gray).
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


# ==========================================================================
# 0. Shared primitives
# ==========================================================================
# --------------------------------------------------------------------------
# dtype / colour helpers
# --------------------------------------------------------------------------
def to_float(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as float64 scaled to ``[0, 1]``.

    ``uint8`` input is divided by 255; floating point input is passed through
    unchanged (it is assumed to already live in ``[0, 1]``).
    """
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    if image.dtype == np.bool_:
        return image.astype(np.float64)
    return image.astype(np.float64)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip a float image in ``[0, 1]`` and convert it back to ``uint8``."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB (or RGBA) image to a single-channel float image.

    Grayscale input is returned unchanged (as float).  The conversion uses the
    Rec. 601 luma weights ``0.299 R + 0.587 G + 0.114 B``.
    """
    image = to_float(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return image[:, :, :3] @ _LUMA
    raise ValueError(f"cannot convert array of shape {image.shape} to gray")


# --------------------------------------------------------------------------
# padding
# --------------------------------------------------------------------------
def pad(image: np.ndarray, pad_y: int, pad_x: int, mode: str = "reflect",
        value: float = 0.0) -> np.ndarray:
    """Pad a 2-D or 3-D array by ``pad_y``/``pad_x`` pixels on every side.

    Supported modes:

    ``reflect``
        Mirror the interior without repeating the border pixel
        (``abcd`` -> ``cb|abcd|cb``).  This is the default because it keeps
        the local mean unbiased near the border, which matters for the
        Gaussian/median filters.
    ``replicate``
        Repeat the border pixel (``aa|abcd|dd``).
    ``constant``
        Fill with ``value``.
    """
    if pad_y == 0 and pad_x == 0:
        return image
    widths = [(pad_y, pad_y), (pad_x, pad_x)] + [(0, 0)] * (image.ndim - 2)
    if mode == "reflect":
        return np.pad(image, widths, mode="reflect")
    if mode == "replicate":
        return np.pad(image, widths, mode="edge")
    if mode == "constant":
        return np.pad(image, widths, mode="constant", constant_values=value)
    raise ValueError(f"unknown padding mode {mode!r}")


# --------------------------------------------------------------------------
# neighbourhood extraction
# --------------------------------------------------------------------------
def sliding_window_view(image: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """Return every ``kh x kw`` neighbourhood of ``image`` as a stacked view.

    For an ``(H, W)`` input the result has shape ``(H-kh+1, W-kw+1, kh, kw)``
    and shares memory with the input -- no copy is made.  This is the
    mechanism that lets us express convolution, the median filter, the
    morphological operators and non-maximum suppression as *array*
    operations instead of per-pixel Python loops.
    """
    image = np.ascontiguousarray(image)
    out_h = image.shape[0] - kh + 1
    out_w = image.shape[1] - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("window larger than image")
    shape = (out_h, out_w, kh, kw) + image.shape[2:]
    strides = image.strides[:2] + image.strides[:2] + image.strides[2:]
    return np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides,
                                           writeable=False)


# --------------------------------------------------------------------------
# convolution
# --------------------------------------------------------------------------
def correlate2d(image: np.ndarray, kernel: np.ndarray, mode: str = "reflect",
                value: float = 0.0) -> np.ndarray:
    """2-D *correlation* of ``image`` with ``kernel`` ("same" output size).

    Multi-channel images are filtered channel-by-channel.  The kernel is
    applied as-is (no flipping) -- see :func:`convolve2d` for the flipped
    version.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError("kernel must be 2-D")
    kh, kw = kernel.shape
    img = to_float(image)

    if img.ndim == 3:
        return np.stack([correlate2d(img[:, :, c], kernel, mode, value)
                         for c in range(img.shape[2])], axis=2)

    padded = pad(img, kh // 2, kw // 2, mode, value)
    windows = sliding_window_view(padded, kh, kw)
    # tensordot contracts the (kh, kw) axes of every window with the kernel.
    return np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))


def convolve2d(image: np.ndarray, kernel: np.ndarray, mode: str = "reflect",
               value: float = 0.0) -> np.ndarray:
    """True 2-D convolution: correlation with the 180-degree rotated kernel."""
    return correlate2d(image, np.asarray(kernel, dtype=np.float64)[::-1, ::-1],
                       mode, value)


def convolve_separable(image: np.ndarray, k_y: np.ndarray, k_x: np.ndarray,
                       mode: str = "reflect", value: float = 0.0) -> np.ndarray:
    """Convolve with the separable kernel ``k_y (column) x k_x (row)``.

    Two 1-D passes cost ``O(k)`` per pixel instead of ``O(k^2)``; this is used
    by the Gaussian blur, whose kernel is separable by construction.
    """
    k_y = np.asarray(k_y, dtype=np.float64).reshape(-1, 1)
    k_x = np.asarray(k_x, dtype=np.float64).reshape(1, -1)
    tmp = correlate2d(image, k_y[::-1, :], mode, value)
    return correlate2d(tmp, k_x[:, ::-1], mode, value)


# --------------------------------------------------------------------------
# geometric resampling
# --------------------------------------------------------------------------
def resize_bilinear(image: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Resize to ``out_shape = (height, width)`` with bilinear interpolation.

    Uses the "align corners" mapping, so the corners of the input land exactly
    on the corners of the output.  Needed to compare a reconstruction with a
    reference picture whose cell size differs by a few pixels.
    """
    img = to_float(image)
    single = img.ndim == 2
    if single:
        img = img[:, :, None]
    h, w = img.shape[:2]
    oh, ow = int(out_shape[0]), int(out_shape[1])
    if (oh, ow) == (h, w):
        return image if not single else np.asarray(image)

    ys = np.linspace(0, h - 1, oh) if oh > 1 else np.zeros(1)
    xs = np.linspace(0, w - 1, ow) if ow > 1 else np.zeros(1)
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, max(h - 2, 0))
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, max(w - 2, 0))
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)

    top = img[np.ix_(y0, x0)] * (1 - wx) + img[np.ix_(y0, x1)] * wx
    bot = img[np.ix_(y1, x0)] * (1 - wx) + img[np.ix_(y1, x1)] * wx
    out = top * (1 - wy) + bot * wy
    return out[:, :, 0] if single else out


# --------------------------------------------------------------------------
# integral image
# --------------------------------------------------------------------------
def integral_image(image: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero first row/column.

    ``S[y, x]`` is the sum of ``image[:y, :x]``, so the sum over any window
    is obtained with four lookups in constant time -- this is what makes
    ``adaptive_threshold`` independent of the block size.
    """
    img = to_float(image)
    out = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
    return out

# ==========================================================================
# 1a. Noise reduction
# ==========================================================================
def mean_filter(image: np.ndarray, size: int = 3,
                mode: str = "reflect") -> np.ndarray:
    """Arithmetic mean (box) filter over a ``size x size`` window.

    The simplest linear smoother: every pixel is replaced by the unweighted
    average of its neighbourhood.  It is the maximum-likelihood estimator of a
    constant patch corrupted by additive Gaussian noise, so it suppresses that
    noise by a factor of ``size``.

    It is nevertheless not what the pipeline smooths with.  Because the kernel
    is flat, its frequency response is a sinc with large side lobes, and in
    the spatial domain it turns a step into a *piecewise linear* ramp whose
    slope jumps at both ends -- visible as hard corners along an edge, where
    the Gaussian's response is smooth everywhere.  It is provided because the
    brief asks for mean, Gaussian and median noise reduction, and because it
    is the baseline the other two are measured against in the report.

    Like the Gaussian, the box kernel is separable (a 2-D average is a
    horizontal average of vertical averages), so it runs as two 1-D passes at
    ``O(k)`` per pixel rather than ``O(k^2)``.
    """
    size = int(size)
    if size < 1:
        raise ValueError("size must be >= 1")
    if size % 2 == 0:
        size += 1
    if size == 1:
        return to_float(image)
    k = np.full(size, 1.0 / size, dtype=np.float64)
    return convolve_separable(image, k, k, mode)


def gaussian_kernel_1d(size: int | None = None, sigma: float = 1.0) -> np.ndarray:
    """Sampled, normalised 1-D Gaussian ``G(x) = exp(-x^2 / 2 sigma^2)``.

    The kernel is *derived from* both the requested size and sigma, as the
    brief demands.  If ``size`` is omitted it defaults to
    ``2 * ceil(3 * sigma) + 1`` so that the truncated tails hold < 0.3 % of
    the mass.  ``size`` is forced odd so the kernel has a well-defined centre.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if size is None:
        size = int(2 * np.ceil(3.0 * sigma) + 1)
    size = int(size)
    if size < 1:
        raise ValueError("size must be >= 1")
    if size % 2 == 0:
        size += 1
    r = (size - 1) // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_kernel(size: int | None = None, sigma: float = 1.0) -> np.ndarray:
    """2-D Gaussian kernel obtained as the outer product of two 1-D kernels.

    The 2-D Gaussian is separable, ``G(x, y) = G(x) G(y)``, which is exactly
    why :func:`gaussian_blur` can run two 1-D passes.
    """
    k = gaussian_kernel_1d(size, sigma)
    k2 = np.outer(k, k)
    return k2 / k2.sum()


def gaussian_blur(image: np.ndarray, sigma: float = 1.0,
                  size: int | None = None, mode: str = "reflect") -> np.ndarray:
    """Gaussian smoothing via two separable 1-D convolutions.

    Cost is ``O(k)`` per pixel instead of ``O(k^2)``; the result is identical
    to convolving with :func:`gaussian_kernel` up to floating point error.
    """
    k = gaussian_kernel_1d(size, sigma)
    return convolve_separable(image, k, k, mode)


def median_filter(image: np.ndarray, size: int = 3,
                  mode: str = "reflect") -> np.ndarray:
    """Median filter over a ``size x size`` window.

    Why a loop is needed (and how it is avoided)
    --------------------------------------------
    The median is a *rank* statistic: unlike the Gaussian it is not a linear
    operator, so it cannot be written as a convolution and there is no
    separable/​recursive form.  Every output pixel genuinely requires the
    sorted order of its own ``size^2`` neighbours, i.e. an explicit pass over
    the window.

    Rather than writing that pass as a Python ``for`` loop over pixels (which
    would be ~10^6 iterations for a 1 MP image), we materialise all windows
    at once with :func:`~src.core.sliding_window_view` and let a single
    vectorised ``np.median`` perform the ``H*W`` independent selections in
    compiled code.  The only Python-level loop that remains is over the
    (at most 3) colour channels.  Memory is bounded by processing the image
    in horizontal stripes.
    """
    size = int(size)
    if size < 1:
        raise ValueError("size must be >= 1")
    if size % 2 == 0:
        size += 1
    if size == 1:
        return to_float(image)

    img = to_float(image)
    if img.ndim == 3:
        # One pass per channel: a colour median must be taken per band,
        # otherwise channels would be mixed and colours invented.
        return np.stack([median_filter(img[:, :, c], size, mode)
                         for c in range(img.shape[2])], axis=2)

    r = size // 2
    padded = pad(img, r, r, mode)
    h = img.shape[0]
    out = np.empty_like(img)
    # Stripe height chosen so each temporary window block stays ~<= 64 MB.
    stripe = max(1, int(64e6 / (8.0 * max(img.shape[1], 1) * size * size)))
    for y0 in range(0, h, stripe):
        y1 = min(h, y0 + stripe)
        block = padded[y0:y1 + 2 * r]
        win = sliding_window_view(block, size, size)
        out[y0:y1] = np.median(win.reshape(win.shape[0], win.shape[1], -1),
                               axis=2)
    return out


# ==========================================================================
# 1b. Contrast adjustment
# ==========================================================================
def histogram(image: np.ndarray, bins: int = 256,
              value_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Intensity histogram computed from scratch with ``np.bincount``.

    The image is quantised to ``bins`` levels over ``value_range`` and the
    occurrences of each level are counted.  Colour images are converted to
    luma first.  Returned array has length ``bins``.
    """
    img = to_gray(image)
    lo, hi = value_range
    scaled = (img - lo) / max(hi - lo, 1e-12)
    idx = np.clip(np.floor(scaled * bins).astype(np.int64), 0, bins - 1)
    return np.bincount(idx.ravel(), minlength=bins).astype(np.float64)


def cumulative_histogram(hist: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Cumulative distribution of a histogram (optionally scaled to ``[0,1]``)."""
    cdf = np.cumsum(np.asarray(hist, dtype=np.float64))
    if normalize and cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return cdf


def histogram_equalization(image: np.ndarray, bins: int = 256,
                           per_channel: bool = False) -> np.ndarray:
    """Global histogram equalisation.

    The transfer function is the normalised CDF of the image histogram,
    ``s = CDF(r)``, which maps the intensity distribution towards uniform and
    therefore stretches the most populated (visually dominant) ranges.

    For colour input the default is to equalise the *luma* only and re-apply
    the original chroma ratios, which preserves hue; ``per_channel=True``
    equalises R, G and B independently (stronger, but shifts colours).
    """
    img = to_float(image)
    if img.ndim == 3 and per_channel:
        return np.stack([histogram_equalization(img[:, :, c], bins)
                         for c in range(img.shape[2])], axis=2)

    gray = to_gray(img) if img.ndim == 3 else img
    hist = histogram(gray, bins)
    cdf = cumulative_histogram(hist)
    idx = np.clip(np.floor(gray * bins).astype(np.int64), 0, bins - 1)
    eq = cdf[idx]

    if img.ndim == 2:
        return eq
    # Scale each channel by the same luma gain to keep the hue constant.
    gain = np.where(gray > 1e-6, eq / np.maximum(gray, 1e-6), 1.0)
    return np.clip(img * gain[:, :, None], 0.0, 1.0)


def contrast_stretch(image: np.ndarray, low_percentile: float = 2.0,
                     high_percentile: float = 98.0,
                     bins: int = 256) -> np.ndarray:
    """Linear contrast stretching between two histogram percentiles.

    ``out = (in - lo) / (hi - lo)`` clipped to ``[0, 1]``, where ``lo``/``hi``
    are read off the *cumulative histogram* built by this library (not from
    ``np.percentile``), so the histogram machinery is genuinely used.
    Percentiles rather than the raw min/max make the operator robust to a
    handful of hot/dead pixels.
    """
    img = to_float(image)
    gray = to_gray(img) if img.ndim == 3 else img
    cdf = cumulative_histogram(histogram(gray, bins))
    centers = (np.arange(bins) + 0.5) / bins
    lo = float(centers[np.searchsorted(cdf, low_percentile / 100.0,
                                       side="left").clip(0, bins - 1)])
    hi = float(centers[np.searchsorted(cdf, high_percentile / 100.0,
                                       side="left").clip(0, bins - 1)])
    if hi - lo < 1e-6:
        lo, hi = float(gray.min()), float(max(gray.max(), gray.min() + 1e-6))
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


# ==========================================================================
# 1c. Sharpening
# ==========================================================================
def laplacian_kernel(diagonal: bool = True) -> np.ndarray:
    """Discrete Laplacian ``∇^2``: 4-neighbour or 8-neighbour variant."""
    if diagonal:
        return np.array([[1.0, 1.0, 1.0],
                         [1.0, -8.0, 1.0],
                         [1.0, 1.0, 1.0]])
    return np.array([[0.0, 1.0, 0.0],
                     [1.0, -4.0, 1.0],
                     [0.0, 1.0, 0.0]])


def laplacian_sharpen(image: np.ndarray, alpha: float = 1.0,
                      diagonal: bool = True) -> np.ndarray:
    """Sharpen with the Laplacian operator: ``out = in - alpha * ∇^2 in``.

    The Laplacian responds to the second derivative, so subtracting it boosts
    exactly the high-frequency content.  Built on the shared
    :func:`~src.core.convolve2d` routine.
    """
    img = to_float(image)
    lap = convolve2d(img, laplacian_kernel(diagonal))
    return np.clip(img - alpha * lap, 0.0, 1.0)


def unsharp_mask(image: np.ndarray, sigma: float = 1.5, amount: float = 1.0,
                 threshold: float = 0.0) -> np.ndarray:
    """Unsharp masking: ``out = in + amount * (in - blur(in))``.

    ``in - blur(in)`` is the high-pass residual; adding a multiple of it back
    increases local contrast at edges.  ``threshold`` suppresses the boost
    where the residual is small, so flat, noisy regions are left alone.
    """
    img = to_float(image)
    blurred = gaussian_blur(img, sigma)
    detail = img - blurred
    if threshold > 0:
        detail = np.where(np.abs(detail) < threshold, 0.0, detail)
    return np.clip(img + amount * detail, 0.0, 1.0)


# ==========================================================================
# convenience pipeline
# ==========================================================================
def enhance_for_segmentation(image: np.ndarray, median_size: int = 3,
                             gaussian_sigma: float = 1.0,
                             stretch: bool = True,
                             sharpen_amount: float = 0.6) -> np.ndarray:
    """The enhancement chain the puzzle pipeline uses by default.

    Order and rationale:

    1. **median** -- removes salt-and-pepper style sensor speckle *before*
       any linear filter, because a linear filter would smear impulses
       instead of deleting them;
    2. **Gaussian** -- suppresses the remaining Gaussian sensor noise and
       fabric texture of the background cloth;
    3. **contrast stretch** -- normalises exposure differences between photos
       so a single threshold behaves consistently;
    4. **unsharp mask** -- restores the piece outline softened by steps 1-2.
    """
    out = to_float(image)
    if median_size and median_size > 1:
        out = median_filter(out, median_size)
    if gaussian_sigma and gaussian_sigma > 0:
        out = gaussian_blur(out, gaussian_sigma)
    if stretch:
        out = contrast_stretch(out, 1.0, 99.0)
    if sharpen_amount and sharpen_amount > 0:
        out = unsharp_mask(out, sigma=1.5, amount=sharpen_amount)
    return out
