"""Milestone 1 / task 3a-b -- Foreground segmentation and connected components.

Provides, from scratch:

* binary morphology (:func:`erode`, :func:`dilate`, :func:`opening`,
  :func:`closing`, :func:`fill_holes`, :func:`remove_border_components`),
* :func:`foreground_mask` -- pieces vs. background, using the thresholding
  routines of :mod:`src.thresholding`,
* :func:`connected_components` -- two-pass labelling with union-find,
* :func:`component_stats` / :func:`filter_components` -- area filtering that
  drops the screwdrivers, bolts and cloth folds present in the dataset
  photographs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .enhancement import to_float, to_gray
from .enhancement import gaussian_blur, median_filter
from .thresholding import adaptive_threshold, otsu_threshold

__all__ = [
    "disk",
    "erode",
    "dilate",
    "majority_smooth",
    "opening",
    "closing",
    "fill_holes",
    "reconstruct",
    "remove_border_components",
    "background_color",
    "background_distance",
    "foreground_mask",
    "connected_components",
    "distance_transform",
    "split_touching",
    "ComponentStats",
    "component_stats",
    "filter_components",
]


# ==========================================================================
# structuring elements + binary morphology
# ==========================================================================
def disk(radius: int) -> np.ndarray:
    """Boolean disk structuring element of the given radius."""
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 1e-9


def _as_selem(selem: np.ndarray | int) -> np.ndarray:
    """Accept an integer radius or an explicit boolean structuring element."""
    if np.isscalar(selem):
        return disk(int(selem))
    return np.asarray(selem, dtype=bool)


def _shifted(mask: np.ndarray, dy: int, dx: int, fill: bool) -> np.ndarray:
    """``mask`` translated by ``(dy, dx)``, padded with ``fill``."""
    out = np.full_like(mask, fill, dtype=bool)
    h, w = mask.shape
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    dy0, dy1 = max(0, dy), min(h, h + dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    dx0, dx1 = max(0, dx), min(w, w + dx)
    if sy0 < sy1 and sx0 < sx1:
        out[dy0:dy1, dx0:dx1] = mask[sy0:sy1, sx0:sx1]
    return out


def erode(mask: np.ndarray, selem: np.ndarray | int = 3) -> np.ndarray:
    """Binary erosion: a pixel survives only if the whole SE fits inside.

    Implemented as an intersection of translated copies of the mask -- one
    per element of the structuring element -- which needs only ``|SE|``
    boolean AND passes and no ``H*W*|SE|`` temporary.

    Outside the image the mask is treated as *foreground* so that objects
    touching the border are not eaten away from the frame edge.
    """
    mask = np.asarray(mask, dtype=bool)
    se = _as_selem(selem)
    ry, rx = se.shape[0] // 2, se.shape[1] // 2
    out = np.ones_like(mask)
    for oy, ox in zip(*np.nonzero(se)):
        out &= _shifted(mask, int(oy) - ry, int(ox) - rx, fill=True)
    return out


def dilate(mask: np.ndarray, selem: np.ndarray | int = 3) -> np.ndarray:
    """Binary dilation: a pixel is set if the SE hits any foreground pixel.

    Dual of :func:`erode`: a union of translated copies of the mask.
    """
    mask = np.asarray(mask, dtype=bool)
    se = _as_selem(selem)
    ry, rx = se.shape[0] // 2, se.shape[1] // 2
    out = np.zeros_like(mask)
    for oy, ox in zip(*np.nonzero(se)):
        out |= _shifted(mask, ry - int(oy), rx - int(ox), fill=False)
    return out


def majority_smooth(mask: np.ndarray, radius: int = 2,
                    threshold: float = 0.5) -> np.ndarray:
    """Self-dual boundary smoothing: keep a pixel if most of its disk agrees.

    Why not opening followed by closing?  Opening only rounds *convex*
    features and closing only rounds *concave* ones, so applying them in
    sequence distorts a tab and the blank it fits into by different amounts
    and destroys the complementarity the shape matcher relies on.  A majority
    (binary median) filter treats foreground and background symmetrically --
    it is self-dual -- so a tab and its blank stay mirror images of each
    other.
    """
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    se = disk(int(radius))
    ry, rx = se.shape[0] // 2, se.shape[1] // 2
    ones = np.ones(mask.shape, dtype=bool)
    count = np.zeros(mask.shape, dtype=np.int32)
    # Only neighbours that actually exist are counted, so the rule stays
    # self-dual next to the image border as well as in the interior.
    total = np.zeros(mask.shape, dtype=np.int32)
    for oy, ox in zip(*np.nonzero(se)):
        dy, dx = ry - int(oy), rx - int(ox)
        count += _shifted(mask, dy, dx, fill=False)
        total += _shifted(ones, dy, dx, fill=False)
    return count > threshold * total


def opening(mask: np.ndarray, selem: np.ndarray | int = 3) -> np.ndarray:
    """Erosion then dilation -- deletes speckle smaller than the SE."""
    return dilate(erode(mask, selem), selem)


def closing(mask: np.ndarray, selem: np.ndarray | int = 3) -> np.ndarray:
    """Dilation then erosion -- seals pin-holes and hairline cracks."""
    return erode(dilate(mask, selem), selem)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that are not connected to the image border.

    The complement of the mask is labelled with :func:`connected_components`;
    every background component that does **not** touch the frame is an
    enclosed hole and gets filled.  (Doing it with connected components
    instead of an iterative flood keeps the cost at one labelling pass rather
    than one dilation per pixel of hole radius.)

    This repairs the specular highlights inside a piece that Otsu
    occasionally classifies as background.
    """
    mask = np.asarray(mask, dtype=bool)
    bg = ~mask
    if not bg.any():
        return mask
    labels, n = connected_components(bg, connectivity=4)
    if n == 0:
        return mask
    border = np.concatenate([labels[0, :], labels[-1, :],
                             labels[:, 0], labels[:, -1]])
    outside = np.unique(border[border > 0])
    is_hole = np.ones(n + 1, dtype=bool)
    is_hole[0] = False
    is_hole[outside] = False
    return mask | is_hole[labels]


def reconstruct(seed: np.ndarray, allowed: np.ndarray,
                connectivity: int = 8) -> np.ndarray:
    """Binary geodesic reconstruction of ``seed`` inside ``allowed``.

    Keeps every connected component of ``allowed`` that contains at least one
    seed pixel.  Used for Canny's hysteresis edge linking.
    """
    seed = np.asarray(seed, dtype=bool) & np.asarray(allowed, dtype=bool)
    allowed = np.asarray(allowed, dtype=bool)
    if not seed.any():
        return np.zeros_like(allowed)
    labels, n = connected_components(allowed, connectivity)
    if n == 0:
        return np.zeros_like(allowed)
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(labels[seed])] = True
    keep[0] = False
    return keep[labels]


def remove_border_components(mask: np.ndarray, labels: np.ndarray | None = None
                             ) -> np.ndarray:
    """Drop objects touching the image frame (pieces cut off by the crop)."""
    if labels is None:
        labels, _ = connected_components(mask)
    border = np.concatenate([labels[0, :], labels[-1, :],
                             labels[:, 0], labels[:, -1]])
    bad = set(int(v) for v in np.unique(border) if v != 0)
    if not bad:
        return mask.astype(bool)
    keep = np.ones(int(labels.max()) + 1, dtype=bool)
    for b in bad:
        keep[b] = False
    keep[0] = False
    return keep[labels]


# ==========================================================================
# foreground mask
# ==========================================================================
def background_color(image: np.ndarray, border: int = 8) -> np.ndarray:
    """Estimate the background colour as the median of a frame of pixels.

    The outermost ``border`` pixels of a puzzle photograph are essentially
    always table/cloth, and the median is insensitive to the occasional piece
    that touches the frame.
    """
    img = to_float(image)
    if img.ndim == 2:
        img = img[:, :, None]
    b = max(1, int(border))
    frame = np.concatenate([img[:b, :].reshape(-1, img.shape[2]),
                            img[-b:, :].reshape(-1, img.shape[2]),
                            img[:, :b].reshape(-1, img.shape[2]),
                            img[:, -b:].reshape(-1, img.shape[2])], axis=0)
    return np.median(frame, axis=0)


def background_distance(image: np.ndarray, border: int = 8) -> np.ndarray:
    """Per-pixel Euclidean colour distance to the estimated background.

    Thresholding *this* instead of the luma is what makes segmentation
    survive dark picture content on a dark table: a navy-blue patch printed
    on a piece has almost the same luma as a black cloth but a very
    different chroma, so it stays part of the piece.
    """
    img = to_float(image)
    if img.ndim == 2:
        img = img[:, :, None]
    bg = background_color(img, border)
    d = np.sqrt(((img - bg[None, None, :]) ** 2).sum(axis=2))
    m = float(d.max())
    return d / m if m > 0 else d


def foreground_mask(image: np.ndarray, method: str = "auto",
                    smooth_sigma: float = 1.0, median_size: int = 3,
                    open_radius: int = 3, close_radius: int = 3,
                    smooth_radius: int = 0,
                    fill: bool = True, block_size: int = 101,
                    c: float = 0.02, invert: bool | None = None,
                    border: int = 8) -> np.ndarray:
    """Separate puzzle pieces from the background.

    Parameters
    ----------
    method:
        ``"otsu"``
            Global Otsu on the luma channel; the polarity (pieces bright or
            dark) is decided from a thin frame around the image, which is
            overwhelmingly background in every photograph of the dataset.
        ``"adaptive"``
            Local mean thresholding (``block_size``, ``c``) -- the choice
            when the illumination varies strongly across the table.
        ``"background"``
            Otsu applied to the *colour distance* to the estimated background
            (see :func:`background_distance`).  Most robust for coloured
            pieces on a uniform table.
        ``"auto"``
            ``"background"`` for colour input, ``"otsu"`` for grayscale.
    smooth_sigma, median_size:
        Pre-smoothing applied before thresholding (see
        :func:`src.enhancement.enhance_for_segmentation` for the rationale).
    open_radius, close_radius:
        Morphological clean-up radii.  Opening removes the cloth speckle,
        closing seals the thin bright rim around a piece.
    invert:
        Force the polarity instead of detecting it.

    Returns a boolean mask, ``True`` = piece.
    """
    img = to_float(image)
    if method == "auto":
        method = "background" if (img.ndim == 3 and img.shape[2] >= 3) else "otsu"

    if method == "background":
        score = background_distance(img, border)
    else:
        score = to_gray(img)
    if median_size and median_size > 1:
        score = median_filter(score, median_size)
    if smooth_sigma and smooth_sigma > 0:
        score = gaussian_blur(score, smooth_sigma)

    if method == "background":
        mask = score > otsu_threshold(score)
        if invert:
            mask = ~mask
    elif method == "otsu":
        mask = score > otsu_threshold(score)
        if invert is None:
            frame = np.concatenate([mask[:8, :].ravel(), mask[-8:, :].ravel(),
                                    mask[:, :8].ravel(), mask[:, -8:].ravel()])
            if frame.mean() > 0.5:      # frame is "bright" -> pieces are dark
                mask = ~mask
        elif invert:
            mask = ~mask
    elif method == "adaptive":
        mask = adaptive_threshold(score, block_size=block_size, c=c,
                                  method="mean", invert=bool(invert))
    else:
        raise ValueError(f"unknown method {method!r}")

    if open_radius and open_radius > 0:
        mask = opening(mask, disk(open_radius))
    if close_radius and close_radius > 0:
        mask = closing(mask, disk(close_radius))
    if smooth_radius and smooth_radius > 0:
        mask = majority_smooth(mask, smooth_radius)
    if fill:
        mask = fill_holes(mask)
    return mask


# ==========================================================================
# connected components (from scratch)
# ==========================================================================
def connected_components(mask: np.ndarray, connectivity: int = 8):
    """Label the connected foreground regions of a boolean mask.

    Classic **two-pass algorithm with union-find**, applied to *runs* rather
    than to individual pixels:

    1. *First pass* -- each row is decomposed into maximal horizontal runs of
       foreground pixels (found with one vectorised ``diff``).  Every run
       receives a provisional label.  Runs in consecutive rows that overlap
       are neighbours, so their labels are ``union``-ed; with
       8-connectivity the overlap test is widened by one pixel on each side
       so that diagonal contacts count.
    2. *Second pass* -- every provisional label is replaced by the root of
       its equivalence class, the roots are renumbered ``1..n``, and the runs
       are painted into the output image.

    Working on runs instead of pixels leaves only ``O(number of runs)`` work
    in Python (a few thousand for a typical mask) while producing exactly the
    same labelling as the textbook pixel-wise scan.  Union-find uses path
    compression, so the pass is effectively linear.

    Returns ``(labels, count)`` where ``labels`` is int32, background = 0.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if not mask.any():
        return labels, 0

    # ---- decompose every row into runs (vectorised) ----------------------
    padded = np.zeros((h, w + 2), dtype=bool)
    padded[:, 1:-1] = mask
    d = np.diff(padded.astype(np.int8), axis=1)
    starts_y, starts_x = np.nonzero(d == 1)
    _, ends_x = np.nonzero(d == -1)
    # run r covers columns [starts_x[r], ends_x[r]) of row starts_y[r]
    run_row = starts_y
    run_a = starts_x
    run_b = ends_x                                   # exclusive
    n_runs = run_row.size

    # index of the first run of each row (rows are already sorted)
    row_first = np.searchsorted(run_row, np.arange(h + 1))

    parent = np.arange(n_runs + 1, dtype=np.int64)   # 0 = background sentinel

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:                     # path compression
            parent[a], a = root, parent[a]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    slack = 1 if connectivity == 8 else 0
    for y in range(1, h):
        i, i_end = row_first[y - 1], row_first[y]     # previous row runs
        j, j_end = row_first[y], row_first[y + 1]     # current row runs
        while i < i_end and j < j_end:
            # runs overlap when [a1, b1) and [a2, b2) intersect (+ slack)
            if run_b[i] + slack <= run_a[j]:
                i += 1
            elif run_b[j] + slack <= run_a[i]:
                j += 1
            else:
                union(int(i) + 1, int(j) + 1)
                if run_b[i] < run_b[j]:
                    i += 1
                else:
                    j += 1

    # ---- second pass: resolve equivalences, renumber, paint --------------
    roots = np.array([find(i) for i in range(n_runs + 1)], dtype=np.int64)
    uniq = np.unique(roots[1:])
    remap = np.zeros(n_runs + 1, dtype=np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1, dtype=np.int32)
    final = remap[roots]

    for r in range(n_runs):
        labels[run_row[r], run_a[r]:run_b[r]] = final[r + 1]
    return labels, int(len(uniq))


# ==========================================================================
# component statistics / filtering
# ==========================================================================
# ==========================================================================
# distance transform and the splitting of touching pieces
# ==========================================================================
def _edt_1d(f: np.ndarray) -> np.ndarray:
    """Squared 1-D distance transform of a sampled function (lower envelope).

    Felzenszwalb & Huttenlocher's algorithm: the transform is the lower
    envelope of the parabolas ``(q - v)^2 + f(v)``, built in one left-to-right
    sweep that maintains the envelope's break points.  Linear in the length
    of the row.
    """
    n = len(f)
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    z[0], z[1] = -np.inf, np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance from every foreground pixel to the background.

    Computed as two 1-D passes (columns then rows) of :func:`_edt_1d`, which
    is exact for the Euclidean metric -- unlike the chamfer approximations
    that are usually taught alongside it.
    """
    mask = np.asarray(mask, dtype=bool)
    big = float(mask.size) * 4.0 + 4.0
    f = np.where(mask, big, 0.0)
    for x in range(f.shape[1]):
        f[:, x] = _edt_1d(f[:, x])
    for y in range(f.shape[0]):
        f[y, :] = _edt_1d(f[y, :])
    return np.sqrt(f)


def _grow_markers(markers: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Grow labelled seeds over ``region`` until they meet.

    Each iteration lets every label claim the still-unassigned neighbours of
    the pixels it already owns; a pixel reached by two labels in the same
    iteration goes to the lower label.  Growing all labels at the same speed
    means they meet on the *geodesic influence zone* boundary -- which for a
    pair of pieces joined at a narrow contact is exactly the join.
    """
    out = markers.copy()
    todo = region & (out == 0)
    while todo.any():
        grown = np.zeros_like(out)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.zeros_like(out)
            h, w = out.shape
            sy0, sy1 = max(0, -dy), min(h, h - dy)
            sx0, sx1 = max(0, -dx), min(w, w - dx)
            shifted[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
                out[sy0:sy1, sx0:sx1]
            take = todo & (shifted > 0) & ((grown == 0) | (shifted < grown))
            grown[take] = shifted[take]
        if not grown.any():
            break
        out[grown > 0] = grown[grown > 0]
        todo = region & (out == 0)
    return out


def split_touching(labels: np.ndarray, target_area: float | None = None,
                   min_ratio: float = 1.45, max_parts: int = 6,
                   n_levels: int = 60, min_core: float = 0.035):
    """Split components that merge several touching objects into one blob.

    Pieces that touch in the photograph share a connected component, and no
    threshold can separate them -- the information needed is *shape*, not
    intensity.  The classical answer is a watershed on the distance
    transform, and that is what this is:

    1. a component whose area is at least ``min_ratio`` times the expected
       piece area is assumed to hold ``round(area / target_area)`` pieces;
    2. its distance transform is smoothed and thresholded, sweeping the level
       from high to low and keeping the **highest** level at which exactly
       that many substantial cores survive -- a high level keeps only the
       "thick" middle of each piece and drops the narrow isthmus where two
       pieces touch.  The smoothing matters: the tip of a tab is a local
       maximum of the raw distance map too, and without it the sweep counts
       tabs as pieces.  Cores below ``min_core`` of the blob are discarded
       for the same reason;
    3. the surviving cores are grown back over the blob at equal speed
       (:func:`_grow_markers`), so they meet on the watershed line.

    ``target_area`` defaults to the median component area, which is the
    correct estimate whenever most pieces are already isolated.

    Returns ``(labels, n_split)``.
    """
    labels = np.asarray(labels, dtype=np.int32)
    stats = component_stats(labels)
    if not stats:
        return labels, 0
    if target_area is None:
        # the median is only meaningful once a few pieces are already isolated
        if len(stats) < 2:
            return labels, 0
        target_area = float(np.median([s.area for s in stats]))
    if target_area <= 0:
        return labels, 0

    out = labels.copy()
    next_label = int(labels.max())
    n_split = 0

    for st in stats:
        n_parts = int(round(st.area / target_area))
        if st.area < min_ratio * target_area or n_parts < 2:
            continue
        n_parts = min(n_parts, max_parts)

        y0, x0, y1, x1 = st.bbox
        pad = 2
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1, x1 = min(labels.shape[0], y1 + pad), min(labels.shape[1], x1 + pad)
        sub = labels[y0:y1, x0:x1] == st.label
        dist = distance_transform(sub)
        peak = float(dist.max())
        if peak <= 1.0:
            continue
        dist = gaussian_blur(dist, max(1.5, 0.08 * peak))
        peak = float(dist.max())

        # Sweep from high to low and keep the highest level at which exactly
        # n_parts substantial cores survive: the higher the level, the more
        # separated the cores, so the more reliable the split.
        chosen = None
        for level in np.linspace(0.95, 0.15, n_levels) * peak:
            cores, n = connected_components(dist >= level, connectivity=8)
            if n < n_parts:
                continue
            sizes = np.bincount(cores.ravel(), minlength=n + 1)
            keep = np.flatnonzero(sizes[1:] >= min_core * st.area) + 1
            if len(keep) != n_parts:
                continue
            lut = np.zeros(n + 1, dtype=np.int32)
            lut[keep] = np.arange(1, n_parts + 1, dtype=np.int32)
            chosen = lut[cores]
            break
        if chosen is None:
            continue

        grown = _grow_markers(chosen, sub)
        if int(grown.max()) < 2:
            continue

        out[y0:y1, x0:x1][sub] = 0
        for k in range(1, int(grown.max()) + 1):
            next_label += 1
            piece = sub & (grown == k)
            out[y0:y1, x0:x1][piece] = next_label
        n_split += 1

    if n_split:
        # renumber compactly
        used = np.unique(out)
        used = used[used > 0]
        lut = np.zeros(int(out.max()) + 1, dtype=np.int32)
        lut[used] = np.arange(1, len(used) + 1, dtype=np.int32)
        out = lut[out]
    return out, n_split


@dataclass
class ComponentStats:
    """Geometry of one labelled region."""
    label: int
    area: int
    bbox: tuple[int, int, int, int]      # (y0, x0, y1, x1), y1/x1 exclusive
    centroid: tuple[float, float]        # (y, x)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1]


def component_stats(labels: np.ndarray) -> list[ComponentStats]:
    """Area, bounding box and centroid of every non-zero label."""
    n = int(labels.max())
    if n == 0:
        return []
    flat = labels.ravel()
    ys, xs = np.divmod(np.arange(flat.size), labels.shape[1])
    area = np.bincount(flat, minlength=n + 1)
    sum_y = np.bincount(flat, weights=ys, minlength=n + 1)
    sum_x = np.bincount(flat, weights=xs, minlength=n + 1)

    min_y = np.full(n + 1, labels.shape[0], np.int64)
    max_y = np.full(n + 1, -1, np.int64)
    min_x = np.full(n + 1, labels.shape[1], np.int64)
    max_x = np.full(n + 1, -1, np.int64)
    np.minimum.at(min_y, flat, ys)
    np.maximum.at(max_y, flat, ys)
    np.minimum.at(min_x, flat, xs)
    np.maximum.at(max_x, flat, xs)

    out = []
    for lab in range(1, n + 1):
        if area[lab] == 0:
            continue
        out.append(ComponentStats(
            label=lab,
            area=int(area[lab]),
            bbox=(int(min_y[lab]), int(min_x[lab]),
                  int(max_y[lab]) + 1, int(max_x[lab]) + 1),
            centroid=(float(sum_y[lab] / area[lab]),
                      float(sum_x[lab] / area[lab])),
        ))
    return out


def filter_components(labels: np.ndarray, min_area: int = 0,
                      max_area: int | None = None,
                      min_area_ratio: float | None = 0.25,
                      max_area_ratio: float | None = None,
                      max_aspect: float | None = None,
                      relabel: bool = True):
    """Keep only components that look like puzzle pieces.

    ``min_area_ratio`` and ``max_area_ratio`` are expressed relative to the
    **median** component area, which is the robust way to reject clutter
    (bolts, cable ties, cloth folds) in the dataset photographs without
    hard-coding a pixel count.  The upper bound is useful after
    :func:`split_touching`, where anything still much larger than a piece is
    a blob the splitter could not resolve.  Returns ``(labels, stats)``.
    """
    stats = component_stats(labels)
    if not stats:
        return labels, stats

    areas = np.array([s.area for s in stats], dtype=np.float64)
    median = float(np.median(areas)) if areas.size else 0.0
    lo = float(min_area)
    if min_area_ratio is not None and areas.size:
        lo = max(lo, median * float(min_area_ratio))
    hi = float(max_area) if max_area is not None else np.inf
    if max_area_ratio is not None and areas.size:
        hi = min(hi, median * float(max_area_ratio))

    keep_ids = []
    for s, a in zip(stats, areas):
        if a < lo:
            continue
        if a > hi:
            continue
        if max_aspect is not None:
            long_side = max(s.height, s.width)
            short_side = max(min(s.height, s.width), 1)
            if long_side / short_side > max_aspect:
                continue
        keep_ids.append(s.label)

    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for new, old in enumerate(keep_ids, start=1):
        lut[old] = new if relabel else old
    new_labels = lut[labels]
    return new_labels.astype(np.int32), component_stats(new_labels)
