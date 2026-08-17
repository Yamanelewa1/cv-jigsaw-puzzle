"""Milestone 1 / task 3c-d -- Boundary tracing and piece extraction.

From scratch:

* :func:`trace_boundary` -- Moore-neighbour contour tracing with Jacob's
  stopping criterion,
* :func:`convex_hull` (Andrew's monotone chain) and :func:`min_area_rect`
  (rotating-calipers style search over hull edges),
* :func:`rotate_image` / :func:`rotate_mask` -- bilinear inverse warping,
* :func:`extract_pieces` -- crops every labelled region to its bounding box
  and stores its mask, contour, centroid and normalised orientation in a
  :class:`Piece`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .enhancement import to_float
from .segmentation import connected_components, component_stats

__all__ = [
    "trace_boundary",
    "boundary_mask",
    "polygon_area",
    "polygon_perimeter",
    "resample_contour",
    "convex_hull",
    "min_area_rect",
    "rect_corners",
    "rotate_image",
    "rotate_mask",
    "Piece",
    "extract_pieces",
    "normalize_piece",
]

# Moore neighbourhood in clockwise order starting at "west".
_MOORE = np.array([(0, -1), (-1, -1), (-1, 0), (-1, 1),
                   (0, 1), (1, 1), (1, 0), (1, -1)], dtype=np.int64)


# ==========================================================================
# boundary tracing
# ==========================================================================
def trace_boundary(mask: np.ndarray, start: tuple[int, int] | None = None,
                   max_steps: int | None = None) -> np.ndarray:
    """Trace the outer boundary of a binary object.

    Moore-neighbour tracing: from the current boundary pixel we walk around
    its 8-neighbourhood, starting from the neighbour we came from, until the
    next foreground pixel is found; that pixel becomes the new boundary
    point.  Jacob's stopping criterion (stop when the *start* pixel is
    entered a second time *from the same direction*) is used instead of the
    naive "stop when you reach the start", which would truncate contours that
    legitimately pass through the start pixel twice -- common on the thin
    neck of a jigsaw tab.

    Returns an ``(N, 2)`` int array of ``(y, x)`` points in clockwise order
    (image coordinates), or an empty array if the mask is empty.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros((0, 2), dtype=np.int64)

    if start is None:
        ys, xs = np.nonzero(mask)
        i = int(np.argmin(ys.astype(np.int64) * mask.shape[1] + xs))
        start = (int(ys[i]), int(xs[i]))
    h, w = mask.shape
    if max_steps is None:
        max_steps = 8 * int(mask.sum()) + 16

    moore = [(int(d[0]), int(d[1])) for d in _MOORE]
    # direction index -> lookup from an offset
    dir_of = {d: i for i, d in enumerate(moore)}

    def is_fg(p):
        return 0 <= p[0] < h and 0 <= p[1] < w and mask[p[0], p[1]]

    contour = [start]
    b = start
    # The pixel we "came from": the west neighbour of the top-left-most
    # foreground pixel is background by construction.
    last_bg = (start[0], start[1] - 1)
    first_step: tuple | None = None

    for _ in range(max_steps):
        off = (last_bg[0] - b[0], last_bg[1] - b[1])
        d0 = dir_of.get(off, 0)
        c = None
        for k in range(1, 9):
            d = (d0 + k) % 8
            cand = (b[0] + moore[d][0], b[1] + moore[d][1])
            if is_fg(cand):
                c = cand
                break
            last_bg = cand          # remember the last background examined
        if c is None:               # isolated pixel
            break
        step = (b, c)
        if first_step is None:
            first_step = step
        elif step == first_step:
            # Jacob's criterion: the start pixel has been entered a second
            # time from the same direction -> the contour is closed.
            break
        contour.append(c)
        b = c

    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    return np.array(contour, dtype=np.int64)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Boolean image of the traced boundary (handy for visualisations)."""
    c = trace_boundary(mask)
    out = np.zeros_like(mask, dtype=bool)
    if c.size:
        out[c[:, 0], c[:, 1]] = True
    return out


# ==========================================================================
# polygon helpers
# ==========================================================================
def polygon_area(points: np.ndarray) -> float:
    """Absolute polygon area by the shoelace formula."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    y, x = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points: np.ndarray, closed: bool = True) -> float:
    """Total length of a polyline / closed polygon."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    d = np.diff(p, axis=0)
    total = float(np.hypot(d[:, 0], d[:, 1]).sum())
    if closed:
        total += float(np.hypot(*(p[0] - p[-1])))
    return total


def resample_contour(points: np.ndarray, n: int, closed: bool = False) -> np.ndarray:
    """Resample a polyline to ``n`` points equally spaced along arc length.

    This makes contours of different pixel lengths directly comparable, which
    is what the edge-matching stage needs.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return np.zeros((n, 2), dtype=np.float64)
    if len(p) == 1:
        return np.repeat(p, n, axis=0)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    y = np.interp(target, s, p[:, 0])
    x = np.interp(target, s, p[:, 1])
    return np.stack([y, x], axis=1)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull via Andrew's monotone chain, ``O(n log n)``.

    Points are given/returned as ``(y, x)``; the hull is counter-clockwise in
    the ``(x, y)`` plane and does not repeat the first point.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3:
        return p.copy()
    order = np.lexsort((p[:, 0], p[:, 1]))    # sort by x then y
    pts = p[order]

    def cross(o, a, b):
        return ((a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1]))

    lower: list = []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def min_area_rect(points: np.ndarray):
    """Minimum-area enclosing rectangle of a point set.

    Uses the rotating-calipers property that an optimal rectangle has one
    side collinear with a hull edge: we test every hull edge orientation,
    rotate the hull into that frame, and keep the orientation with the
    smallest axis-aligned bounding box.

    Returns ``(center_yx, (height, width), angle)`` where ``angle`` is the
    rotation of the rectangle's "width" axis in radians (counter-clockwise in
    image coordinates, i.e. ``+x`` is right and ``+y`` is down).
    """
    hull = convex_hull(np.asarray(points, dtype=np.float64))
    if len(hull) < 2:
        c = hull.mean(axis=0) if len(hull) else np.zeros(2)
        return (float(c[0]), float(c[1])), (0.0, 0.0), 0.0

    best = None
    n = len(hull)
    for i in range(n):
        p0, p1 = hull[i], hull[(i + 1) % n]
        dy, dx = p1[0] - p0[0], p1[1] - p0[1]
        if abs(dy) < 1e-12 and abs(dx) < 1e-12:
            continue
        ang = float(np.arctan2(dy, dx))
        c, s = np.cos(-ang), np.sin(-ang)
        # rotate (x, y) by -ang
        xr = hull[:, 1] * c - hull[:, 0] * s
        yr = hull[:, 1] * s + hull[:, 0] * c
        w = float(xr.max() - xr.min())
        h = float(yr.max() - yr.min())
        area = w * h
        if best is None or area < best[0]:
            cx = 0.5 * (xr.max() + xr.min())
            cy = 0.5 * (yr.max() + yr.min())
            # rotate the centre back
            ci, si = np.cos(ang), np.sin(ang)
            center_x = cx * ci - cy * si
            center_y = cx * si + cy * ci
            best = (area, (float(center_y), float(center_x)), (h, w), ang)

    if best is None:
        c = hull.mean(axis=0)
        return (float(c[0]), float(c[1])), (0.0, 0.0), 0.0
    return best[1], best[2], best[3]


def rect_corners(center: Sequence[float], size: Sequence[float],
                 angle: float) -> np.ndarray:
    """The four corners ``(y, x)`` of a rotated rectangle, clockwise."""
    cy, cx = float(center[0]), float(center[1])
    h, w = float(size[0]), float(size[1])
    local = np.array([[-h / 2, -w / 2], [-h / 2, w / 2],
                      [h / 2, w / 2], [h / 2, -w / 2]], dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    ys = local[:, 0] * c + local[:, 1] * s
    xs = -local[:, 0] * s + local[:, 1] * c
    return np.stack([ys + cy, xs + cx], axis=1)


# ==========================================================================
# geometric transforms (from scratch, bilinear)
# ==========================================================================
def rotate_image(image: np.ndarray, angle: float,
                 center: Sequence[float] | None = None,
                 output_shape: tuple[int, int] | None = None,
                 fill: float = 0.0, order: int = 1) -> np.ndarray:
    """Rotate an image by ``angle`` radians about ``center`` (``(y, x)``).

    Inverse mapping: every output pixel is mapped back into the input and
    sampled with bilinear interpolation (``order=1``) or nearest neighbour
    (``order=0``).  Positive angles rotate counter-clockwise on screen.
    """
    img = to_float(image)
    single = img.ndim == 2
    if single:
        img = img[:, :, None]
    h, w = img.shape[:2]
    oh, ow = output_shape if output_shape is not None else (h, w)
    if center is None:
        center = ((h - 1) / 2.0, (w - 1) / 2.0)
    ocy, ocx = (oh - 1) / 2.0, (ow - 1) / 2.0

    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float64)
    dy = yy - ocy
    dx = xx - ocx
    c, s = np.cos(angle), np.sin(angle)
    # inverse rotation (screen-CCW for positive angle with y pointing down)
    src_y = center[0] + (-dx * s + dy * c)
    src_x = center[1] + (dx * c + dy * s)

    if order == 0:
        yi = np.rint(src_y).astype(np.int64)
        xi = np.rint(src_x).astype(np.int64)
        valid = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
        out = np.full((oh, ow, img.shape[2]), float(fill))
        out[valid] = img[yi[valid], xi[valid]]
    else:
        y0 = np.floor(src_y).astype(np.int64)
        x0 = np.floor(src_x).astype(np.int64)
        wy = src_y - y0
        wx = src_x - x0
        out = np.full((oh, ow, img.shape[2]), float(fill))
        valid = (y0 >= 0) & (y0 + 1 < h) & (x0 >= 0) & (x0 + 1 < w)
        y0c, x0c = np.clip(y0, 0, h - 2), np.clip(x0, 0, w - 2)
        wy3, wx3 = wy[..., None], wx[..., None]
        top = img[y0c, x0c] * (1 - wx3) + img[y0c, x0c + 1] * wx3
        bot = img[y0c + 1, x0c] * (1 - wx3) + img[y0c + 1, x0c + 1] * wx3
        blended = top * (1 - wy3) + bot * wy3
        out[valid] = blended[valid]

    return out[:, :, 0] if single else out


def rotate_mask(mask: np.ndarray, angle: float,
                center: Sequence[float] | None = None,
                output_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Rotate a boolean mask (bilinear + 0.5 threshold keeps edges smooth)."""
    r = rotate_image(mask.astype(np.float64), angle, center, output_shape,
                     fill=0.0, order=1)
    return r > 0.5


# ==========================================================================
# piece extraction
# ==========================================================================
@dataclass
class Piece:
    """One segmented puzzle piece.

    Attributes
    ----------
    index:
        Position in the extracted list (0-based).
    label:
        Label value in the connected-component image.
    bbox:
        ``(y0, x0, y1, x1)`` of the piece in the source image.
    image:
        RGB crop (float, ``[0, 1]``), background pixels zeroed.
    mask:
        Boolean crop, ``True`` = piece.
    contour:
        ``(N, 2)`` traced boundary in crop coordinates.
    centroid:
        Centre of mass in crop coordinates.
    angle:
        Normalised orientation: the rotation (radians) of the piece's
        minimum-area rectangle.  Rotating the crop by ``-angle`` makes the
        piece body axis aligned.
    rect_size:
        ``(height, width)`` of that rectangle, i.e. the piece's body size.
    """
    index: int
    label: int
    bbox: tuple[int, int, int, int]
    image: np.ndarray
    mask: np.ndarray
    contour: np.ndarray
    centroid: tuple[float, float]
    angle: float
    rect_size: tuple[float, float]
    area: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.mask.shape[0]

    @property
    def width(self) -> int:
        return self.mask.shape[1]


def extract_pieces(image: np.ndarray, labels: np.ndarray,
                   pad_px: int = 4, stats=None) -> list[Piece]:
    """Crop every labelled component into a :class:`Piece`.

    ``pad_px`` pixels of margin are kept around the bounding box so that the
    traced contour never touches the crop border (which would break the
    Moore-neighbour walk and the later corner search).
    """
    img = to_float(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    if stats is None:
        stats = component_stats(labels)

    pieces: list[Piece] = []
    H, W = labels.shape
    for i, st in enumerate(stats):
        y0, x0, y1, x1 = st.bbox
        y0 = max(0, y0 - pad_px)
        x0 = max(0, x0 - pad_px)
        y1 = min(H, y1 + pad_px)
        x1 = min(W, x1 + pad_px)
        m = labels[y0:y1, x0:x1] == st.label
        crop = img[y0:y1, x0:x1].copy()
        crop[~m] = 0.0
        contour = trace_boundary(m)
        if len(contour) < 8:
            continue
        center, size, angle = min_area_rect(contour.astype(np.float64))
        ys, xs = np.nonzero(m)
        pieces.append(Piece(
            index=len(pieces),
            label=int(st.label),
            bbox=(y0, x0, y1, x1),
            image=crop,
            mask=m,
            contour=contour,
            centroid=(float(ys.mean()), float(xs.mean())),
            angle=float(angle),
            rect_size=(float(size[0]), float(size[1])),
            area=int(m.sum()),
        ))
    return pieces


def normalize_piece(piece: Piece, pad_px: int = 6) -> Piece:
    """Return a copy of ``piece`` rotated so its body is axis aligned.

    The rotation angle is the piece's :attr:`Piece.angle`; the output canvas
    is large enough to contain the rotated piece plus ``pad_px`` margin.  The
    contour is re-traced on the rotated mask so it stays consistent with the
    pixels.
    """
    h, w = piece.mask.shape
    diag = int(np.ceil(np.hypot(h, w))) + 2 * pad_px
    center = ((h - 1) / 2.0, (w - 1) / 2.0)
    rot_img = rotate_image(piece.image, -piece.angle, center, (diag, diag))
    rot_mask = rotate_mask(piece.mask, -piece.angle, center, (diag, diag))
    if not rot_mask.any():
        return piece

    ys, xs = np.nonzero(rot_mask)
    y0, y1 = max(0, ys.min() - pad_px), min(diag, ys.max() + pad_px + 1)
    x0, x1 = max(0, xs.min() - pad_px), min(diag, xs.max() + pad_px + 1)
    m = rot_mask[y0:y1, x0:x1]
    im = rot_img[y0:y1, x0:x1]
    im[~m] = 0.0
    contour = trace_boundary(m)
    center2, size2, angle2 = min_area_rect(contour.astype(np.float64))
    ys2, xs2 = np.nonzero(m)
    return Piece(
        index=piece.index,
        label=piece.label,
        bbox=piece.bbox,
        image=im,
        mask=m,
        contour=contour,
        centroid=(float(ys2.mean()), float(xs2.mean())),
        angle=float(angle2),
        rect_size=(float(size2[0]), float(size2[1])),
        area=int(m.sum()),
        meta={**piece.meta, "source_angle": piece.angle},
    )
