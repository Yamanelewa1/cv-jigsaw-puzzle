"""Milestone 1 / task 4 -- Piece-edge description.

For every extracted piece this module

1. locates the **four corners** on the traced contour,
2. splits the boundary into the **four sides** between them,
3. classifies each side as a **tab** (protrusion), a **blank** (indentation)
   or a **flat** (straight border edge) from its geometry, and
4. samples a **strip of colour** just inside each side, the photometric
   signature used by the matcher.

Corner detection
----------------
Getting the four corners right is the single most important step in this
module: every side profile and every colour strip is measured *between* two
corners, so a corner that is a few percent off shifts a whole descriptor and
the matcher pays for it.  Three strategies are tried in order.

1. **Body-edge model** (:func:`find_corners_by_edges`, the one that normally
   fires).  A jigsaw piece is a rectangle plus bumps, so its four straight
   body edges are parallel or perpendicular to each other.  Their common
   direction is recovered modulo 90 degrees by
   :func:`dominant_orientation`, the contour is rotated into that frame, each
   edge is located as the *mode* of the perpendicular coordinate of the
   boundary points whose tangent is parallel to it, and the corners are the
   four line intersections.
2. **Curvature search** (fallback).  The contour is resampled and the turning
   angle between the chords to the neighbours ``+-k`` samples away is
   measured; the convex local maxima are corner candidates and the best four
   are chosen by a small combinatorial search rewarding a large enclosed
   quadrilateral with near-right angles.
3. **Minimum-area rectangle** (last resort), snapped onto the contour.

The minimum-area rectangle is deliberately *not* the first choice: tabs push
its sides outwards, so its corners can sit far from the real ones.  Nor is
curvature alone reliable -- the tip of a tab of radius ``rho`` turns through
90 degrees over an arc of only ``1.6 rho`` pixels, which is *sharper* than
the piece's own corner at any scale large enough to be noise-free, so a
curvature search can and does mistake tab tips for corners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .contour_extraction import (Piece, min_area_rect, polygon_area,
                                 rect_corners, resample_contour, rotate_mask)
from .enhancement import integral_image, to_float

__all__ = [
    "Side",
    "PieceDescription",
    "smooth_contour",
    "turning_angles",
    "dominant_orientation",
    "find_corners_by_edges",
    "find_corners",
    "split_sides",
    "classify_side",
    "sample_color_strip",
    "fit_body_rectangle",
    "describe_piece",
    "calibrate_side_types",
    "refine_descriptions",
    "describe_pieces",
    "SIDE_TAB", "SIDE_BLANK", "SIDE_FLAT",
    "DIRECTIONS",
]

SIDE_FLAT = "flat"
SIDE_TAB = "tab"
SIDE_BLANK = "blank"

#: Grid directions in clockwise order, matching the clockwise side order.
DIRECTIONS = ("N", "E", "S", "W")
#: ``(dr, dc)`` step for each direction in :data:`DIRECTIONS`.
DIRECTION_STEPS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
#: The direction a side faces when it is matched against another side.
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}


@dataclass
class Side:
    """One of the four sides of a piece.

    Attributes
    ----------
    index:
        0..3, clockwise, starting at the piece's first corner.
    type:
        ``"tab"``, ``"blank"`` or ``"flat"``.
    points:
        ``(M, 2)`` resampled ``(y, x)`` contour points from corner ``index``
        to corner ``index + 1``.
    profile:
        ``(M,)`` signed perpendicular deviation from the corner-to-corner
        chord, **normalised by the chord length**, sampled at equal arc
        length.  Positive = outward (away from the piece centre).
    colors:
        ``(M, D, 3)`` colours sampled at ``D`` depths just inside the side.
    length:
        Chord length in pixels.
    corners:
        The two endpoint coordinates ``((y0, x0), (y1, x1))``.
    amplitude:
        Peak absolute deviation, normalised by the chord length.
    """
    index: int
    type: str
    points: np.ndarray
    profile: np.ndarray
    colors: np.ndarray
    length: float
    corners: tuple
    amplitude: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.type == SIDE_FLAT


@dataclass
class PieceDescription:
    """A piece plus its four described sides."""
    piece: Piece
    corners: np.ndarray                     # (4, 2) clockwise, (y, x)
    sides: list[Side]
    centroid: tuple[float, float]
    body_size: tuple[float, float]          # (height, width) of the body
    meta: dict = field(default_factory=dict)

    @property
    def index(self) -> int:
        return self.piece.index

    @property
    def n_flats(self) -> int:
        return sum(1 for s in self.sides if s.is_flat)

    @property
    def is_corner_piece(self) -> bool:
        """Exactly two flats that are *adjacent* -> a grid corner."""
        flags = [s.is_flat for s in self.sides]
        if sum(flags) != 2:
            return False
        return any(flags[i] and flags[(i + 1) % 4] for i in range(4))

    @property
    def is_border_piece(self) -> bool:
        return self.n_flats >= 1

    def side_facing(self, rotation: int, direction: str) -> Side:
        """The side pointing ``direction`` when the piece is in ``rotation``.

        ``rotation`` is the index of the side that faces North.  Because the
        sides are stored clockwise and ``N, E, S, W`` is also clockwise, the
        side facing direction ``d`` is simply ``sides[(rotation + d) % 4]``.
        """
        d = DIRECTIONS.index(direction)
        return self.sides[(rotation + d) % 4]


# ==========================================================================
# contour signal processing
# ==========================================================================
def smooth_contour(contour: np.ndarray, window: int = 5) -> np.ndarray:
    """Circular moving average along the contour.

    Removes the one-pixel staircase produced by rasterisation, which would
    otherwise dominate the curvature estimate.
    """
    c = np.asarray(contour, dtype=np.float64)
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(c) < window:
        return c
    k = np.ones(window) / window
    pad = window // 2
    ext = np.vstack([c[-pad:], c, c[:pad]])
    ys = np.convolve(ext[:, 0], k, mode="valid")
    xs = np.convolve(ext[:, 1], k, mode="valid")
    return np.stack([ys, xs], axis=1)[:len(c)]


def turning_angles(contour: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Turning angle and its sign at every contour point.

    For point ``i`` the incoming chord is ``p_i - p_{i-k}`` and the outgoing
    chord is ``p_{i+k} - p_i``.  The angle between them is the amount the
    boundary turns over that span; the sign of the 2-D cross product tells
    whether the turn is convex or concave.

    Returns ``(angle, cross)`` with ``angle`` in radians (0 = straight,
    ``pi/2`` = right-angle turn).
    """
    c = np.asarray(contour, dtype=np.float64)
    n = len(c)
    prev = np.roll(c, k, axis=0)
    nxt = np.roll(c, -k, axis=0)
    v1 = c - prev
    v2 = nxt - c
    n1 = np.hypot(v1[:, 0], v1[:, 1]) + 1e-12
    n2 = np.hypot(v2[:, 0], v2[:, 1]) + 1e-12
    dot = (v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]) / (n1 * n2)
    cross = (v1[:, 1] * v2[:, 0] - v1[:, 0] * v2[:, 1]) / (n1 * n2)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return angle, cross


def _local_maxima(signal: np.ndarray, min_distance: int) -> np.ndarray:
    """Indices of circular local maxima, greedily thinned by distance."""
    n = len(signal)
    order = np.argsort(signal)[::-1]
    taken: list[int] = []
    for i in order:
        if signal[i] <= 0:
            break
        if all(min(abs(i - j), n - abs(i - j)) >= min_distance for j in taken):
            taken.append(int(i))
    return np.array(sorted(taken), dtype=np.int64)


def dominant_orientation(points: np.ndarray) -> float:
    """Orientation of a piece's straight body edges, modulo 90 degrees.

    The four body edges of a jigsaw piece are mutually parallel or
    perpendicular, so their tangent directions all coincide *modulo* 90
    degrees.  Mapping every tangent angle ``phi`` to ``exp(4 i phi)`` folds
    those four directions onto the same complex phasor while a circular tab
    arc, which sweeps all directions uniformly, cancels itself out.  The
    argument of the sum therefore recovers the edge direction robustly:

        ``theta = arg( sum_i exp(4 i phi_i) ) / 4``

    Returned in radians in ``[0, pi/2)``.
    """
    p = np.asarray(points, dtype=np.float64)
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    phi = np.arctan2(t[:, 0], t[:, 1])
    z = np.exp(4j * phi).sum()
    if abs(z) < 1e-12:
        return 0.0
    return float(np.angle(z) / 4.0) % (np.pi / 2)


def _edge_pair(coord: np.ndarray, weight: np.ndarray,
               min_separation: float = 0.35) -> tuple[float, float]:
    """Positions of the two opposite body edges along one axis.

    ``coord`` holds the perpendicular coordinate of every boundary point
    whose tangent runs along the edge, and ``weight`` how parallel that
    tangent is.  The flat part of each edge dumps a tall, one-pixel-wide
    spike into the histogram, whereas a tab arc spreads its mass over its
    whole radius -- so the two **modes** find the edges even when the arcs
    contribute more points in total, which a median would not.

    Both edges are taken from the *same* histogram, as its two strongest
    peaks separated by at least ``min_separation`` of the coordinate range.
    Splitting the points at the centroid instead (one half per edge) is what
    an earlier version did, and it fails on pieces with a large tab, because
    the tab drags the centroid across the true midline.

    Returns ``(low, high)``, or ``(nan, nan)`` if two peaks cannot be found.
    """
    if coord.size == 0:
        return float("nan"), float("nan")
    lo, hi = float(coord.min()), float(coord.max())
    if hi - lo < 2:
        return lo, hi
    nb = max(8, int(np.ceil(hi - lo)) + 1)
    hist, edges = np.histogram(coord, bins=nb, range=(lo, hi + 1e-9),
                               weights=weight)
    if hist.sum() <= 0:
        return float("nan"), float("nan")
    centers = 0.5 * (edges[:-1] + edges[1:])
    # smooth by 3 bins so a spike split across two bins is not lost
    sm = np.convolve(hist, np.ones(3), mode="same")

    def refine(peak: int) -> float:
        a, b = max(0, peak - 1), min(nb, peak + 2)
        w = hist[a:b]
        if w.sum() <= 0:
            return float(centers[peak])
        return float((w * centers[a:b]).sum() / w.sum())

    first = int(np.argmax(sm))
    guard = max(2.0, min_separation * (hi - lo))
    masked = sm.copy()
    masked[np.abs(centers - centers[first]) < guard] = -1.0
    if masked.max() <= 0:
        return float("nan"), float("nan")
    second = int(np.argmax(masked))
    a, b = refine(first), refine(second)
    return (a, b) if a <= b else (b, a)


def find_corners_by_edges(contour: np.ndarray, n_samples: int = 720,
                          tangent_tol_deg: float = 22.0):
    """Corner detection by fitting the four straight body edges.

    1. the dominant orientation ``theta`` is estimated with
       :func:`dominant_orientation` and the contour is rotated into that
       frame, where the body edges become axis aligned;
    2. each edge is located as the mode of the perpendicular coordinate of
       the boundary points whose tangent is parallel to it (and which lie on
       the correct side of the centroid);
    3. the corners are the four intersections of those lines, rotated back.

    Unlike a curvature search this cannot be fooled by the tip of a tab: a
    tab of radius ``rho`` turns through 90 degrees over an arc of only
    ``1.6 rho`` pixels, which is *sharper* than the piece's own corner at any
    scale large enough to be noise-free.

    Returns ``(corners, diagnostics)`` or ``(None, diagnostics)`` if the four
    lines do not form a plausible rectangle.
    """
    res = resample_contour(np.asarray(contour, dtype=np.float64), n_samples,
                           closed=True)
    theta = dominant_orientation(res)
    c, s = np.cos(-theta), np.sin(-theta)
    # rotate (y, x) by -theta
    ry = res[:, 0] * c + res[:, 1] * s
    rx = -res[:, 0] * s + res[:, 1] * c

    t = np.stack([np.roll(ry, -1) - np.roll(ry, 1),
                  np.roll(rx, -1) - np.roll(rx, 1)], axis=1)
    tn = np.hypot(t[:, 0], t[:, 1]) + 1e-12
    ty, tx = t[:, 0] / tn, t[:, 1] / tn
    tol = np.cos(np.deg2rad(90.0 - tangent_tol_deg))

    horizontal = np.abs(ty) <= tol          # tangent runs along x
    vertical = np.abs(tx) <= tol            # tangent runs along y

    top, bottom = _edge_pair(ry[horizontal], np.abs(tx[horizontal]))
    left, right = _edge_pair(rx[vertical], np.abs(ty[vertical]))

    diag = {"theta": theta, "top": top, "bottom": bottom,
            "left": left, "right": right}
    if not all(np.isfinite(v) for v in (top, bottom, left, right)):
        return None, diag
    h, w = bottom - top, right - left
    if h <= 1 or w <= 1:
        return None, diag
    if min(h, w) < 0.45 * max(h, w):
        return None, diag                  # not a plausible piece body

    local = np.array([[top, left], [top, right], [bottom, right], [bottom, left]])
    ci, si = np.cos(theta), np.sin(theta)
    corners = np.stack([local[:, 0] * ci + local[:, 1] * si,
                        -local[:, 0] * si + local[:, 1] * ci], axis=1)

    # Snap onto the real contour so the side split happens at a real point.
    # If a predicted corner is not actually *on* the boundary, the four lines
    # do not describe this piece and the model must be rejected -- that is a
    # far better outcome than describing the piece from a wrong quadrilateral.
    tol = 0.08 * min(h, w)
    out, order = [], []
    for corner in corners:
        d = np.hypot(res[:, 0] - corner[0], res[:, 1] - corner[1])
        i = int(np.argmin(d))
        if d[i] > tol:
            diag["snap_error"] = float(d[i])
            return None, diag
        out.append(res[i])
        order.append(i)
    if len(set(order)) < 4:
        return None, diag

    # keep the clockwise order of the contour
    out = np.roll(np.array(out, dtype=np.float64),
                  -int(np.argmin(order)), axis=0)
    return out, diag


def _order_on_contour(corners: np.ndarray, resampled: np.ndarray,
                      tol: float | None = None):
    """Snap corners onto the contour and put them in contour (clockwise) order.

    Returns ``None`` if a corner is further than ``tol`` from the boundary --
    a predicted corner that is not actually *on* the piece means the model
    does not describe this piece and must be rejected.
    """
    out, order = [], []
    for corner in corners:
        d = np.hypot(resampled[:, 0] - corner[0], resampled[:, 1] - corner[1])
        i = int(np.argmin(d))
        if tol is not None and d[i] > tol:
            return None
        out.append(resampled[i])
        order.append(i)
    if len(set(order)) < 4:
        return None
    return np.roll(np.array(out, dtype=np.float64),
                   -int(np.argmin(order)), axis=0)


def fit_body_rectangle(mask: np.ndarray, theta: float, size, contour=None,
                       n_samples: int = 720):
    """Fit a rectangle of *known* size to a piece, at orientation ``theta``.

    Every piece of one puzzle is cut from the same grid, so they all share a
    body size.  When a piece's own corner estimate is an outlier, that shared
    size is the strongest information available, and this routine exploits
    it: the mask is rotated so the body becomes axis aligned, and the
    ``size``-shaped window covering the most piece area is found.  Because a
    tab only adds area outside the body while a blank only removes area
    inside it, the maximum-coverage window sits on the body.

    The search is exhaustive but costs one pass: box sums for *every*
    candidate position come from a single integral image.

    Returns ``(4, 2)`` corners in mask coordinates, clockwise, or ``None``.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    bh, bw = int(round(size[0])), int(round(size[1]))
    if bh < 3 or bw < 3:
        return None

    diag = int(np.ceil(np.hypot(h, w))) + 4
    if bh >= diag or bw >= diag:
        return None
    center = ((h - 1) / 2.0, (w - 1) / 2.0)
    rot = rotate_mask(mask, -theta, center, (diag, diag))

    integ = integral_image(rot.astype(np.float64))
    sums = (integ[bh:, bw:] - integ[:-bh, bw:]
            - integ[bh:, :-bw] + integ[:-bh, :-bw])
    if sums.size == 0:
        return None
    y0, x0 = np.unravel_index(int(np.argmax(sums)), sums.shape)
    coverage = float(sums[y0, x0]) / float(bh * bw)

    local = np.array([[y0, x0], [y0, x0 + bw],
                      [y0 + bh, x0 + bw], [y0 + bh, x0]], dtype=np.float64)

    # invert exactly the mapping rotate_image applies (angle = -theta)
    ocy = ocx = (diag - 1) / 2.0
    c, s = np.cos(-theta), np.sin(-theta)
    dy, dx = local[:, 0] - ocy, local[:, 1] - ocx
    ys = center[0] + (-dx * s + dy * c)
    xs = center[1] + (dx * c + dy * s)
    corners = np.stack([ys, xs], axis=1)

    if contour is not None:
        res = resample_contour(np.asarray(contour, dtype=np.float64),
                               n_samples, closed=True)
        ordered = _order_on_contour(corners, res, tol=0.25 * min(bh, bw))
        if ordered is None:
            return None
        corners = ordered
    return corners, coverage


def _corners_plausible(corners: np.ndarray, min_side_ratio: float = 0.7,
                       angle_tol_deg: float = 18.0) -> bool:
    """Reject corner sets that are not a convex, near-right-angled quad.

    A jigsaw piece body is a rectangle, so the accepted band is deliberately
    tight: a candidate whose corner angles stray more than ``angle_tol_deg``
    from 90 degrees, or whose shortest side is under ``min_side_ratio`` of
    its longest, is a mis-detection and it is better to fall through to the
    next strategy than to describe the piece from it.
    """
    if corners is None or len(corners) != 4:
        return False
    sides = [float(np.hypot(*(corners[(i + 1) % 4] - corners[i]))) for i in range(4)]
    if min(sides) < 1e-6 or min(sides) < min_side_ratio * max(sides):
        return False
    for i in range(4):
        a = corners[(i - 1) % 4] - corners[i]
        b = corners[(i + 1) % 4] - corners[i]
        na, nb = np.hypot(*a) + 1e-12, np.hypot(*b) + 1e-12
        ang = np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1)))
        if abs(ang - 90.0) > angle_tol_deg:
            return False
    return True


def find_corners(contour: np.ndarray, n_samples: int = 720,
                 n_candidates: int = 14, min_angle_deg: float = 35.0,
                 fallback_mask: np.ndarray | None = None) -> np.ndarray:
    """Locate the four corners of a jigsaw piece on its contour.

    Tries :func:`find_corners_by_edges` first (the body-edge model), then the
    curvature search described in the module docstring, and finally the
    minimum-area rectangle.  Returns a ``(4, 2)`` array of ``(y, x)`` corner
    coordinates in the same clockwise order as the contour.
    """
    contour = np.asarray(contour, dtype=np.float64)
    if len(contour) < 8:
        center, size, angle = min_area_rect(contour)
        return rect_corners(center, size, angle)

    by_edges, _ = find_corners_by_edges(contour, n_samples)
    if _corners_plausible(by_edges):
        return by_edges

    res = resample_contour(contour, n_samples, closed=True)
    res = smooth_contour(res, window=max(3, n_samples // 120 * 2 + 1))
    k = max(4, n_samples // 22)
    angle, cross = turning_angles(res, k)

    # A clockwise contour in image coordinates turns with a consistent sign at
    # convex points; take whichever sign is in the majority as "convex".
    sign = 1.0 if float(np.sum(cross)) >= 0 else -1.0
    convex = angle * (sign * cross > 0)

    cand = _local_maxima(convex, min_distance=max(6, n_samples // 12))
    cand = cand[convex[cand] > np.deg2rad(min_angle_deg)]
    if len(cand) > n_candidates:
        cand = cand[np.argsort(convex[cand])[::-1][:n_candidates]]
        cand = np.sort(cand)

    best = None
    if len(cand) >= 4:
        for quad in combinations(cand.tolist(), 4):
            pts = res[list(quad)]
            area = polygon_area(pts)
            if area <= 0:
                continue
            # interior angles of the quadrilateral
            ang = []
            ok = True
            for i in range(4):
                a = pts[(i - 1) % 4] - pts[i]
                b = pts[(i + 1) % 4] - pts[i]
                na = np.hypot(*a) + 1e-12
                nb = np.hypot(*b) + 1e-12
                th = np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))
                ang.append(th)
                if not (np.deg2rad(55) < th < np.deg2rad(125)):
                    ok = False
                    break
            if not ok:
                continue
            sides = [np.hypot(*(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
            if min(sides) < 0.35 * max(sides):
                continue
            rect_penalty = float(np.sum(np.abs(np.array(ang) - np.pi / 2)))
            corner_bonus = float(np.sum(convex[list(quad)]))
            score = area * (1.0 - 0.25 * rect_penalty) + 40.0 * corner_bonus
            if best is None or score > best[0]:
                best = (score, pts)

    if best is not None:
        return best[1]
    # a badly-conditioned edge model still beats the minimum-area rectangle,
    # which the tabs push outwards, so use it if we got one at all
    if by_edges is not None:
        return by_edges

    # ---- fallback: minimum-area rectangle, snapped onto the contour -------
    src = fallback_mask if fallback_mask is not None else contour
    if isinstance(src, np.ndarray) and src.dtype == np.bool_:
        ys, xs = np.nonzero(src)
        pts = np.stack([ys, xs], axis=1).astype(np.float64)
    else:
        pts = contour
    center, size, ang = min_area_rect(pts)
    rc = rect_corners(center, size, ang)
    out = []
    for corner in rc:
        d = np.hypot(res[:, 0] - corner[0], res[:, 1] - corner[1])
        out.append(res[int(np.argmin(d))])
    return np.array(out, dtype=np.float64)


# ==========================================================================
# sides
# ==========================================================================
def split_sides(contour: np.ndarray, corners: np.ndarray,
                n_samples: int = 96) -> list[np.ndarray]:
    """Cut the closed contour at the four corners into four resampled sides."""
    contour = np.asarray(contour, dtype=np.float64)
    idx = []
    for corner in corners:
        d = np.hypot(contour[:, 0] - corner[0], contour[:, 1] - corner[1])
        idx.append(int(np.argmin(d)))

    n = len(contour)
    sides = []
    for i in range(4):
        a, b = idx[i], idx[(i + 1) % 4]
        if b > a:
            seg = contour[a:b + 1]
        else:                                # wraps around the array end
            seg = np.vstack([contour[a:], contour[:b + 1]])
        if len(seg) < 2:
            seg = np.vstack([contour[a], contour[b]])
        sides.append(resample_contour(seg, n_samples))
    return sides


def classify_side(points: np.ndarray, centroid, flat_tol: float = 0.055):
    """Classify one side and return ``(type, profile, length, amplitude)``.

    The chord between the two corners is the reference line.  ``profile[i]``
    is the perpendicular distance of sample ``i`` from that chord, signed so
    that **positive means outward** (pointing away from the piece centroid),
    and divided by the chord length so that pieces of different sizes are
    comparable.

    * ``max |profile| < flat_tol``           -> ``flat``
    * else the extreme is positive           -> ``tab``
    * else                                   -> ``blank``

    ``flat_tol = 0.055`` corresponds to a bump smaller than ~5.5 % of the side
    length, which is well below the ~25 % of a real tab and comfortably above
    the rasterisation noise of a straight cut.
    """
    p = np.asarray(points, dtype=np.float64)
    a, b = p[0], p[-1]
    v = b - a
    length = float(np.hypot(v[0], v[1]))
    if length < 1e-9:
        return SIDE_FLAT, np.zeros(len(p)), 0.0, 0.0
    u = v / length
    # left-hand normal of the chord (image coords, y down)
    nrm = np.array([u[1], -u[0]])
    rel = p - a
    dev = rel[:, 0] * nrm[0] + rel[:, 1] * nrm[1]

    # Orient the normal outward: the centroid must be on the negative side.
    mid = 0.5 * (a + b)
    to_centroid = np.array([centroid[0] - mid[0], centroid[1] - mid[1]])
    if float(np.dot(to_centroid, nrm)) > 0:
        dev = -dev

    profile = dev / length
    amplitude = float(np.max(np.abs(profile)))
    if amplitude < flat_tol:
        return SIDE_FLAT, profile, length, amplitude
    if abs(float(profile.max())) >= abs(float(profile.min())):
        return SIDE_TAB, profile, length, amplitude
    return SIDE_BLANK, profile, length, amplitude


def sample_color_strip(image: np.ndarray, mask: np.ndarray,
                       points: np.ndarray, centroid,
                       depths=(2.0, 4.0, 6.0)) -> np.ndarray:
    """Sample the piece's colour just *inside* a side.

    The offset direction is the inward normal of the side's **chord**, not
    the local tangent normal: in the throat of a blank the local normal runs
    almost along the boundary and steps straight into the background, filling
    the strip with black.  The chord normal always points into the body, and
    because mating sides have exactly opposite chord normals it samples
    corresponding material on both pieces.

    Validity is tested against the mask **eroded by two pixels**: a segmented
    piece's outermost ring is anti-aliased against the background, and the
    closing that seals the mask pulls in a further pixel or so of real
    background.  Invalid samples are retried at smaller depths, then pulled
    towards the centroid, so the strip never contains background.

    Returns ``(M, len(depths), 3)``.
    """
    from .segmentation import disk, erode          # local: avoids a cycle

    img = to_float(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    h, w = mask.shape
    # A sample is "safe" when it is at least two pixels inside the mask *and*
    # carries actual colour.  The second condition matters because
    # ``extract_pieces`` zeroes everything outside the mask, so a black pixel
    # inside the mask is background that the morphological closing pulled in.
    has_colour = img.sum(axis=2) > 1e-6
    inner = erode(mask, disk(2)) & has_colour
    if not inner.any():
        inner = erode(mask, disk(1)) & has_colour
    if not inner.any():
        inner = np.asarray(mask, dtype=bool)
    p = np.asarray(points, dtype=np.float64)
    m = len(p)

    chord = p[-1] - p[0]
    cn = float(np.hypot(chord[0], chord[1]))
    if cn < 1e-9:
        nrm = np.zeros(2)
    else:
        u = chord / cn
        nrm = np.array([u[1], -u[0]])
    mid = 0.5 * (p[0] + p[-1])
    to_c = np.array([centroid[0] - mid[0], centroid[1] - mid[1]])
    if float(np.dot(to_c, nrm)) < 0:
        nrm = -nrm
    nrm = np.broadcast_to(nrm, (m, 2))

    def inside(q):
        yi = np.clip(np.rint(q[:, 0]).astype(np.int64), 0, h - 1)
        xi = np.clip(np.rint(q[:, 1]).astype(np.int64), 0, w - 1)
        return inner[yi, xi]

    inner_pts = None
    out = np.zeros((m, len(depths), 3), dtype=np.float64)
    for j, d in enumerate(depths):
        q = p + nrm * float(d)
        bad = ~inside(q)
        for frac in (0.6, 0.3, 0.0):            # retry closer to the boundary
            if not bad.any():
                break
            alt = p + nrm * (float(d) * frac)
            q[bad] = alt[bad]
            bad = ~inside(q)
        if bad.any():
            # Last resort: snap onto the nearest safely-interior pixel.  This
            # is needed where the piece is locally concave (the throat of a
            # blank), so that neither the normal nor the direction of the
            # centroid points into the piece.
            if inner_pts is None:
                iy, ix = np.nonzero(inner)
                inner_pts = np.stack([iy, ix], axis=1).astype(np.float64)
            for idx in np.flatnonzero(bad):
                dist = np.hypot(inner_pts[:, 0] - q[idx, 0],
                                inner_pts[:, 1] - q[idx, 1])
                q[idx] = inner_pts[int(np.argmin(dist))]
        out[:, j, :] = _bilinear(img, q[:, 0], q[:, 1])
    return out


def _bilinear(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinear sample of an RGB image at floating point coordinates."""
    h, w = img.shape[:2]
    ys = np.clip(ys, 0, h - 1.001)
    xs = np.clip(xs, 0, w - 1.001)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[:, None]
    top = img[y0, x0] * (1 - wx) + img[y0, x0 + 1] * wx
    bot = img[y0 + 1, x0] * (1 - wx) + img[y0 + 1, x0 + 1] * wx
    return top * (1 - wy) + bot * wy


# ==========================================================================
# top level
# ==========================================================================
def describe_piece(piece: Piece, n_side_samples: int = 96,
                   flat_tol: float = 0.055,
                   depths=(2.0, 4.0, 6.0),
                   contour_smooth: int = 7,
                   corners: np.ndarray | None = None) -> PieceDescription:
    """Build the full description (corners + four sides) of one piece.

    ``contour_smooth`` is the width of the moving average applied to the
    traced boundary first.  A raw 8-connected contour is a staircase whose
    arc length is up to ~8 % longer than the underlying curve, and by an
    amount that depends on the piece's rotation; since sides are resampled by
    arc length, that would misalign the profiles of two mating sides
    photographed at different angles.  Smoothing removes the staircase
    without moving the boundary.
    """
    contour = smooth_contour(np.asarray(piece.contour, dtype=np.float64),
                             window=contour_smooth)
    if corners is None:
        corners = find_corners(contour, fallback_mask=piece.mask)
    centroid = piece.centroid
    side_points = split_sides(contour, corners, n_side_samples)

    # scale the sampling depth with the piece size so the descriptor is
    # resolution independent
    body = float(np.mean([np.hypot(*(corners[(i + 1) % 4] - corners[i]))
                          for i in range(4)]))
    scale = max(body / 100.0, 0.35)
    d_scaled = tuple(float(d) * scale for d in depths)

    sides: list[Side] = []
    for i, pts in enumerate(side_points):
        stype, profile, length, amp = classify_side(pts, centroid, flat_tol)
        colors = sample_color_strip(piece.image, piece.mask, pts, centroid,
                                    d_scaled)
        sides.append(Side(index=i, type=stype, points=pts, profile=profile,
                          colors=colors, length=length,
                          corners=(tuple(pts[0]), tuple(pts[-1])),
                          amplitude=amp))

    h = 0.5 * (np.hypot(*(corners[1] - corners[0]))
               + np.hypot(*(corners[3] - corners[2])))
    v = 0.5 * (np.hypot(*(corners[2] - corners[1]))
               + np.hypot(*(corners[0] - corners[3])))
    return PieceDescription(piece=piece, corners=corners, sides=sides,
                            centroid=centroid, body_size=(float(v), float(h)))


def calibrate_side_types(descriptions: list[PieceDescription],
                         lo: float = 0.02, hi: float = 0.22) -> float:
    """Re-classify flat vs. tab/blank with a threshold learnt from the puzzle.

    A fixed ``flat_tol`` has to hold across resolutions, camera angles and how
    crisply a puzzle is cut.  Two facts about the *whole* puzzle pin the
    boundary down instead: the amplitudes are strongly bimodal (flats near
    zero, tabs and blanks near 0.3), and an ``R x C`` grid exposes **exactly**
    ``2(R + C)`` flat sides, so for every factor pair of ``N`` the flat count
    is known in advance.

    Every gap between consecutive sorted amplitudes in ``[lo, hi]`` is a
    candidate threshold.  One whose flat count matches ``2(R + C)`` for some
    factor pair wins (widest such gap, since a wide gap means the classes
    separate cleanly there); the widest gap overall is the fallback when
    segmentation lost pieces and no count can match.  On a dataset photograph
    this recovers 24 of 24 expected flats where the fixed default found 14.

    Modifies the descriptions in place and returns the threshold used.
    """
    amps = np.sort(np.array([s.amplitude for d in descriptions
                             for s in d.sides], dtype=np.float64))
    if amps.size < 8:
        return float("nan")

    n = len(descriptions)
    expected = {2 * (r + n // r) for r in range(1, int(np.sqrt(n)) + 1)
                if n % r == 0}

    best_valid = None      # (gap width, level) with a grid-consistent count
    best_any = None        # (gap width, level) regardless of count
    for k in range(len(amps) - 1):
        gap = amps[k + 1] - amps[k]
        level = 0.5 * (amps[k] + amps[k + 1])
        if not (lo <= level <= hi) or gap <= 1e-9:
            continue
        cand = (gap, level)
        if best_any is None or cand > best_any:
            best_any = cand
        if (k + 1) in expected and (best_valid is None or cand > best_valid):
            best_valid = cand

    chosen = best_valid or best_any
    if chosen is None:
        return float("nan")
    level = float(chosen[1])

    for d in descriptions:
        for s in d.sides:
            if s.amplitude < level:
                s.type = SIDE_FLAT
            elif abs(float(s.profile.max())) >= abs(float(s.profile.min())):
                s.type = SIDE_TAB
            else:
                s.type = SIDE_BLANK
    return level


def refine_descriptions(descriptions: list[PieceDescription],
                        tolerance: float = 0.10,
                        **describe_kwargs) -> list[PieceDescription]:
    """Repair pieces whose corners disagree with the rest of the puzzle.

    Corner detection works on one piece at a time, and on a handful of pieces
    it still lands on a skewed quadrilateral -- a tab that sits close to a
    corner can mislead the edge model, and the fallbacks are weaker.  But the
    pieces of one puzzle are all cut from the same grid, so the **population**
    knows something no single piece does: they share a body size.

    This pass takes the median body size over the pieces whose own estimate
    is self-consistent, flags the pieces that disagree with it by more than
    ``tolerance``, and re-fits those with :func:`fit_body_rectangle`, which
    searches for the maximum-coverage rectangle of the agreed size (trying
    both orientations, since a piece may be lying on its side).  A repair is
    kept only if it covers more of the piece than the original.

    Returns a new list; unaffected descriptions are passed through unchanged.
    """
    if len(descriptions) < 4:
        return list(descriptions)

    good = [d for d in descriptions if _corners_plausible(d.corners)]
    ref = good if len(good) >= max(3, len(descriptions) // 3) else descriptions
    shorts = [min(d.body_size) for d in ref]
    longs = [max(d.body_size) for d in ref]
    s_short = float(np.median(shorts))
    s_long = float(np.median(longs))

    out = []
    for d in descriptions:
        lo, hi = min(d.body_size), max(d.body_size)
        ok = (_corners_plausible(d.corners)
              and abs(lo - s_short) <= tolerance * s_short
              and abs(hi - s_long) <= tolerance * s_long)
        if ok:
            out.append(d)
            continue

        theta = dominant_orientation(
            resample_contour(np.asarray(d.piece.contour, dtype=np.float64),
                             720, closed=True))
        best = None
        for size in ((s_short, s_long), (s_long, s_short)):
            fitted = fit_body_rectangle(d.piece.mask, theta, size,
                                        contour=d.piece.contour)
            if fitted is None:
                continue
            corners, coverage = fitted
            if best is None or coverage > best[1]:
                best = (corners, coverage)
        if best is None:
            out.append(d)
            continue

        repaired = describe_piece(d.piece, corners=best[0], **describe_kwargs)
        repaired.meta["repaired"] = True
        repaired.meta["fit_coverage"] = best[1]
        out.append(repaired)
    return out


def describe_pieces(pieces, refine: bool = True, calibrate: bool = True,
                    **kwargs) -> list[PieceDescription]:
    """Describe every piece, then apply the two whole-puzzle passes.

    Both passes exploit the same fact: the pieces of one puzzle are not
    independent samples, they were cut from one grid with one tool. So

    * :func:`refine_descriptions` repairs the few pieces whose corners
      disagree with the size the rest of the puzzle agrees on -- a single
      skewed piece produces a skewed descriptor *and* a visibly warped piece
      in the rendered reconstruction;
    * :func:`calibrate_side_types` learns the flat/tab boundary from the
      puzzle's own amplitude distribution instead of trusting a fixed
      constant across resolutions and cameras.
    """
    descs = [describe_piece(p, **kwargs) for p in pieces]
    if refine:
        descs = refine_descriptions(descs, **kwargs)
    if calibrate:
        calibrate_side_types(descs)
    return descs
