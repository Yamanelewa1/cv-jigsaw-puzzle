"""Milestone 1 / task 5 -- Piece-edge matching.

A single number says how well two sides fit together.  The exact formula
used by this library is

.. math::

    D(a, b) \\;=\\; w_s \\, D_{shape}(a,b) \\;+\\; w_c \\, D_{colour}(a,b)
                \\;+\\; w_l \\, D_{length}(a,b)

evaluated **only** for admissible pairs; inadmissible pairs get
``D = +inf``.

Admissibility (the "does the shape fit" rule, hard constraints)
---------------------------------------------------------------
* a **flat** side is a border of the whole puzzle -- it can never be an
  interior seam, so any pair involving a flat is inadmissible;
* a **tab** must meet a **blank**: ``tab-tab`` and ``blank-blank`` are
  inadmissible;
* a side never matches another side of the same piece.

Shape term (does the outline continue)
--------------------------------------
Each side carries a profile :math:`p(t)`, :math:`t\\in[0,1]`, the signed
perpendicular deviation from its corner-to-corner chord divided by the chord
length (positive = outward).  When two pieces are placed next to each other,
one side is traversed in the opposite direction to the other and the two
outward normals point in opposite directions, so a **perfect** fit satisfies
:math:`p_a(t) = -p_b(1-t)`.  The residual of that identity, in RMS, is the
shape cost:

.. math::

    D_{shape}(a,b) = \\sqrt{\\tfrac{1}{M}\\sum_{i=1}^{M}
                     \\bigl(p_a(t_i) + p_b(1-t_i)\\bigr)^2}

Because both profiles are normalised by their own chord length, the term is
dimensionless and independent of piece size.  A length-consistency term

.. math::

    D_{length}(a,b) = \\frac{|L_a - L_b|}{\\max(L_a, L_b)}

penalises pairing sides of visibly different physical size.

Colour term (does the picture line up)
--------------------------------------
Each side carries a strip of colours :math:`C(t, d)` sampled at :math:`D`
depths just inside the piece.  With the same reversal as above, the cost is
the RMS of the squared colour differences (the "sum of squared differences"
the brief asks for, divided by the number of terms so it stays in
``[0, 1]``):

.. math::

    D_{colour}(a,b) = \\sqrt{\\tfrac{1}{3MD}\\sum_{i,d}
                      \\bigl\\|C_a(t_i,d) - C_b(1-t_i,d)\\bigr\\|^2}

Photometric alternatives
------------------------
``build_compatibility(..., colour_metric=...)`` selects what the photometric
term measures:

``"ssd"`` (default)
    the squared colour differences above -- "is B the same colour as A?".
``"mgc"``
    :func:`gradient_compatibility`, Gallagher's Mahalanobis gradient
    compatibility -- "does A's picture *continue* into B?".  On the dataset
    photographs it raises the matcher's top-1 rate from 0.203 to 0.276.
``"mgc+ssd"``
    both, each divided by its own median so neither dominates.

``colour_norm="meanstd"`` additionally removes each strip's own mean and
spread before the SSD comparison, which cancels the per-piece illumination
differences of a photographed puzzle.

Weights
-------
The defaults are :math:`w_s = 1.0`, :math:`w_c = 1.0`, :math:`w_l = 0.5`, so
shape and colour contribute **equally** and length is a half-weight tie
breaker.  This is justified by the two terms having the same order of
magnitude on true and on random pairs, so neither needs rescaling to be heard
(``results/evaluation_results/matching_study.json``, produced by
``main.py --run matching``):

===========  ============  =====================  ============
Term         True seams    All admissible pairs   Separation
===========  ============  =====================  ============
D_shape      0.024         0.057                  2.4x
D_colour     0.058         0.194                  3.3x
D_length     0.005         0.005                  1.1x
===========  ============  =====================  ============

The two are **not** equally strong, and the report says so rather than
implying otherwise.  A sweep over eight weightings
(``results/evaluation_results/weight_study.json``, ``main.py --run weights``)
measures shape alone at a top-1 match rate of ~0.08 and colour alone at
~0.88-0.91: on a machine-cut puzzle every tab has nearly the same silhouette,
so geometry barely separates candidates.  Shape is kept at full weight
because adding it to colour still *helps* -- top-1 rises to ~0.90-0.92 --
since it disambiguates precisely the seams colour cannot, those crossing a
uniformly coloured region.  ``D_length`` is near-constant within a puzzle
(all pieces are cut to one size), which is why it carries only half weight:
it contributes nothing on clean data and acts as a safety net when
segmentation produces a size outlier, as happens on the real photographs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .piece_description import (PieceDescription, SIDE_BLANK, SIDE_FLAT,
                                SIDE_TAB)

__all__ = [
    "MatchWeights",
    "shape_distance",
    "colour_distance",
    "length_distance",
    "sides_admissible",
    "side_compatibility",
    "gradient_compatibility",
    "CompatibilityTable",
    "build_compatibility",
    "best_buddies",
]


@dataclass(frozen=True)
class MatchWeights:
    """Weights of the three terms of the dissimilarity formula."""
    shape: float = 1.0
    colour: float = 1.0
    length: float = 0.5

    def as_dict(self) -> dict:
        return {"shape": self.shape, "colour": self.colour,
                "length": self.length}


# ==========================================================================
# individual terms
# ==========================================================================
#: Corner localisation is accurate to a few pixels, and a corner found a
#: little early on one piece and a little late on its neighbour shifts the
#: whole profile.  Both the shape and the colour term are therefore minimised
#: over integer sample shifts of up to ``MAX_SHIFT_FRACTION`` of the side.
MAX_SHIFT_FRACTION = 0.05


def _shift_range(m: int) -> int:
    return max(1, int(round(MAX_SHIFT_FRACTION * m)))


def shape_distance(profile_a: np.ndarray, profile_b: np.ndarray,
                   max_shift: int | None = None) -> float:
    """RMS of ``p_a(t) + p_b(1-t)`` -- zero for a perfect tab/blank fit.

    Minimised over shifts of up to ``max_shift`` samples (default: 5 % of the
    side), evaluated on the overlapping part of the two profiles.
    """
    a = np.asarray(profile_a, dtype=np.float64)
    b = np.asarray(profile_b, dtype=np.float64)[::-1]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    k = _shift_range(n) if max_shift is None else int(max_shift)
    best = np.inf
    for s in range(-k, k + 1):
        aa = a[max(0, s):n + min(0, s)]
        bb = b[max(0, -s):n + min(0, -s)]
        if aa.size == 0:
            continue
        r = aa + bb
        best = min(best, float(np.sqrt(np.mean(r * r))))
    return best


def colour_distance(colors_a: np.ndarray, colors_b: np.ndarray,
                    max_shift: int | None = None) -> float:
    """RMS colour difference between the two strips, one of them reversed.

    Minimised over the same shift range as :func:`shape_distance`.
    """
    a = np.asarray(colors_a, dtype=np.float64)
    b = np.asarray(colors_b, dtype=np.float64)[::-1]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    k = _shift_range(n) if max_shift is None else int(max_shift)
    best = np.inf
    for s in range(-k, k + 1):
        aa = a[max(0, s):n + min(0, s)]
        bb = b[max(0, -s):n + min(0, -s)]
        if aa.size == 0:
            continue
        d = aa - bb
        best = min(best, float(np.sqrt(np.mean(d * d))))
    return best


def _pairwise_min_shift_rms(a: np.ndarray, b_rev: np.ndarray, sign: float,
                            max_shift: int) -> np.ndarray:
    """Pairwise RMS of ``a[i] + sign * b_rev[j]``, minimised over shifts.

    ``a`` and ``b_rev`` have shape ``(S, M, F)``: ``S`` sides, ``M`` samples
    along the side, ``F`` features per sample.  For each shift the distance
    is obtained from Gram matrices via ``||u - v||^2 = ||u||^2 - 2 u.v +
    ||v||^2``, which keeps the whole table one matrix product per shift.
    """
    s, m, f = a.shape
    best = np.full((s, s), np.inf)
    for k in range(-max_shift, max_shift + 1):
        aa = a[:, max(0, k):m + min(0, k), :]
        bb = b_rev[:, max(0, -k):m + min(0, -k), :]
        w = aa.shape[1]
        if w <= 0:
            continue
        af = aa.reshape(s, w * f)
        bf = bb.reshape(s, w * f)
        sq_a = (af * af).sum(axis=1)
        sq_b = (bf * bf).sum(axis=1)
        cross = af @ bf.T
        d2 = (sq_a[:, None] + sq_b[None, :] + 2.0 * sign * cross) / (w * f)
        np.minimum(best, np.sqrt(np.maximum(d2, 0.0)), out=best)
    return best


def length_distance(len_a: float, len_b: float) -> float:
    """Relative difference of the two chord lengths."""
    m = max(float(len_a), float(len_b))
    return abs(float(len_a) - float(len_b)) / m if m > 0 else 0.0


def sides_admissible(type_a: str, type_b: str) -> bool:
    """``True`` when the two side types may form an interior seam."""
    if type_a == SIDE_FLAT or type_b == SIDE_FLAT:
        return False
    return (type_a == SIDE_TAB and type_b == SIDE_BLANK) or \
           (type_a == SIDE_BLANK and type_b == SIDE_TAB)


def side_compatibility(side_a, side_b,
                       weights: MatchWeights = MatchWeights()) -> float:
    """The dissimilarity ``D(a, b)`` of the module docstring (lower = better).

    Returns ``+inf`` for inadmissible pairs.
    """
    if not sides_admissible(side_a.type, side_b.type):
        return float("inf")
    return (weights.shape * shape_distance(side_a.profile, side_b.profile)
            + weights.colour * colour_distance(side_a.colors, side_b.colors)
            + weights.length * length_distance(side_a.length, side_b.length))


# ==========================================================================
# full table
# ==========================================================================
@dataclass
class CompatibilityTable:
    """All pairwise side dissimilarities of one puzzle.

    ``cost[i, s, j, t]`` is ``D`` between side ``s`` of piece ``i`` and side
    ``t`` of piece ``j``; inadmissible entries are ``+inf``.  The individual
    terms are kept for the report and for the weight study.
    """
    cost: np.ndarray                 # (N, 4, N, 4)
    shape: np.ndarray
    colour: np.ndarray
    length: np.ndarray
    weights: MatchWeights

    @property
    def n_pieces(self) -> int:
        return self.cost.shape[0]

    def side_cost(self, i: int, s: int, j: int, t: int) -> float:
        return float(self.cost[i, s, j, t])

    def rank_of(self, i: int, s: int, j: int, t: int) -> int:
        """Rank of ``(j, t)`` among all candidates for side ``(i, s)``."""
        flat = self.cost[i, s].reshape(-1)
        order = np.argsort(flat, kind="stable")
        target = j * 4 + t
        pos = int(np.flatnonzero(order == target)[0])
        return pos


def _normalise_strip(c: np.ndarray, mode: str) -> np.ndarray:
    """Remove per-strip illumination from a colour strip.

    ``"none"``
        Compare absolute colours.  Correct when both pieces were imaged
        under the same light, as in a scan or a synthetic puzzle.
    ``"meanstd"``
        Subtract each strip's own mean and divide by its own spread, so only
        the *pattern* along the strip is compared.  On the dataset
        photographs the pieces lie all over a table and each is lit slightly
        differently, so two strips facing each other across a true seam
        differ by an offset and a gain even where the picture continues
        perfectly; removing both raises the matcher's top-1 rate from 0.19 to
        0.25 and the precision of its best-buddy pairs from 0.26 to 0.40.
    """
    if mode == "none":
        return c
    if mode == "meanstd":
        c = c - c.mean(axis=1, keepdims=True)
        return c / (c.std(axis=1, keepdims=True) + 1e-3)
    raise ValueError(f"unknown colour normalisation {mode!r}")


def gradient_compatibility(colors: np.ndarray, max_shift: int) -> np.ndarray:
    """Mahalanobis gradient compatibility (Gallagher's MGC), pairwise.

    Comparing absolute colour asks *"is B the same colour as A?"*.  That is
    the wrong question for a printed picture photographed piece by piece: the
    right one is *"does A's picture continue into B?"*.  MGC asks it directly.

    From A's colour strip, the outward gradient at sample ``t`` is
    ``G_a(t) = C_a(t, d0) - C_a(t, d1)`` -- how the colour is already changing
    as it approaches the cut.  Extrapolating one step gives a **prediction**
    of what lies just beyond the edge, ``P_a(t) = C_a(t, d0) + G_a(t)``, and
    the residual against what B actually shows there is

        ``r(t) = C_b(1-t, d0) - P_a(t)``

    That residual is measured not in raw RGB but in the metric of A's *own*
    gradient covariance ``S_a``:

        ``D_mgc(a, b) = (1 / 3M) * sum_t  r(t)^T S_a^-1 r(t)``

    so a colour change that A's own texture makes routinely is cheap, while
    one it never makes is expensive.  Because only differences enter, the
    measure is invariant to any illumination offset, and because it rewards
    continuation rather than agreement it keeps working where the picture is
    nearly uniform -- which is exactly the dataset's situation.

    The result is symmetrised, ``(D(a,b) + D(b,a)) / 2``, and minimised over
    the same small shift range as the other terms.  ``colors`` is
    ``(S, M, D, 3)`` with ``D >= 2`` depths ordered shallowest first.
    """
    c = np.asarray(colors, dtype=np.float64)
    if c.ndim != 4 or c.shape[2] < 2:
        raise ValueError("MGC needs colour strips with at least two depths")
    s, m, _, _ = c.shape

    grad = c[:, :, 0, :] - c[:, :, 1, :]         # outward gradient
    pred = c[:, :, 0, :] + grad                  # predicted just outside
    edge = c[:, :, 0, :]                         # actual boundary colour

    # whitening matrix of each side's own gradient covariance
    whiten = np.zeros((s, 3, 3))
    for a in range(s):
        cov = np.cov(grad[a].T) + np.eye(3) * 1e-3
        try:
            whiten[a] = np.linalg.cholesky(np.linalg.inv(cov)).T
        except np.linalg.LinAlgError:            # degenerate strip
            whiten[a] = np.eye(3)

    edge_rev = edge[:, ::-1, :]
    best = np.full((s, s), np.inf)
    for k in range(-max_shift, max_shift + 1):
        a0, a1 = max(0, k), m + min(0, k)
        b0, b1 = max(0, -k), m + min(0, -k)
        pa, eb = pred[:, a0:a1, :], edge_rev[:, b0:b1, :]
        w = pa.shape[1]
        if w <= 4:
            continue
        d = np.empty((s, s))
        for a in range(s):
            r = (eb - pa[a][None, :, :]) @ whiten[a].T
            d[a] = (r * r).sum(axis=(1, 2)) / (w * 3)
        np.minimum(best, d, out=best)
    return 0.5 * (best + best.T)


def _median_scale(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Divide by the median of the admissible entries, so terms are comparable."""
    vals = x[mask & np.isfinite(x)]
    med = float(np.median(vals)) if vals.size else 1.0
    return x / max(med, 1e-9)


def build_compatibility(descriptions: list[PieceDescription],
                        weights: MatchWeights = MatchWeights(),
                        colour_norm: str = "none",
                        colour_metric: str = "ssd") -> CompatibilityTable:
    """Compute the full ``(N, 4, N, 4)`` dissimilarity table.

    The three terms are evaluated for every pair at once with matrix
    algebra: expanding ``||u - v||^2 = ||u||^2 - 2 u.v + ||v||^2`` turns the
    ``(4N)^2`` comparisons into two Gram matrices, which keeps a 100-piece
    puzzle (160 000 pairs) well under a second.
    """
    n = len(descriptions)
    if n == 0:
        empty = np.zeros((0, 4, 0, 4))
        return CompatibilityTable(empty, empty, empty, empty, weights)

    m = len(descriptions[0].sides[0].profile)
    n_feat = int(np.prod(descriptions[0].sides[0].colors.shape[1:]))
    prof = np.zeros((n * 4, m, 1), dtype=np.float64)
    cols = np.zeros((n * 4, m, n_feat), dtype=np.float64)
    lens = np.zeros(n * 4, dtype=np.float64)
    types: list[str] = []

    for i, d in enumerate(descriptions):
        for s, side in enumerate(d.sides):
            k = i * 4 + s
            prof[k, :, 0] = np.asarray(side.profile, dtype=np.float64)[:m]
            cols[k] = np.asarray(side.colors, dtype=np.float64).reshape(m, -1)
            lens[k] = side.length
            types.append(side.type)

    max_shift = _shift_range(m)
    # shape: the fit condition is p_a(t) = -p_b(1-t), hence the "+" sign
    shape = _pairwise_min_shift_rms(prof, prof[:, ::-1], +1.0, max_shift)
    # colour: the fit condition is C_a(t) = C_b(1-t), hence the "-" sign
    normed = _normalise_strip(cols, colour_norm)
    colour = _pairwise_min_shift_rms(normed, normed[:, ::-1], -1.0, max_shift)

    # ---- length ----------------------------------------------------------
    lmax = np.maximum(lens[:, None], lens[None, :])
    length = np.abs(lens[:, None] - lens[None, :]) / np.maximum(lmax, 1e-9)

    # ---- admissibility ---------------------------------------------------
    t_arr = np.array(types)
    is_tab = t_arr == SIDE_TAB
    is_blank = t_arr == SIDE_BLANK
    ok = (is_tab[:, None] & is_blank[None, :]) | (is_blank[:, None] & is_tab[None, :])
    piece_of = np.repeat(np.arange(n), 4)
    ok &= piece_of[:, None] != piece_of[None, :]

    # ---- combine ---------------------------------------------------------
    if colour_metric == "ssd":
        photometric = colour
    else:
        mgc = gradient_compatibility(cols.reshape(n * 4, m, -1, 3), max_shift)
        if colour_metric == "mgc":
            photometric = mgc
        elif colour_metric == "mgc+ssd":
            # each term divided by its own median over admissible pairs, so
            # the sum is not dominated by whichever happens to be larger
            photometric = _median_scale(mgc, ok) + _median_scale(colour, ok)
        else:
            raise ValueError(f"unknown colour metric {colour_metric!r}")
        colour = photometric

    cost = (weights.shape * shape + weights.colour * photometric
            + weights.length * length)
    cost = np.where(ok, cost, np.inf)

    shp = (n, 4, n, 4)
    return CompatibilityTable(
        cost=cost.reshape(shp),
        shape=shape.reshape(shp),
        colour=colour.reshape(shp),
        length=length.reshape(shp),
        weights=weights,
    )


def best_buddies(table: CompatibilityTable) -> list[tuple]:
    """Side pairs that are each other's *mutual* best candidate.

    Best-buddy pairs are the high-confidence seeds of the assembly: if side
    ``a`` prefers ``b`` above everything else and ``b`` also prefers ``a``,
    the pair is very likely a true seam.  Returned as
    ``[(i, s, j, t, cost), ...]`` sorted by cost.
    """
    cost = table.cost
    n = cost.shape[0]
    flat = cost.reshape(n * 4, n * 4)
    best = np.argmin(flat, axis=1)
    out = []
    for a in range(n * 4):
        b = int(best[a])
        if not np.isfinite(flat[a, b]):
            continue
        if int(best[b]) == a and a < b:
            out.append((a // 4, a % 4, b // 4, b % 4, float(flat[a, b])))
    out.sort(key=lambda r: r[4])
    return out
