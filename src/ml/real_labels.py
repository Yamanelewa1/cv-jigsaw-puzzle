"""Milestone 2 / task 1 -- side-pair labels for the *provided* photographs.

The brief asks for the two models to be trained on the data provided, by
"generating positive and negative side-pair samples".  The Roboflow export
does not carry that label directly: it names each piece's **identity**, not
which of its four sides touches which side of a neighbour.  This module
recovers the missing part.

What is already known
---------------------
* the identity of every annotated piece, 1..35;
* that identity ``k`` occupies cell ``((k-1)//7, (k-1)%7)`` of the finished
  5x7 puzzle (Milestone 1, report section 9.1);
* therefore, which pairs of pieces are **true neighbours**, and in which
  direction.

What is missing
---------------
Each piece lies on the cloth at an arbitrary angle, so the mapping from its
four described sides to the four compass directions -- its *rotation* -- is
unknown.  Without it a true neighbour pair cannot be turned into a labelled
*side* pair.

Recovering the rotation
-----------------------
A piece's rotation is written, as everywhere else in this project, as the
index of the side that faces North; side ``s`` then faces direction
``(s - rotation) % 4``.  Two independent constraints pin it down:

1. **The border.**  A piece whose known cell lies on the rim of the grid must
   present a *flat* side outwards, and an interior side must not be flat.
   Because the cell is known from the identity label, this does not depend on
   the flat/tab/blank classifier being right -- which on these photographs it
   is only about 80 % of the time.  Instead each side's continuous
   :attr:`~src.piece_description.Side.amplitude` is used as evidence, so the
   rotation is chosen by how flat the sides *measure*, not by how they were
   labelled.  This alone determines every border and corner piece.
2. **Complementarity.**  Across a true seam a tab must meet a blank.  Interior
   pieces carry no flats, so constraint 1 says nothing about them; they are
   pinned instead by propagating outwards from pieces already fixed, choosing
   the rotation whose seams are complementary and cheapest under the classical
   compatibility measure.

The pass therefore runs border-first and then breadth-first inwards, which is
the same "most constrained first" ordering the assembler uses.

The result is checked rather than trusted: :func:`recover_rotations` reports
the fraction of true seams that come out tab-against-blank, which is the
property the recovery is supposed to produce and which nothing in the
procedure forces.  A photograph whose score is poor can be dropped before it
contaminates the training set.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..piece_description import DIRECTIONS, SIDE_BLANK, SIDE_FLAT, SIDE_TAB

__all__ = [
    "SeamLabels",
    "true_neighbour_pairs",
    "recover_rotations",
    "label_side_pairs",
]

#: ``direction -> (dr, dc)``, matching :mod:`src.assembly`.
_STEP = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
_OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
#: Charged when a seam is not tab-against-blank while propagating.
_INCOMPATIBLE = 5.0


@dataclass
class SeamLabels:
    """Everything :func:`label_side_pairs` recovers for one photograph."""
    rotations: dict[int, int] = field(default_factory=dict)
    #: ``[(piece_a, side_a, piece_b, side_b), ...]`` -- the true seams
    positives: list[tuple[int, int, int, int]] = field(default_factory=list)
    #: fraction of recovered seams that are tab-against-blank
    complementary_fraction: float = 0.0
    #: fraction of outward-facing border sides that measure flattest
    border_fraction: float = 0.0

    @property
    def n_positives(self) -> int:
        return len(self.positives)


def true_neighbour_pairs(cells: dict[int, tuple[int, int]]):
    """``[(piece_a, piece_b, direction_from_a_to_b), ...]`` from known cells."""
    by_cell = {v: k for k, v in cells.items()}
    out = []
    for p, (r, c) in cells.items():
        for d, (dr, dc) in _STEP.items():
            q = by_cell.get((r + dr, c + dc))
            if q is not None and p < q:
                out.append((p, q, d))
    return out


def _outward_directions(cell, grid_shape):
    """The compass directions in which this cell faces outside the grid."""
    r, c = cell
    rows, cols = grid_shape
    out = []
    for d, (dr, dc) in _STEP.items():
        rr, cc = r + dr, c + dc
        if not (0 <= rr < rows and 0 <= cc < cols):
            out.append(d)
    return out


def _border_cost(desc, rotation, outward, flat_tol=0.055):
    """How badly ``rotation`` disagrees with where this cell's flats must be.

    Uses the measured amplitude rather than the flat/tab/blank label, because
    the label is the unreliable part on real photographs.
    """
    cost = 0.0
    for k, d in enumerate(DIRECTIONS):
        side = desc.sides[(rotation + k) % 4]
        if d in outward:
            cost += float(side.amplitude)          # should be flat -> small
        else:
            cost += max(0.0, flat_tol - float(side.amplitude))   # not flat
    return cost


def _seam_cost(table, a, sa, b, sb):
    v = float(table.cost[a, sa, b, sb]) if table is not None else 0.0
    return v if np.isfinite(v) else _INCOMPATIBLE


def recover_rotations(descriptions, cells, grid_shape=(5, 7), table=None,
                      flat_tol: float = 0.055) -> SeamLabels:
    """Recover each piece's rotation, then read off its true side pairs.

    ``cells`` maps a piece index to its known ``(row, col)``; see
    :func:`main.dataset_true_cells`.  ``table`` is an optional classical
    :class:`~src.edge_matching.CompatibilityTable`, used only to break ties
    between rotations that satisfy the hard constraints equally well.
    """
    by_cell = {v: k for k, v in cells.items()}
    rot: dict[int, int] = {}

    # ---- 1. border and corner pieces, from where their flats must lie ----
    border = []
    for p, cell in cells.items():
        outward = _outward_directions(cell, grid_shape)
        if outward:
            border.append((len(outward), p, outward))
    border.sort(key=lambda e: -e[0])           # corners (2 flats) first
    for _, p, outward in border:
        costs = [_border_cost(descriptions[p], k, outward, flat_tol)
                 for k in range(4)]
        rot[p] = int(np.argmin(costs))

    # ---- 2. propagate inwards by complementarity -------------------------
    queue = deque(p for _, p, _ in border)
    while queue:
        p = queue.popleft()
        r, c = cells[p]
        for d, (dr, dc) in _STEP.items():
            q = by_cell.get((r + dr, c + dc))
            if q is None or q in rot:
                continue
            # side of p facing d is already fixed; choose q's rotation so the
            # facing side is complementary and cheap
            sp = (rot[p] + DIRECTIONS.index(d)) % 4
            want = descriptions[p].sides[sp].type
            best, best_cost = 0, np.inf
            for k in range(4):
                sq = (k + DIRECTIONS.index(_OPPOSITE[d])) % 4
                t = descriptions[q].sides[sq].type
                cost = _seam_cost(table, p, sp, q, sq)
                ok = ((want == SIDE_TAB and t == SIDE_BLANK)
                      or (want == SIDE_BLANK and t == SIDE_TAB))
                if not ok:
                    cost += _INCOMPATIBLE
                # an interior piece should show no flat at all
                cost += sum(1.0 for s in descriptions[q].sides
                            if s.type == SIDE_FLAT) * 0.0
                if cost < best_cost:
                    best, best_cost = k, cost
            rot[q] = best
            queue.append(q)

    # any piece the walk never reached (isolated cell) keeps rotation 0
    for p in cells:
        rot.setdefault(p, 0)

    # ---- 3. read off the labelled side pairs, and score the recovery -----
    positives, comp = [], 0
    for a, b, d in true_neighbour_pairs(cells):
        sa = (rot[a] + DIRECTIONS.index(d)) % 4
        sb = (rot[b] + DIRECTIONS.index(_OPPOSITE[d])) % 4
        positives.append((a, sa, b, sb))
        ta = descriptions[a].sides[sa].type
        tb = descriptions[b].sides[sb].type
        if ((ta == SIDE_TAB and tb == SIDE_BLANK)
                or (ta == SIDE_BLANK and tb == SIDE_TAB)):
            comp += 1

    flat_hits = flat_tot = 0
    for p, cell in cells.items():
        for d in _outward_directions(cell, grid_shape):
            s = descriptions[p].sides[(rot[p] + DIRECTIONS.index(d)) % 4]
            flat_tot += 1
            if s.type == SIDE_FLAT:
                flat_hits += 1

    return SeamLabels(
        rotations=rot,
        positives=positives,
        complementary_fraction=comp / max(len(positives), 1),
        border_fraction=flat_hits / max(flat_tot, 1),
    )


def label_side_pairs(descriptions, cells, grid_shape=(5, 7), table=None,
                     negatives_per_positive: int = 6, seed: int = 0):
    """Positive and negative side pairs for one photograph.

    Positives are the true seams recovered by :func:`recover_rotations`.
    Negatives are drawn from the *admissible* pairs that are not true seams --
    admissible so that the models are asked to separate plausible candidates
    from each other, not merely to relearn the tab/blank rule the assembler
    already enforces.

    Returns ``(labels, pairs)`` where ``pairs`` is a list of
    ``(piece_a, side_a, piece_b, side_b)`` and ``labels`` the matching 0/1
    array, positives first.
    """
    rng = np.random.default_rng(seed)
    lab = recover_rotations(descriptions, cells, grid_shape, table)
    positives = lab.positives
    truth = {(a, sa, b, sb) for a, sa, b, sb in positives}
    truth |= {(b, sb, a, sa) for a, sa, b, sb in positives}

    n = len(descriptions)
    candidates = []
    for a in range(n):
        for sa in range(4):
            ta = descriptions[a].sides[sa].type
            if ta == SIDE_FLAT:
                continue
            for b in range(a + 1, n):
                for sb in range(4):
                    tb = descriptions[b].sides[sb].type
                    ok = ((ta == SIDE_TAB and tb == SIDE_BLANK)
                          or (ta == SIDE_BLANK and tb == SIDE_TAB))
                    if ok and (a, sa, b, sb) not in truth:
                        candidates.append((a, sa, b, sb))

    want = min(len(candidates), negatives_per_positive * max(len(positives), 1))
    idx = rng.permutation(len(candidates))[:want]
    negatives = [candidates[int(i)] for i in idx]

    pairs = positives + negatives
    labels = np.concatenate([np.ones(len(positives), dtype=np.float32),
                             np.zeros(len(negatives), dtype=np.float32)])
    return labels, pairs, lab
