"""Milestone 1 / task 6 -- Greedy best-first puzzle assembly.

The compatibility table of :mod:`src.edge_matching` is turned into an actual
arrangement by a greedy best-first search over a fixed ``R x C`` grid.

Algorithm
---------
1. **Seed.**  The grid is anchored with a *corner* piece (two adjacent flat
   sides) placed at cell ``(0, 0)``, rotated so that its two flats face North
   and West.  If the piece descriptions contain no corner piece (a puzzle
   without a border, or a failed flat classification), the seed is the piece
   of the most confident seam, placed at the centre of the grid.  The whole
   pass is repeated from every such seed (see *Restarts* below).
2. **Frontier.**  Every empty cell that touches at least one placed piece is
   a candidate.  For each such cell, every unused piece and each of its four
   rotations is scored by

       ``cost(cell, piece, rot) = sum over placed neighbours of
                                  D(side of neighbour, facing side of piece)``

   normalised by the number of matched neighbours, and rejected outright if
   any hard constraint is violated (a border cell must show a flat side, an
   interior seam may not, and every seam must be an admissible tab/blank
   pair).
3. **Selection.**  The best admissible ``(cell, piece, rotation)`` is
   committed and the frontier is updated, until the grid is full.

Tie-breaking rule
-----------------
Candidates are compared on the tuple

    ``(not a best-buddy seam, mean seam cost, -number of matched neighbours,
       -confidence margin, piece index, rotation)``

in that order.  *Best-buddy* seams -- where the two sides are each other's
cheapest partner in the entire puzzle (:func:`_best_buddy_set`) -- are
committed first, because a mutual first choice is far stronger evidence than
a merely cheap one and stops the walk from spending an irreplaceable piece on
a locally attractive but wrong cell.  Then the cheapest mean seam cost; then
the placement constrained by *more* already-placed neighbours (more
evidence); then the larger *margin* to the second-best piece for the same
cell (a uniquely good placement is safer than one where two pieces tie); the
last two entries make the result deterministic.

Restarts
--------
One greedy walk is only as good as its first few decisions, so the pass is
run once per candidate seed and the best arrangement is kept, ranked by
``(pieces placed, forced placements, mean seam cost)``.

Rendering
---------
Once the arrangement is known, :func:`render_assembly` draws it back into a
single image: each piece is mapped onto its grid cell with the similarity
transform (rotation + uniform scale + translation) that sends the two corners
of its North-facing side onto the two top corners of the cell.  Two point
correspondences determine a similarity exactly, and because the cut is
complementary the tabs land precisely in the neighbours' blanks.

Dead ends
---------
If no candidate satisfies the hard constraints, the search does not abort.
It relaxes them in three documented stages -- (i) drop the border/flat
requirement, (ii) allow tab-tab and blank-blank seams at a fixed penalty
``DEADEND_PENALTY``, (iii) place the remaining pieces in reading order --
and records which placements were forced.  Because the best arrangement seen
so far is stored at every step, :func:`assemble` **always** returns the best
arrangement it obtained, complete or not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .edge_matching import CompatibilityTable
from .enhancement import to_float, to_uint8
from .piece_description import DIRECTIONS, PieceDescription, SIDE_FLAT

__all__ = [
    "Placement",
    "Assembly",
    "infer_grid_shape",
    "assemble",
    "similarity_from_pairs",
    "warp_similarity",
    "cell_size_estimate",
    "render_assembly",
]

#: Cost charged for a seam that violates the tab/blank rule (stage-ii
#: relaxation).  Chosen far above any legitimate seam cost so that a forced
#: placement is never preferred to a legal one.
DEADEND_PENALTY = 10.0
#: Cost charged for a seam against a border constraint (stage-i relaxation).
BORDER_PENALTY = 5.0

_STEP = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
_OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}


@dataclass
class Placement:
    """One piece placed in the grid."""
    piece: int
    rotation: int          # index of the side that faces North
    cost: float = 0.0
    forced: bool = False


@dataclass
class Assembly:
    """The result of :func:`assemble`."""
    grid_shape: tuple[int, int]
    #: ``grid[r][c]`` is a :class:`Placement` or ``None``
    grid: list[list[Placement | None]]
    total_cost: float = 0.0
    n_placed: int = 0
    n_forced: int = 0
    seam_costs: list[float] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return self.grid_shape[0]

    @property
    def cols(self) -> int:
        return self.grid_shape[1]

    @property
    def complete(self) -> bool:
        return self.n_placed == self.rows * self.cols

    def piece_at(self, r: int, c: int):
        return self.grid[r][c]

    def cell_of(self, piece: int) -> tuple[int, int] | None:
        for r in range(self.rows):
            for c in range(self.cols):
                p = self.grid[r][c]
                if p is not None and p.piece == piece:
                    return r, c
        return None

    def as_arrays(self):
        """``(piece_id_grid, rotation_grid)`` with ``-1`` for empty cells."""
        pid = np.full(self.grid_shape, -1, dtype=np.int64)
        rot = np.full(self.grid_shape, -1, dtype=np.int64)
        for r in range(self.rows):
            for c in range(self.cols):
                p = self.grid[r][c]
                if p is not None:
                    pid[r, c] = p.piece
                    rot[r, c] = p.rotation
        return pid, rot

    @property
    def mean_seam_cost(self) -> float:
        return float(np.mean(self.seam_costs)) if self.seam_costs else float("nan")


# ==========================================================================
# grid shape
# ==========================================================================
def infer_grid_shape(descriptions: list[PieceDescription]) -> tuple[int, int]:
    """Deduce ``(rows, cols)`` from the number of pieces and their flat sides.

    A ``R x C`` puzzle has ``R*C`` pieces of which ``2(R + C) - 4`` are border
    pieces (at least one flat).  Given ``N = R*C`` and the observed border
    count ``B`` we solve ``R + C = (B + 4) / 2`` together with ``R*C = N``;
    if that has no integer solution we fall back to the factor pair of ``N``
    with the aspect ratio closest to the mean piece aspect ratio.
    """
    n = len(descriptions)
    if n == 0:
        return (0, 0)
    factors = [(r, n // r) for r in range(1, int(np.sqrt(n)) + 1) if n % r == 0]
    if not factors:
        return (1, n)

    border = sum(1 for d in descriptions if d.is_border_piece)
    half = (border + 4) / 2.0
    if abs(half - round(half)) < 1e-6:
        s = int(round(half))
        for r, c in factors:
            if r + c == s:
                return (r, c)

    # fallback: pick the factor pair whose aspect ratio best matches the
    # average piece aspect ratio (pieces are usually near-square, so this
    # amounts to choosing the most square grid)
    aspect = float(np.mean([d.body_size[1] / max(d.body_size[0], 1e-6)
                            for d in descriptions]))
    best = min(factors, key=lambda rc: abs((rc[1] / rc[0]) - 1.0 / max(aspect, 1e-6)))
    return best


# ==========================================================================
# assembly
# ==========================================================================
def _seam_cost(table: CompatibilityTable, placed: Placement,
               direction: str, cand_piece: int, cand_rot: int) -> float:
    """``D`` across the seam between an empty cell and a placed neighbour.

    ``direction`` points *from the empty cell towards the placed piece*.  The
    candidate therefore contributes the side facing ``direction`` and the
    already-placed piece the side facing the opposite way.
    """
    s = (placed.rotation + DIRECTIONS.index(_OPPOSITE[direction])) % 4
    t = (cand_rot + DIRECTIONS.index(direction)) % 4
    return float(table.cost[placed.piece, s, cand_piece, t])


def _side_type(descs, piece: int, rot: int, direction: str) -> str:
    d_i = DIRECTIONS.index(direction)
    return descs[piece].sides[(rot + d_i) % 4].type


def _corner_seeds(descs) -> list[tuple[int, int]]:
    """Every ``(piece, rotation)`` that puts a corner piece's flats N and W."""
    out = []
    for d in descs:
        flags = [s.is_flat for s in d.sides]
        if sum(flags) != 2:
            continue
        for i in range(4):
            # sides i (N) and i+3 (W) must both be flat
            if flags[i] and flags[(i + 3) % 4]:
                out.append((d.index, i))
    return out


def _best_buddy_set(table: CompatibilityTable) -> set:
    """``{(i, s, j, t)}`` for side pairs that are each other's best match.

    A pair is a *best buddy* when neither side has a cheaper partner anywhere
    in the puzzle.  Best buddies are the highest-confidence evidence
    available before any search, and preferring them turns the greedy walk
    away from the locally cheap but globally wrong seams that make a plain
    cheapest-first strategy brittle.
    """
    n = table.cost.shape[0]
    flat = table.cost.reshape(n * 4, n * 4)
    if flat.size == 0:
        return set()
    best = np.argmin(np.where(np.isfinite(flat), flat, np.inf), axis=1)
    out = set()
    for a in range(n * 4):
        b = int(best[a])
        if np.isfinite(flat[a, b]) and int(best[b]) == a:
            out.add((a // 4, a % 4, b // 4, b % 4))
            out.add((b // 4, b % 4, a // 4, a % 4))
    return out


def _greedy_once(descs, table, rows, cols, seed_piece, seed_rot, seed_cell,
                 buddies, verbose=False) -> Assembly:
    """One greedy best-first pass from a fixed seed."""
    n = len(descs)
    grid: list[list[Placement | None]] = [[None] * cols for _ in range(rows)]
    used = np.zeros(n, dtype=bool)
    log: list[str] = []
    seam_costs: list[float] = []
    total = 0.0
    forced = 0

    sr, sc = seed_cell
    grid[sr][sc] = Placement(piece=seed_piece, rotation=seed_rot)
    used[seed_piece] = True
    log.append(f"seed: piece {seed_piece} rotation {seed_rot} at ({sr},{sc})")

    def neighbours_of(r, c):
        out = []
        for d, (dr, dc) in _STEP.items():
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] is not None:
                out.append((d, grid[rr][cc]))
        return out

    def frontier():
        out = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] is None and neighbours_of(r, c):
                    out.append((r, c))
        return out

    # ---------------- main loop -------------------------------------------
    for _ in range(rows * cols - 1):
        cells = frontier()
        if not cells:
            break

        # relaxation stages: 0 = all constraints, 1 = ignore border/flat
        # rule, 2 = also allow illegal tab/blank pairs
        chosen = None
        for stage in (0, 1, 2):
            best_per_cell = {}
            for (r, c) in cells:
                nbrs = neighbours_of(r, c)
                scored = []
                for j in range(n):
                    if used[j]:
                        continue
                    for rot in range(4):
                        cost = 0.0
                        ok = True
                        # border constraints
                        for d, (dr, dc) in _STEP.items():
                            rr, cc = r + dr, c + dc
                            outside = not (0 <= rr < rows and 0 <= cc < cols)
                            t = _side_type(descs, j, rot, d)
                            if outside and t != SIDE_FLAT:
                                if stage == 0:
                                    ok = False
                                    break
                                cost += BORDER_PENALTY
                            if (not outside) and t == SIDE_FLAT:
                                if stage == 0:
                                    ok = False
                                    break
                                cost += BORDER_PENALTY
                        if not ok:
                            continue
                        # seam costs
                        acc, k, bb = 0.0, 0, False
                        for d, placed in nbrs:
                            cs = _seam_cost(table, placed, d, j, rot)
                            if not np.isfinite(cs):
                                if stage < 2:
                                    ok = False
                                    break
                                cs = DEADEND_PENALTY
                            s_placed = (placed.rotation
                                        + DIRECTIONS.index(_OPPOSITE[d])) % 4
                            t_cand = (rot + DIRECTIONS.index(d)) % 4
                            if (placed.piece, s_placed, j, t_cand) in buddies:
                                bb = True
                            acc += cs
                            k += 1
                        if not ok or k == 0:
                            continue
                        scored.append((cost + acc / k, k, bb, j, rot))
                if scored:
                    scored.sort(key=lambda e: e[0])
                    margin = (scored[1][0] - scored[0][0]) if len(scored) > 1 else 1e3
                    # prefer the cheapest best-buddy candidate for this cell
                    bbs = [e for e in scored if e[2]]
                    entry = bbs[0] if bbs else scored[0]
                    best_per_cell[(r, c)] = (entry, margin)
            if best_per_cell:
                # ordering: best-buddy placements first, then cheapest, then
                # the most constrained cell, then the largest margin, then
                # piece/rotation index for determinism
                cell, (entry, margin) = min(
                    best_per_cell.items(),
                    key=lambda kv: (not kv[1][0][2], kv[1][0][0], -kv[1][0][1],
                                    -kv[1][1], kv[1][0][3], kv[1][0][4]))
                chosen = (cell, entry, margin, stage)
                break

        if chosen is None:
            break
        (r, c), (cost, k, bb, j, rot), margin, stage = chosen
        grid[r][c] = Placement(piece=j, rotation=rot, cost=float(cost),
                               forced=stage > 0)
        used[j] = True
        total += float(cost)
        seam_costs.append(float(cost))
        if stage > 0:
            forced += 1
        if verbose:
            log.append(f"place piece {j} rot {rot} at ({r},{c}) "
                       f"cost {cost:.4f} margin {margin:.4f} stage {stage}")

    # ---------------- stage (iii): anything still unplaced ------------------
    left = [i for i in range(n) if not used[i]]
    empty = [(r, c) for r in range(rows) for c in range(cols)
             if grid[r][c] is None]
    if left and empty:
        k = min(len(left), len(empty))
        log.append(f"dead end: {k} piece(s) placed in reading order")
        for (r, c) in empty[:k]:
            j = left.pop(0)
            grid[r][c] = Placement(piece=j, rotation=0, cost=DEADEND_PENALTY,
                                   forced=True)
            used[j] = True
            forced += 1
            total += DEADEND_PENALTY
    if left:
        log.append(f"{len(left)} piece(s) left over: more pieces than cells")
    elif empty and len(empty) > (rows * cols - n):
        log.append(f"{len(empty)} cell(s) left empty")

    n_placed = sum(1 for r in range(rows) for c in range(cols)
                   if grid[r][c] is not None)
    return Assembly(grid_shape=(rows, cols), grid=grid, total_cost=float(total),
                    n_placed=n_placed, n_forced=forced, seam_costs=seam_costs,
                    log=log)


def _arrangement_key(a: Assembly):
    """Ranking key for restarts: most placed, fewest forced, cheapest seams."""
    mean = a.mean_seam_cost
    if not np.isfinite(mean):
        mean = float("inf")
    return (-a.n_placed, a.n_forced, mean)


def assemble(descriptions: list[PieceDescription], table: CompatibilityTable,
             grid_shape: tuple[int, int] | None = None,
             max_restarts: int = 8, verbose: bool = False) -> Assembly:
    """Greedy best-first reconstruction.  See the module docstring.

    The greedy pass is repeated from several seeds -- every corner piece in
    every orientation that puts its two flats on the outside, plus the most
    confident best-buddy pair when there are no corner pieces -- and the best
    arrangement obtained is returned, ranked by (pieces placed, forced
    placements, mean seam cost).  This is what makes the "always returns the
    best arrangement obtained, even when incomplete" guarantee meaningful:
    a single greedy walk can be led astray by one bad early seam, whereas the
    restarts give it several independent chances and never discard a better
    result.
    """
    descs = list(descriptions)
    n = len(descs)
    if n == 0:
        return Assembly((0, 0), [], 0.0, 0)
    # An explicitly supplied grid is authoritative even when the piece count
    # does not match it.  On a real photograph segmentation may recover 34 or
    # 36 pieces of a 35-piece puzzle, and honouring the known 5x7 layout --
    # leaving a cell empty, or leaving a spare piece unplaced -- is far more
    # useful than silently re-deriving a 2x17 grid from the wrong count.
    if grid_shape is not None:
        rows, cols = int(grid_shape[0]), int(grid_shape[1])
    else:
        rows, cols = infer_grid_shape(descs)

    buddies = _best_buddy_set(table)

    seeds: list[tuple[int, int, tuple[int, int]]] = []
    for p, r0 in _corner_seeds(descs):
        seeds.append((p, r0, (0, 0)))
    if not seeds:
        # no usable corner piece: seed from the most confident best buddy,
        # placed in the middle so the grid can grow in every direction
        flat = table.cost.reshape(n * 4, n * 4)
        order = np.argsort(np.where(np.isfinite(flat), flat, np.inf), axis=None)
        for a in order[:max_restarts]:
            i = int(a) // (n * 4)
            seeds.append((i // 4, 0, (rows // 2, cols // 2)))
    if not seeds:
        seeds = [(0, 0, (0, 0))]

    # keep the search bounded: too many corner candidates means the flat
    # classification is unreliable anyway
    seeds = seeds[:max(1, int(max_restarts))]

    best: Assembly | None = None
    for (p, r0, cell) in seeds:
        cand = _greedy_once(descs, table, rows, cols, p, r0, cell, buddies,
                            verbose=verbose)
        if best is None or _arrangement_key(cand) < _arrangement_key(best):
            best = cand
    assert best is not None
    best.log.insert(0, f"{len(seeds)} restart(s); best kept "
                       f"(placed {best.n_placed}, forced {best.n_forced}, "
                       f"mean seam {best.mean_seam_cost:.4f})")
    return best


# ==========================================================================
# rendering the arrangement back into an image
# ==========================================================================
def similarity_from_pairs(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (rotation + uniform scale + translation).

    Takes **any** number of point correspondences (at least two), given as
    ``(y, x)``, and returns the ``2 x 3`` matrix ``M`` with
    ``[y', x'] = M @ [y, x, 1]``.  With exactly two points the fit is exact;
    with the piece's four corners it is a least-squares fit, which is what
    the renderer uses -- a single mis-located corner then carries a quarter
    of the leverage instead of half, and the piece is not visibly skewed by
    it.
    """
    p = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    q = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2 or len(p) != len(q):
        raise ValueError("need at least two matching point pairs")

    pm, qm = p.mean(axis=0), q.mean(axis=0)
    u, v = p - pm, q - qm
    den = float((u * u).sum())
    if den < 1e-12:
        return np.array([[1.0, 0.0, qm[0] - pm[0]], [0.0, 1.0, qm[1] - pm[1]]])
    # [Y; X] = [[a, -b], [b, a]] [y; x] + t
    a = float((u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]).sum()) / den
    b = float((u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()) / den
    ty = qm[0] - (a * pm[0] - b * pm[1])
    tx = qm[1] - (b * pm[0] + a * pm[1])
    return np.array([[a, -b, ty], [b, a, tx]], dtype=np.float64)


def _invert_affine(m: np.ndarray) -> np.ndarray:
    a = m[:, :2]
    t = m[:, 2]
    inv = np.linalg.inv(a)
    return np.hstack([inv, (-inv @ t).reshape(2, 1)])


def warp_similarity(image: np.ndarray, mask: np.ndarray, m: np.ndarray,
                    out_shape: tuple[int, int]):
    """Warp ``image``/``mask`` by the ``2 x 3`` transform ``m``.

    Returns ``(warped_image, warped_mask)`` of shape ``out_shape``.  Sampling
    is bilinear for the image and nearest for the mask (thresholded at 0.5),
    using the inverse transform so every output pixel is filled exactly once.
    """
    img = to_float(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    h, w = img.shape[:2]
    oh, ow = out_shape
    inv = _invert_affine(np.asarray(m, dtype=np.float64))

    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float64)
    sy = inv[0, 0] * yy + inv[0, 1] * xx + inv[0, 2]
    sx = inv[1, 0] * yy + inv[1, 1] * xx + inv[1, 2]

    valid = (sy >= 0) & (sy <= h - 1.001) & (sx >= 0) & (sx <= w - 1.001)
    syc = np.clip(sy, 0, h - 1.001)
    sxc = np.clip(sx, 0, w - 1.001)
    y0 = np.floor(syc).astype(np.int64)
    x0 = np.floor(sxc).astype(np.int64)
    wy = (syc - y0)[..., None]
    wx = (sxc - x0)[..., None]
    top = img[y0, x0] * (1 - wx) + img[y0, x0 + 1] * wx
    bot = img[y0 + 1, x0] * (1 - wx) + img[y0 + 1, x0 + 1] * wx
    out_img = top * (1 - wy) + bot * wy

    mf = mask.astype(np.float64)
    mt = mf[y0, x0] * (1 - wx[..., 0]) + mf[y0, x0 + 1] * wx[..., 0]
    mb = mf[y0 + 1, x0] * (1 - wx[..., 0]) + mf[y0 + 1, x0 + 1] * wx[..., 0]
    out_mask = (mt * (1 - wy[..., 0]) + mb * wy[..., 0]) > 0.5
    out_mask &= valid
    out_img[~out_mask] = 0.0
    return out_img, out_mask


def cell_size_estimate(descriptions: list[PieceDescription]) -> tuple[int, int]:
    """Median piece-body size, used as the size of a grid cell."""
    if not descriptions:
        return (1, 1)
    hs = [d.body_size[0] for d in descriptions]
    ws = [d.body_size[1] for d in descriptions]
    return (max(1, int(round(float(np.median(hs))))),
            max(1, int(round(float(np.median(ws))))))


def _fill_gaps(canvas: np.ndarray, filled: np.ndarray, region: np.ndarray,
               iterations: int = 4) -> np.ndarray:
    """Close the hairline gaps that survive between two warped pieces.

    Neighbouring pieces are each scaled to make their own body exactly one
    cell wide, so a piece whose corners were localised a pixel early can
    leave a one-pixel crack against its neighbour.  Uncovered pixels inside
    the grid are filled with the mean of their already-filled 8-neighbours,
    repeated a few times; this is cosmetic only and never invents content
    where a whole piece is missing (an uncovered pixel with no filled
    neighbour is left alone).
    """
    out = canvas.copy()
    have = filled.copy()
    for _ in range(iterations):
        todo = region & ~have
        if not todo.any():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(out.shape[:2], dtype=np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted_v = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
                shifted_m = np.roll(np.roll(have, dy, axis=0), dx, axis=1)
                acc += shifted_v * shifted_m[:, :, None]
                cnt += shifted_m
        usable = todo & (cnt > 0)
        out[usable] = acc[usable] / cnt[usable][:, None]
        have |= usable
    return out


def render_assembly(descriptions: list[PieceDescription], assembly,
                    cell_size: tuple[int, int] | None = None,
                    background=(0, 0, 0), margin: float = 0.45,
                    fill_gaps: bool = True, return_geometry: bool = False):
    """Draw the arrangement as a single reconstructed image.

    ``margin`` is the extra canvas, as a fraction of a cell, reserved around
    the grid so that tabs sticking out of the border pieces are not clipped.

    With ``return_geometry`` the call also returns
    ``(pad_y, pad_x, cell_h, cell_w)`` so the caller can crop the exact
    ``rows*cell_h x cols*cell_w`` body of the puzzle and compare it with a
    reference picture.
    """
    rows, cols = assembly.grid_shape
    ch, cw = cell_size or cell_size_estimate(descriptions)
    pad_y = int(round(margin * ch))
    pad_x = int(round(margin * cw))
    H = rows * ch + 2 * pad_y
    W = cols * cw + 2 * pad_x

    canvas = np.zeros((H, W, 3), dtype=np.float64)
    canvas[:] = np.asarray(background, dtype=np.float64) / 255.0
    filled = np.zeros((H, W), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            pl = assembly.grid[r][c]
            if pl is None:
                continue
            d = descriptions[pl.piece]
            k = pl.rotation                     # side k faces North
            # all four corners, clockwise from the piece's north-west one,
            # onto the four corners of the cell
            src = np.array([d.corners[(k + i) % 4] for i in range(4)])
            y0, x0 = pad_y + r * ch, pad_x + c * cw
            dst = np.array([[y0, x0], [y0, x0 + cw],
                            [y0 + ch, x0 + cw], [y0 + ch, x0]], dtype=np.float64)
            m = similarity_from_pairs(src, dst)
            wi, wm = warp_similarity(d.piece.image, d.piece.mask, m, (H, W))
            put = wm & ~filled
            canvas[put] = wi[put]
            filled |= wm

    if fill_gaps:
        region = np.zeros((H, W), dtype=bool)
        region[pad_y:pad_y + rows * ch, pad_x:pad_x + cols * cw] = True
        canvas = _fill_gaps(canvas, filled, region)

    out = to_uint8(canvas)
    if return_geometry:
        return out, (pad_y, pad_x, ch, cw)
    return out
