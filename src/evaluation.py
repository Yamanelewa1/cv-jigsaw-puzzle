"""Milestone 1 -- Reconstruction-quality metrics.

Two families of measures are provided.

**Reference-free** (always available, this is the number the end-to-end
routine returns together with the reconstructed image):

* :func:`seam_quality` -- statistics of the dissimilarities actually paid
  along the seams of the arrangement, and a single ``quality`` score in
  ``[0, 1]`` obtained by comparing them with the distribution of *all*
  admissible seams of the same puzzle.  A perfect reconstruction pays only
  the very best seams and scores near 1; a random arrangement pays average
  seams and scores near 0.

**Ground-truth based** (available for the synthetic puzzles, and for any
puzzle whose layout is known):

* :func:`direct_accuracy` -- fraction of pieces in the right cell with the
  right rotation, maximised over the four global rotations of the grid,
  because a jigsaw solved upside-down is still solved;
* :func:`neighbour_accuracy` -- fraction of true adjacencies that are
  reproduced, which is invariant to any global transform;
* :func:`rotation_accuracy` -- fraction of pieces whose orientation relative
  to the grid is correct;
* :func:`matching_accuracy` -- how often the *compatibility measure alone*
  (before assembly) ranks the true neighbouring side first;
* :func:`image_metrics` -- MSE / PSNR / SSIM between the reconstructed image
  and the original picture.

**Ground truth itself.**  The dataset photographs come with no answer key --
nobody labels which cell of the finished picture each photographed piece
belongs to -- so this module also contains the synthetic puzzle *generator*
(:func:`generate_puzzle`) that cuts any image into interlocking pieces,
shuffles and rotates them onto a background, and remembers the answer.  That
is what makes the ground-truth metrics above measurable at all, and it is
what the automated tests reconstruct.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os

import numpy as np

from .contour_extraction import rotate_image, rotate_mask
from .enhancement import gaussian_blur, to_float, to_gray, to_uint8
from .piece_description import DIRECTIONS

__all__ = [
    # --- synthetic puzzles with ground truth -------------------------------
    "PuzzleGroundTruth",
    "synthetic_source_image",
    "cut_puzzle",
    "scatter_pieces",
    "generate_puzzle",
    "save_ground_truth",
    "load_ground_truth",
    # --- metrics -----------------------------------------------------------
    "seam_quality",
    "direct_accuracy",
    "neighbour_accuracy",
    "neighbour_accuracy_from_cells",
    "position_accuracy_from_cells",
    "orientation_accuracy_from_cells",
    "rotation_accuracy",
    "matching_accuracy",
    "mse", "psnr", "ssim",
    "image_metrics",
    "associate_with_ground_truth",
    "gt_side_directions",
    "EvaluationReport",
    "save_report",
]

_STEP = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


# ==========================================================================
# synthetic puzzles with ground truth
# ==========================================================================
@dataclass
class PuzzleGroundTruth:
    """Everything needed to score a reconstruction."""
    grid_shape: tuple[int, int]
    cell_size: tuple[int, int]                 # (height, width) of a body
    # placements[i] = (row, col, rotation_degrees) of scattered piece i
    placements: list[tuple[int, int, float]] = field(default_factory=list)
    # centroid of each scattered piece on the canvas, used to associate the
    # pieces the pipeline segments with the pieces the generator drew
    centroids: list[tuple[float, float]] = field(default_factory=list)
    source_image: str = ""
    seed: int = 0

    @property
    def rows(self) -> int:
        return int(self.grid_shape[0])

    @property
    def cols(self) -> int:
        return int(self.grid_shape[1])

    def cell_of(self, piece_index: int) -> tuple[int, int]:
        r, c, _ = self.placements[piece_index]
        return int(r), int(c)


# --------------------------------------------------------------------------
# a reproducible source picture
# --------------------------------------------------------------------------
def synthetic_source_image(height: int = 480, width: int = 640,
                           seed: int = 0) -> np.ndarray:
    """A colourful, textured picture used when no real photo is supplied.

    Deliberately built from low-frequency colour fields *plus* hard-edged
    shapes and fine noise: the smooth part makes colour matching meaningful,
    the hard edges give the edge detector something to find, and the noise
    stops neighbouring pieces from being trivially interchangeable.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    v = yy / max(height - 1, 1)
    u = xx / max(width - 1, 1)

    img = np.zeros((height, width, 3), dtype=np.float64)
    for ch in range(3):
        acc = np.zeros((height, width))
        for _ in range(5):
            fx, fy = rng.uniform(0.8, 4.0), rng.uniform(0.8, 4.0)
            ph = rng.uniform(0, 2 * np.pi)
            acc += np.sin(2 * np.pi * (fx * u + fy * v) + ph)
        img[:, :, ch] = acc
    # Rescale to [0.25, 1]: the picture must stay clearly brighter than the
    # (black) table, otherwise dark picture content would be indistinguishable
    # from background for *any* thresholding rule.
    img = (img - img.min()) / max(img.max() - img.min(), 1e-9)
    img = 0.25 + 0.75 * img

    # hard-edged blobs and bars
    for _ in range(14):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        r = rng.uniform(0.04, 0.13) * min(height, width)
        col = rng.uniform(0.25, 1.0, size=3)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[m] = col
    for _ in range(8):
        y0 = int(rng.uniform(0, height - 10))
        x0 = int(rng.uniform(0, width - 10))
        hh = int(rng.uniform(6, 0.10 * height))
        ww = int(rng.uniform(6, 0.30 * width))
        img[y0:y0 + hh, x0:x0 + ww] = rng.uniform(0.25, 1.0, size=3)

    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    return to_uint8(img)


# --------------------------------------------------------------------------
# cutting
# --------------------------------------------------------------------------
def cut_puzzle(image: np.ndarray, rows: int = 3, cols: int = 4,
               tab_ratio: float = 0.22, neck: float = 0.35,
               rng: np.random.Generator | None = None):
    """Cut ``image`` into ``rows x cols`` interlocking pieces.

    Parameters
    ----------
    tab_ratio:
        Tab circle radius as a fraction of the shorter cell side.
    neck:
        How far (in radii) the circle centre sits outside the cut line.
        ``0`` gives a plain semicircle; larger values give the pinched neck of
        a real jigsaw tab.

    Returns
    -------
    ``(masks, cells)`` where ``masks[i]`` is a full-size boolean mask of piece
    ``i`` and ``cells[i]`` is its ``(row, col)``.  Pieces are returned in
    row-major order.
    """
    rng = rng or np.random.default_rng(0)
    img = to_float(image)
    H, W = img.shape[:2]
    ch, cw = H // rows, W // cols
    base_rad = tab_ratio * min(ch, cw)

    # Randomly orient every internal cut: +1 -> the tab belongs to the piece
    # on the smaller index side, -1 -> to the other one.
    v_sign = rng.choice([-1, 1], size=(rows, cols - 1))       # vertical cuts
    h_sign = rng.choice([-1, 1], size=(rows - 1, cols))       # horizontal cuts

    # Every cut gets its own radius, neck depth and along-edge offset, so no
    # two seams have the same silhouette.  A real puzzle is die-cut the same
    # way; without this variation the shape term would carry no information
    # at all and every tab would fit every blank equally well.
    def cut_params(shape):
        return (base_rad * rng.uniform(0.82, 1.18, size=shape),
                rng.uniform(0.22, 0.48, size=shape) if neck else np.zeros(shape),
                rng.uniform(-0.13, 0.13, size=shape))

    v_rad, v_neck, v_off = cut_params((rows, cols - 1))
    h_rad, h_neck, h_off = cut_params((rows - 1, cols))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    def circle(cy, cx, rad):
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad

    masks: list[np.ndarray] = []
    cells: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * ch, (r + 1) * ch if r < rows - 1 else H
            x0, x1 = c * cw, (c + 1) * cw if c < cols - 1 else W
            m = np.zeros((H, W), dtype=bool)
            m[y0:y1, x0:x1] = True

            # ---- vertical cut on the right of this piece -----------------
            if c < cols - 1:
                s = int(v_sign[r, c])
                rad = float(v_rad[r, c])
                cy = (y0 + y1) / 2.0 + float(v_off[r, c]) * (y1 - y0)
                cx = x1 + s * float(v_neck[r, c]) * rad
                circ = circle(cy, cx, rad)
                m = (m | circ) if s > 0 else (m & ~circ)
            # ---- vertical cut on the left --------------------------------
            if c > 0:
                s = int(v_sign[r, c - 1])
                rad = float(v_rad[r, c - 1])
                cy = (y0 + y1) / 2.0 + float(v_off[r, c - 1]) * (y1 - y0)
                cx = x0 + s * float(v_neck[r, c - 1]) * rad
                circ = circle(cy, cx, rad)
                m = (m & ~circ) if s > 0 else (m | circ)
            # ---- horizontal cut below ------------------------------------
            if r < rows - 1:
                s = int(h_sign[r, c])
                rad = float(h_rad[r, c])
                cy = y1 + s * float(h_neck[r, c]) * rad
                cx = (x0 + x1) / 2.0 + float(h_off[r, c]) * (x1 - x0)
                circ = circle(cy, cx, rad)
                m = (m | circ) if s > 0 else (m & ~circ)
            # ---- horizontal cut above -----------------------------------
            if r > 0:
                s = int(h_sign[r - 1, c])
                rad = float(h_rad[r - 1, c])
                cy = y0 + s * float(h_neck[r - 1, c]) * rad
                cx = (x0 + x1) / 2.0 + float(h_off[r - 1, c]) * (x1 - x0)
                circ = circle(cy, cx, rad)
                m = (m & ~circ) if s > 0 else (m | circ)

            masks.append(m)
            cells.append((r, c))
    return masks, cells


# --------------------------------------------------------------------------
# scattering
# --------------------------------------------------------------------------
def scatter_pieces(image: np.ndarray, masks, cells, rotate: bool = True,
                   background=(0, 0, 0), gap: int = 14,
                   rng: np.random.Generator | None = None,
                   max_angle: float = 180.0, jitter: float = 0.12,
                   snap_to_90: bool = False):
    """Lay the cut pieces out, shuffled and rotated, on a fresh canvas.

    Pieces are dropped into a loose grid of slots that is large enough for the
    rotated bounding boxes, with a random jitter inside each slot, so that
    they never touch (touching pieces would merge into one connected
    component -- exactly the failure mode visible in some of the dataset
    photographs).

    Returns ``(canvas, placements)`` where ``placements[i]`` is
    ``(row, col, rotation_degrees)`` of the piece drawn *i*-th.
    """
    rng = rng or np.random.default_rng(0)
    img = to_float(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)

    order = rng.permutation(len(masks))
    crops, placements = [], []
    for i in order:
        m = masks[i]
        ys, xs = np.nonzero(m)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub_img = img[y0:y1, x0:x1].copy()
        sub_m = m[y0:y1, x0:x1]
        sub_img[~sub_m] = 0.0

        angle_deg = 0.0
        if rotate:
            if snap_to_90:
                angle_deg = float(rng.choice([0.0, 90.0, 180.0, 270.0]))
            else:
                angle_deg = float(rng.uniform(-max_angle, max_angle))
            a = np.deg2rad(angle_deg)
            h, w = sub_m.shape
            diag = int(np.ceil(np.hypot(h, w))) + 6
            center = ((h - 1) / 2.0, (w - 1) / 2.0)
            rot_img = rotate_image(sub_img, a, center, (diag, diag))
            rot_m = rotate_mask(sub_m, a, center, (diag, diag))
            ry, rx = np.nonzero(rot_m)
            sub_img = rot_img[ry.min():ry.max() + 1, rx.min():rx.max() + 1]
            sub_m = rot_m[ry.min():ry.max() + 1, rx.min():rx.max() + 1]
            sub_img = sub_img.copy()
            sub_img[~sub_m] = 0.0

        crops.append((sub_img, sub_m))
        placements.append((cells[i][0], cells[i][1], angle_deg))

    n = len(crops)
    slot_h = max(c[1].shape[0] for c in crops) + gap
    slot_w = max(c[1].shape[1] for c in crops) + gap
    grid_cols = int(np.ceil(np.sqrt(n * slot_h / float(slot_w))))
    grid_cols = max(1, min(n, grid_cols))
    grid_rows = int(np.ceil(n / grid_cols))

    canvas = np.zeros((grid_rows * slot_h + gap, grid_cols * slot_w + gap, 3))
    canvas[:] = np.asarray(background, dtype=np.float64) / 255.0

    centroids = []
    for k, (sub_img, sub_m) in enumerate(crops):
        gr, gc = divmod(k, grid_cols)
        h, w = sub_m.shape
        free_y = slot_h - h
        free_x = slot_w - w
        oy = gap + gr * slot_h + int(free_y * (0.5 + jitter * rng.uniform(-1, 1)))
        ox = gap + gc * slot_w + int(free_x * (0.5 + jitter * rng.uniform(-1, 1)))
        oy = max(0, min(canvas.shape[0] - h, oy))
        ox = max(0, min(canvas.shape[1] - w, ox))
        region = canvas[oy:oy + h, ox:ox + w]
        region[sub_m] = sub_img[sub_m]
        ys, xs = np.nonzero(sub_m)
        centroids.append((float(ys.mean() + oy), float(xs.mean() + ox)))

    return to_uint8(canvas), placements, centroids


def generate_puzzle(image: np.ndarray, rows: int = 3, cols: int = 4,
                    rotate: bool = True, seed: int = 0,
                    tab_ratio: float = 0.22, snap_to_90: bool = False,
                    source_name: str = ""):
    """Cut + scatter in one call.

    Returns ``(scrambled_image, ground_truth)``.
    """
    rng = np.random.default_rng(seed)
    masks, cells = cut_puzzle(image, rows, cols, tab_ratio=tab_ratio, rng=rng)
    canvas, placements, centroids = scatter_pieces(
        image, masks, cells, rotate=rotate, rng=rng, snap_to_90=snap_to_90)
    H, W = np.asarray(image).shape[:2]
    gt = PuzzleGroundTruth(
        grid_shape=(rows, cols),
        cell_size=(H // rows, W // cols),
        placements=[(int(a), int(b), float(c)) for a, b, c in placements],
        centroids=[(float(a), float(b)) for a, b in centroids],
        source_image=source_name,
        seed=seed,
    )
    return canvas, gt


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def save_ground_truth(gt: PuzzleGroundTruth, path: str) -> str:
    """Write the ground truth to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = asdict(gt)
    d["grid_shape"] = list(gt.grid_shape)
    d["cell_size"] = list(gt.cell_size)
    d["placements"] = [list(p) for p in gt.placements]
    d["centroids"] = [list(p) for p in gt.centroids]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
    return path


def load_ground_truth(path: str) -> PuzzleGroundTruth:
    """Read a ground-truth JSON written by :func:`save_ground_truth`."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return PuzzleGroundTruth(
        grid_shape=tuple(d["grid_shape"]),
        cell_size=tuple(d["cell_size"]),
        placements=[tuple(p) for p in d["placements"]],
        centroids=[tuple(p) for p in d.get("centroids", [])],
        source_image=d.get("source_image", ""),
        seed=int(d.get("seed", 0)),
    )


# ==========================================================================
# metrics
# ==========================================================================


# ==========================================================================
# reference-free
# ==========================================================================
def seam_quality(assembly, table) -> dict:
    """Score the arrangement from the seam dissimilarities it pays.

    ``quality = 1 - mean(seam cost) / mean(all admissible costs)`` clipped to
    ``[0, 1]``: the denominator is what an arrangement that pairs sides at
    random would pay on average, so the score answers "how much better than
    chance is this arrangement?".
    """
    rows, cols = assembly.grid_shape
    costs = []
    illegal = 0
    for r in range(rows):
        for c in range(cols):
            a = assembly.grid[r][c]
            if a is None:
                continue
            for d, (dr, dc) in _STEP.items():
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols):
                    continue
                if (rr, cc) < (r, c):
                    continue                       # count each seam once
                b = assembly.grid[rr][cc]
                if b is None:
                    continue
                s = (a.rotation + DIRECTIONS.index(d)) % 4
                opp = {"N": "S", "E": "W", "S": "N", "W": "E"}[d]
                t = (b.rotation + DIRECTIONS.index(opp)) % 4
                v = float(table.cost[a.piece, s, b.piece, t])
                if np.isfinite(v):
                    costs.append(v)
                else:
                    illegal += 1

    finite = table.cost[np.isfinite(table.cost)]
    baseline = float(np.mean(finite)) if finite.size else 1.0
    mean_cost = float(np.mean(costs)) if costs else float("nan")
    quality = 0.0 if not costs else float(
        np.clip(1.0 - mean_cost / max(baseline, 1e-9), 0.0, 1.0))
    return {
        "n_seams": len(costs) + illegal,
        "n_illegal_seams": illegal,
        "mean_seam_cost": mean_cost,
        "median_seam_cost": float(np.median(costs)) if costs else float("nan"),
        "max_seam_cost": float(np.max(costs)) if costs else float("nan"),
        "random_baseline_cost": baseline,
        "quality": quality,
    }


# ==========================================================================
# ground-truth association
# ==========================================================================
def associate_with_ground_truth(pieces, gt) -> list[int]:
    """Map extracted piece index -> index in ``gt.placements``.

    Association is by centroid: the generator records where it drew every
    piece, and the segmentation recovers the same blobs, so nearest-centroid
    matching is exact as long as the pieces do not overlap.
    """
    if not gt.centroids:
        raise ValueError("ground truth has no centroids to associate with")
    gtc = np.asarray(gt.centroids, dtype=np.float64)
    out = []
    for p in pieces:
        y = p.bbox[0] + p.centroid[0]
        x = p.bbox[1] + p.centroid[1]
        d = np.hypot(gtc[:, 0] - y, gtc[:, 1] - x)
        out.append(int(np.argmin(d)))
    return out


def gt_side_directions(description, angle_deg: float) -> list[str]:
    """Which grid direction each side of a piece faced *before* scattering.

    The scatter rotated the piece by ``angle_deg``; undoing that rotation on
    the centroid-to-side-midpoint vectors tells us, for every side, whether
    it was the North, East, South or West side of the assembled puzzle.
    Convention-free: it only uses geometry, never the side ordering.
    """
    a = np.deg2rad(float(angle_deg))
    cy, cx = description.centroid
    scores = []
    for side in description.sides:
        # the chord midpoint is a far more stable direction indicator than the
        # middle contour sample, which a tab pushes well off the side
        mid = 0.5 * (side.points[0] + side.points[-1])
        vy, vx = mid[0] - cy, mid[1] - cx
        # inverse of the rotation applied by contour_extraction.rotate_image
        y0 = vy * np.cos(a) - vx * np.sin(a)
        x0 = vy * np.sin(a) + vx * np.cos(a)
        n = np.hypot(y0, x0) + 1e-12
        scores.append(-y0 / n)                 # +1 when the side points North

    # The sides are stored clockwise and N,E,S,W is clockwise too, so once we
    # know which side points North the rest follow.  Assigning them as a
    # cyclic block (instead of independently) guarantees a valid permutation
    # even when two sides happen to point in similar directions.
    start = int(np.argmax(scores))
    return [DIRECTIONS[(i - start) % 4] for i in range(4)]


# ==========================================================================
# ground-truth metrics
# ==========================================================================
def _gt_grid(_pieces, gt, assoc) -> dict[tuple[int, int], int]:
    """``(row, col) -> extracted piece index`` from the ground truth."""
    out = {}
    for i, g in enumerate(assoc):
        r, c, _ = gt.placements[g]
        out[(int(r), int(c))] = i
    return out


def _rotate_grid(grid: dict, rows: int, cols: int, k: int):
    """Rotate a ``(r, c) -> value`` map by ``k`` quarter turns clockwise."""
    out = dict(grid)
    R, C = rows, cols
    for _ in range(k % 4):
        out = {(c, R - 1 - r): v for (r, c), v in out.items()}
        R, C = C, R
    return out, (R, C)


def _orientation_offsets(assembly, descriptions, gt, assoc) -> dict:
    """``piece -> quarter-turns`` between its placed and its original pose."""
    rows, cols = assembly.grid_shape
    out = {}
    for r in range(rows):
        for c in range(cols):
            pl = assembly.grid[r][c]
            if pl is None:
                continue
            dirs = gt_side_directions(descriptions[pl.piece],
                                      gt.placements[assoc[pl.piece]][2])
            out[pl.piece] = DIRECTIONS.index(dirs[pl.rotation])
    return out


def direct_accuracy(assembly, descriptions, gt, assoc) -> dict:
    """Fraction of pieces in the correct cell (and with correct rotation).

    Maximised over the four global rotations of the reconstructed grid: a
    puzzle assembled rotated as a whole is still correctly assembled, and
    nothing in the input tells the algorithm which way is up.  For the
    combined position+rotation figure, the *same* global quarter-turn offset
    must hold for every counted piece; the offset that satisfies the most
    pieces is used.
    """
    rows, cols = assembly.grid_shape
    truth = _gt_grid(None, gt, assoc)
    pid, _ = assembly.as_arrays()
    offsets = _orientation_offsets(assembly, descriptions, gt, assoc)
    n = len(descriptions)

    best = {"position_accuracy": 0.0, "position_and_rotation_accuracy": 0.0,
            "global_rotation": 0}
    for k in range(4):
        rotated, (R, C) = _rotate_grid(
            {(r, c): int(pid[r, c]) for r in range(rows) for c in range(cols)
             if pid[r, c] >= 0}, rows, cols, k)
        if (R, C) != (gt.rows, gt.cols):
            continue
        correct = [p for (r, c), p in rotated.items() if truth.get((r, c)) == p]
        pos = len(correct)
        if correct:
            hist = np.bincount([offsets.get(p, 0) for p in correct], minlength=4)
            posrot = int(hist.max())
        else:
            posrot = 0
        if pos / max(n, 1) > best["position_accuracy"]:
            best = {"position_accuracy": pos / max(n, 1),
                    "position_and_rotation_accuracy": posrot / max(n, 1),
                    "global_rotation": k}
    return best


def neighbour_accuracy_from_cells(assembly, cells: dict) -> dict:
    """Neighbour accuracy against a plain ``piece -> (row, col)`` map.

    Separated from :func:`neighbour_accuracy` so that ground truth coming
    from somewhere other than the synthetic generator -- for instance the
    identity labels of the provided dataset, whose class ids turn out to be
    the row-major positions of the finished 5x7 puzzle -- can be scored with
    exactly the same measure.
    """
    rows, cols = assembly.grid_shape
    pid, _ = assembly.as_arrays()
    placed = {int(pid[r, c]): (r, c) for r in range(rows) for c in range(cols)
              if pid[r, c] >= 0}

    inv = {v: k for k, v in cells.items()}
    truth_pairs = set()
    for (r, c), i in inv.items():
        for dr, dc in _STEP.values():
            j = inv.get((r + dr, c + dc))
            if j is not None and i < j:
                truth_pairs.add((i, j))

    got = 0
    for (i, j) in truth_pairs:
        if i not in placed or j not in placed:
            continue
        (r1, c1), (r2, c2) = placed[i], placed[j]
        if abs(r1 - r2) + abs(c1 - c2) == 1:
            got += 1
    return {"neighbour_accuracy": got / max(len(truth_pairs), 1),
            "n_true_adjacencies": len(truth_pairs), "n_recovered": got}


def position_accuracy_from_cells(assembly, cells: dict) -> dict:
    """Fraction of pieces in the right cell, best over the four global turns."""
    rows, cols = assembly.grid_shape
    pid, _ = assembly.as_arrays()
    grid = {(r, c): int(pid[r, c]) for r in range(rows) for c in range(cols)
            if pid[r, c] >= 0}
    n = max(len(cells), 1)
    best = 0.0
    best_k = 0
    for k in range(4):
        rotated, _ = _rotate_grid(grid, rows, cols, k)
        hit = sum(1 for (rc, p) in rotated.items() if cells.get(p) == rc)
        if hit / n > best:
            best, best_k = hit / n, k
    return {"position_accuracy": best, "global_rotation": best_k,
            "n_scored": len(cells)}


def orientation_accuracy_from_cells(assembly, rotations: dict,
                                    cells: dict | None = None) -> dict:
    """Fraction of pieces placed at the correct orientation.

    ``rotations`` maps a piece to the side index that faced North in the
    finished puzzle (see :meth:`src.ml.dataset.PuzzleSample.true_rotations`).
    A placement records the same quantity, so a piece is correctly oriented
    when the two agree.

    As with position, the arrangement as a whole may come out turned: nothing
    in the input says which way is up, and a puzzle assembled sideways is
    still assembled.  One global quarter-turn offset is therefore allowed,
    the *same* one for every piece, and the offset satisfying the most pieces
    is used -- which is the rule :func:`rotation_accuracy` applies to the
    synthetic ground truth.

    Passing ``cells`` additionally reports the combined figure: the fraction
    in the right cell **and** at the right orientation under a single shared
    quarter turn, which is the strictest reading of "position and orientation
    accuracy".
    """
    rows, cols = assembly.grid_shape
    pid, rot = assembly.as_arrays()
    placed = [(int(pid[r, c]), int(rot[r, c]))
              for r in range(rows) for c in range(cols) if pid[r, c] >= 0]
    scored = [(p, k) for (p, k) in placed if p in rotations]
    n = max(len(rotations), 1)

    hist = np.zeros(4, dtype=np.int64)
    for piece, placed_rot in scored:
        hist[(placed_rot - rotations[piece]) % 4] += 1
    best_k = int(np.argmax(hist)) if scored else 0
    out = {"orientation_accuracy": float(hist.max()) / n if scored else 0.0,
           "global_rotation": best_k,
           "n_scored": len(scored),
           "offset_histogram": hist.tolist()}

    if cells is not None:
        grid = {(r, c): int(pid[r, c]) for r in range(rows) for c in range(cols)
                if pid[r, c] >= 0}
        rot_of = {int(pid[r, c]): int(rot[r, c]) for r in range(rows)
                  for c in range(cols) if pid[r, c] >= 0}
        best = 0
        for k in range(4):
            rotated, _ = _rotate_grid(grid, rows, cols, k)
            hit = 0
            for rc, p in rotated.items():
                if cells.get(p) != rc or p not in rotations:
                    continue
                if (rot_of[p] - rotations[p]) % 4 == best_k:
                    hit += 1
            best = max(best, hit)
        out["position_and_orientation_accuracy"] = best / n
    return out


def neighbour_accuracy(assembly, gt, assoc) -> dict:
    """Fraction of true adjacencies reproduced by the arrangement.

    Invariant to any global rotation/translation of the solution, which makes
    it the fairest single number for a jigsaw solver.  Both the *unordered*
    version (the two pieces touch) and the *directed* version (they touch on
    the correct pair of sides) are reported.
    """
    rows, cols = assembly.grid_shape
    pid, rot = assembly.as_arrays()
    cell_of = {}
    for r in range(rows):
        for c in range(cols):
            if pid[r, c] >= 0:
                cell_of[int(pid[r, c])] = (r, c)

    truth_pairs = set()
    gt_cell = {}
    for i, g in enumerate(assoc):
        r, c, _ = gt.placements[g]
        gt_cell[i] = (int(r), int(c))
    inv = {v: k for k, v in gt_cell.items()}
    for (r, c), i in inv.items():
        for d, (dr, dc) in _STEP.items():
            j = inv.get((r + dr, c + dc))
            if j is not None and i < j:
                truth_pairs.add((i, j))

    got = 0
    for (i, j) in truth_pairs:
        if i not in cell_of or j not in cell_of:
            continue
        (r1, c1), (r2, c2) = cell_of[i], cell_of[j]
        if abs(r1 - r2) + abs(c1 - c2) == 1:
            got += 1
    total = max(len(truth_pairs), 1)
    return {"neighbour_accuracy": got / total,
            "n_true_adjacencies": len(truth_pairs),
            "n_recovered": got}


def rotation_accuracy(assembly, descriptions, gt, assoc) -> dict:
    """Fraction of pieces whose orientation matches the ground truth.

    For each placed piece we compare the direction its side 0 faces in the
    arrangement with the direction it faced in the original picture (from
    :func:`gt_side_directions`).  A single global quarter-turn offset -- the
    same for every piece -- is allowed and chosen to maximise the score.
    """
    rows, cols = assembly.grid_shape
    offsets = {0: 0, 1: 0, 2: 0, 3: 0}
    n = 0
    for r in range(rows):
        for c in range(cols):
            pl = assembly.grid[r][c]
            if pl is None:
                continue
            d = descriptions[pl.piece]
            angle = gt.placements[assoc[pl.piece]][2]
            dirs = gt_side_directions(d, angle)
            # the side facing North in the arrangement is `rotation`
            true_dir = dirs[pl.rotation]
            k = DIRECTIONS.index(true_dir)     # 0 if it really was North
            offsets[k] = offsets.get(k, 0) + 1
            n += 1
    best = max(offsets.values()) if n else 0
    return {"rotation_accuracy": best / max(n, 1),
            "n_pieces": n,
            "orientation_histogram": offsets}


def matching_accuracy(table, descriptions, gt, assoc) -> dict:
    """Quality of the compatibility measure *before* any assembly.

    For every side that is a true interior seam we check whether the true
    partner side is ranked first (``top1``) or in the top three (``top3``)
    among all admissible candidates, and record the mean rank.  This isolates
    the matcher from the search.
    """
    n = len(descriptions)
    cell = {}
    dirs = {}
    for i in range(n):
        g = assoc[i]
        r, c, ang = gt.placements[g]
        cell[i] = (int(r), int(c))
        dirs[i] = gt_side_directions(descriptions[i], ang)
    by_cell = {v: k for k, v in cell.items()}

    top1 = top3 = total = 0
    ranks = []
    opp = {"N": "S", "E": "W", "S": "N", "W": "E"}
    for i in range(n):
        r, c = cell[i]
        for s in range(4):
            d = dirs[i][s]
            dr, dc = _STEP[d]
            j = by_cell.get((r + dr, c + dc))
            if j is None:
                continue
            want = opp[d]
            try:
                t = dirs[j].index(want)
            except ValueError:
                continue
            row = table.cost[i, s].reshape(-1)
            if not np.isfinite(row).any():
                continue
            order = np.argsort(row, kind="stable")
            pos = int(np.flatnonzero(order == j * 4 + t)[0])
            ranks.append(pos)
            total += 1
            if pos == 0:
                top1 += 1
            if pos < 3:
                top3 += 1
    return {"n_true_seams": total,
            "top1_accuracy": top1 / max(total, 1),
            "top3_accuracy": top3 / max(total, 1),
            "mean_rank": float(np.mean(ranks)) if ranks else float("nan")}


# ==========================================================================
# image similarity
# ==========================================================================
def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images scaled to ``[0, 1]``."""
    x, y = to_float(a), to_float(b)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    return float(np.mean((x - y) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (``inf`` for identical images)."""
    e = mse(a, b)
    return float("inf") if e <= 0 else float(10.0 * np.log10(1.0 / e))


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index, computed from scratch.

    Local means, variances and covariance are estimated with the library's
    own Gaussian filter, then combined with the standard SSIM expression and
    averaged over the image.
    """
    x, y = to_gray(a), to_gray(b)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    c1, c2 = (k1 ** 2), (k2 ** 2)
    mu_x = gaussian_blur(x, sigma)
    mu_y = gaussian_blur(y, sigma)
    xx = gaussian_blur(x * x, sigma) - mu_x * mu_x
    yy = gaussian_blur(y * y, sigma) - mu_y * mu_y
    xy = gaussian_blur(x * y, sigma) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (xx + yy + c2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def image_metrics(reconstructed: np.ndarray, reference: np.ndarray,
                  allow_rotation: bool = True) -> dict:
    """MSE / PSNR / SSIM between a reconstruction and the original picture.

    The reference is resampled onto the reconstruction's own grid, because
    the reconstruction's cell size is the *measured* median piece body and
    need not equal the reference's to the pixel.

    With ``allow_rotation`` (the default) the four global quarter-turns of
    the reconstruction are tried and the best is reported, together with the
    turn that produced it.  Nothing in a bag of scrambled pieces says which
    way is up, so a puzzle solved sideways is still solved -- the same
    convention :func:`direct_accuracy` uses.
    """
    from .enhancement import resize_bilinear

    rec = to_float(reconstructed)
    ref = to_float(reference)
    if rec.ndim != ref.ndim:
        rec, ref = to_gray(rec), to_gray(ref)

    best = None
    for k in (range(4) if allow_rotation else (0,)):
        a = np.rot90(rec, k)
        b = resize_bilinear(ref, a.shape[:2])
        m = {"mse": mse(a, b), "psnr_db": psnr(a, b), "ssim": ssim(a, b),
             "global_rotation": k}
        if best is None or m["mse"] < best["mse"]:
            best = m
    return best


# ==========================================================================
# report
# ==========================================================================
@dataclass
class EvaluationReport:
    """All metrics of one reconstruction, ready to be written to JSON."""
    name: str = ""
    grid_shape: tuple[int, int] = (0, 0)
    n_pieces: int = 0
    timings: dict = field(default_factory=dict)
    seam: dict = field(default_factory=dict)
    matching: dict = field(default_factory=dict)
    placement: dict = field(default_factory=dict)
    neighbour: dict = field(default_factory=dict)
    rotation: dict = field(default_factory=dict)
    image: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grid_shape"] = list(self.grid_shape)
        return d


def save_report(report: EvaluationReport, path: str) -> str:
    """Write an :class:`EvaluationReport` to ``path`` as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=default)
    return path
