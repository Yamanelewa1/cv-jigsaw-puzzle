"""Milestone 2 / task 1 -- dataset preparation.

Generating labelled side pairs
------------------------------
A training example is a *pair of sides* with a label: 1 if the two sides are
genuinely adjacent in the finished puzzle, 0 if they are not.  Producing
those labels needs to know, for every side of every piece, which grid
direction it faced before the pieces were shuffled -- and that is exactly
what the dataset photographs do **not** record.  Their annotations give each
piece's identity, which fixes which *pieces* are neighbours but not which of
a piece's four sides touches which, because the rotation each piece happens
to be lying at is unknown.

Training therefore uses puzzles cut by :func:`src.evaluation.generate_puzzle`,
where the answer is known exactly, and every sample is passed through the
**same** Milestone 1 pipeline the classical matcher uses (scatter, segment,
trace, describe).  The networks consequently see the same segmentation noise,
corner-localisation error and colour sampling as the classical method, so the
comparison in the report is between matchers rather than between front ends.
The real photographs are then used, unchanged, for evaluation.

Sampling
--------
Positives are all true seams of a puzzle.  Negatives are drawn from the
*admissible* pairs only -- a tab facing a blank on another piece -- because a
model that learns merely to reject flat-to-flat or tab-to-tab pairs would
score well while being useless: the classical matcher already excludes those
for free.  :data:`NEGATIVES_PER_POSITIVE` of them are drawn per positive, half
of them from the *hardest* available candidates (those the classical measure
likes best) so the networks spend their capacity where the problem is.

Splitting
---------
Splits are by **puzzle**, never by pair: two sides of one seam must not end up
on opposite sides of the split, and pieces from one puzzle share a source
picture.  The same split object is handed to both models.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, field

import numpy as np

from .. import evaluation as ev
from .. import segmentation as seg
from ..contour_extraction import extract_pieces
from ..edge_matching import build_compatibility
from ..piece_description import SIDE_BLANK, SIDE_TAB, describe_pieces
from .features import relative_orientation, side_strip, side_vector

__all__ = [
    "NEGATIVES_PER_POSITIVE",
    "PuzzleSample",
    "SplitSpec",
    "build_puzzle_sample",
    "generate_dataset",
    "PairDataset",
    "augment_strip",
]

#: How many negative pairs are drawn for each positive one.
NEGATIVES_PER_POSITIVE = 6
#: Fraction of those negatives taken from the hardest candidates.
HARD_NEGATIVE_FRACTION = 0.5

_STEP = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
_OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}


@dataclass
class PuzzleSample:
    """One puzzle, described, with its true side-to-side correspondences."""
    descriptions: list
    positives: list                      # [(i, s, j, t), ...] true seams
    cells: dict                          # piece index -> (row, col)
    grid_shape: tuple
    source: str = ""
    #: classical cost table, used to mine hard negatives and as a baseline
    table: object | None = None

    @property
    def n_pieces(self) -> int:
        return len(self.descriptions)

    def side_types(self) -> list:
        return [s.type for d in self.descriptions for s in d.sides]

    def true_rotations(self) -> dict:
        """``piece -> the side index that faced North in the finished puzzle``.

        This is the orientation ground truth the brief asks the evaluation to
        score against, and it does not need to be stored: it is implied by the
        labels already here.  A piece's rotation is defined as the index of the
        side facing North, and side ``s`` faces direction ``d`` exactly when
        ``s = (rotation + DIRECTIONS.index(d)) % 4``.  Every true seam
        ``(i, s, j, t)`` therefore pins both pieces down, because ``cells``
        says which way the seam runs::

            rotation_i = (s - DIRECTIONS.index(d)) % 4

        Deriving it rather than recording it also means caches written before
        this method existed stay valid.  Pieces whose every seam was filtered
        out are simply absent from the result, so the caller scores what is
        known instead of guessing.
        """
        from ..piece_description import DIRECTIONS

        step_dir = {(-1, 0): "N", (0, 1): "E", (1, 0): "S", (0, -1): "W"}
        votes: dict = {}
        for (i, s, j, t) in self.positives:
            if i not in self.cells or j not in self.cells:
                continue
            (ri, ci), (rj, cj) = self.cells[i], self.cells[j]
            d = step_dir.get((rj - ri, cj - ci))
            if d is None:                     # not a grid-adjacent pair
                continue
            opp = {"N": "S", "E": "W", "S": "N", "W": "E"}[d]
            votes.setdefault(i, []).append((s - DIRECTIONS.index(d)) % 4)
            votes.setdefault(j, []).append((t - DIRECTIONS.index(opp)) % 4)

        out = {}
        for piece, vs in votes.items():
            # every seam of a piece must imply the same rotation; take the
            # majority so one mislabelled seam cannot flip the answer
            out[piece] = int(max(set(vs), key=vs.count))
        return out


@dataclass
class SplitSpec:
    """Which generated puzzles belong to train / validation / test."""
    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    test: list = field(default_factory=list)

    def counts(self) -> dict:
        return {"train": len(self.train), "val": len(self.val),
                "test": len(self.test)}


# ==========================================================================
# building one labelled puzzle
# ==========================================================================
def build_puzzle_sample(rows: int, cols: int, seed: int,
                        cell: int = 110, rotate: bool = True,
                        with_table: bool = True) -> PuzzleSample | None:
    """Cut, scatter, segment and describe one puzzle, and label its seams.

    Returns ``None`` if segmentation did not recover every piece, so that a
    training set never contains a puzzle whose labels are unreliable.
    """
    src = ev.synthetic_source_image(cell * rows, cell * cols, seed=10_000 + seed)
    scrambled, gt = ev.generate_puzzle(src, rows=rows, cols=cols,
                                       rotate=rotate, seed=seed)

    mask = seg.foreground_mask(scrambled, "background", open_radius=1,
                               close_radius=1)
    labels, _ = seg.connected_components(mask)
    labels, stats = seg.filter_components(labels, min_area_ratio=0.3)
    pieces = extract_pieces(scrambled, labels, stats=stats)
    if len(pieces) != rows * cols:
        return None
    descs = describe_pieces(pieces)

    assoc = ev.associate_with_ground_truth(pieces, gt)
    if len(set(assoc)) != len(pieces):
        return None

    cells, dirs = {}, {}
    for i in range(len(descs)):
        r, c, angle = gt.placements[assoc[i]]
        cells[i] = (int(r), int(c))
        dirs[i] = ev.gt_side_directions(descs[i], angle)
    by_cell = {v: k for k, v in cells.items()}

    positives = []
    for i in range(len(descs)):
        r, c = cells[i]
        for s in range(4):
            d = dirs[i][s]
            j = by_cell.get((r + _STEP[d][0], c + _STEP[d][1]))
            if j is None:
                continue
            try:
                t = dirs[j].index(_OPPOSITE[d])
            except ValueError:
                continue
            # a genuine seam must be an admissible tab/blank pair; if the
            # geometry says otherwise the description is wrong and the label
            # would poison the training set
            ta, tb = descs[i].sides[s].type, descs[j].sides[t].type
            if {ta, tb} != {SIDE_TAB, SIDE_BLANK}:
                continue
            if i < j:
                positives.append((i, s, j, t))

    table = build_compatibility(descs) if with_table else None
    return PuzzleSample(descriptions=descs, positives=positives, cells=cells,
                        grid_shape=(rows, cols), table=table,
                        source=f"{rows}x{cols}_seed{seed}")


# ==========================================================================
# a whole dataset
# ==========================================================================
def generate_dataset(n_puzzles: int = 60, sizes=((3, 4), (4, 5), (5, 7)),
                     seed: int = 0, cache_dir: str | None = None,
                     verbose: bool = True):
    """Generate ``n_puzzles`` labelled puzzles and split them by puzzle.

    Sizes are cycled so the training set contains a range of piece counts,
    which stops the models from learning anything specific to one grid.
    Results are cached on disk because building a puzzle costs a second or
    two of segmentation and description.

    Returns ``(samples, split)``.
    """
    key = f"{n_puzzles}-{sizes}-{seed}"
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    path = (os.path.join(cache_dir, f"puzzles_{digest}.pkl")
            if cache_dir else None)
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            samples, split = pickle.load(fh)
        if verbose:
            print(f"  loaded {len(samples)} cached puzzles from {path}")
        return samples, split

    rng = np.random.default_rng(seed)
    samples, attempt = [], 0
    while len(samples) < n_puzzles and attempt < n_puzzles * 3:
        r, c = sizes[len(samples) % len(sizes)]
        s = build_puzzle_sample(r, c, seed=int(rng.integers(1, 10 ** 6)))
        attempt += 1
        if s is None or not s.positives:
            continue
        samples.append(s)
        if verbose and len(samples) % 10 == 0:
            print(f"  {len(samples)}/{n_puzzles} puzzles", flush=True)

    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_tr = int(round(0.70 * len(idx)))
    n_va = int(round(0.15 * len(idx)))
    split = SplitSpec(train=idx[:n_tr].tolist(),
                      val=idx[n_tr:n_tr + n_va].tolist(),
                      test=idx[n_tr + n_va:].tolist())
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump((samples, split), fh)
    return samples, split


# ==========================================================================
# augmentation
# ==========================================================================
def augment_strip(strip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric augmentation of one ``(4, S, D)`` side strip.

    Only the three colour channels are touched; the shape channel is
    geometry and must not be recoloured.  The four effects match the ones the
    brief lists -- illumination, contrast, colour and noise -- and each is
    applied to the *pair* consistently by the caller when it should be (a
    seam is lit by one lamp), or independently when it should not (two pieces
    lying apart on a table are not).
    """
    out = strip.copy()
    col = out[:3]
    col = col * rng.uniform(0.85, 1.15)                       # illumination
    mean = col.mean()
    col = mean + (col - mean) * rng.uniform(0.8, 1.25)        # contrast
    col = col * rng.uniform(0.93, 1.07, size=(3, 1, 1))       # colour balance
    col = col + rng.normal(0.0, 0.02, size=col.shape)         # sensor noise
    out[:3] = np.clip(col, 0.0, 1.0)
    return out.astype(np.float32)


# ==========================================================================
# pair sampling
# ==========================================================================
class PairDataset:
    """Labelled side pairs drawn from a list of :class:`PuzzleSample`.

    ``mode="strip"`` yields the tensors the Siamese CNN wants; ``"vector"``
    yields the flat descriptors used to build graph nodes.  Negatives are
    resampled every epoch when ``resample=True``, so the network sees far
    more of the negative space than one fixed draw would give.
    """

    def __init__(self, samples, indices, mode: str = "strip",
                 negatives_per_positive: int = NEGATIVES_PER_POSITIVE,
                 augment: bool = False, seed: int = 0):
        self.samples = samples
        self.indices = list(indices)
        self.mode = mode
        self.k = int(negatives_per_positive)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        # A side's tensor never changes -- only which sides get paired does.
        # Computing them once instead of once per epoch is the difference
        # between a few seconds and a few minutes per epoch.
        self._cache: dict = {}
        self.pairs: list = []
        self.resample()

    def _features(self, pi: int):
        """Cached ``(forward, reversed)`` tensors for every side of a puzzle."""
        hit = self._cache.get(pi)
        if hit is not None:
            return hit
        sample = self.samples[pi]
        sides = [s for d in sample.descriptions for s in d.sides]
        if self.mode == "vector":
            fwd = np.stack([side_vector(s) for s in sides])
            rev = fwd
        else:
            fwd = np.stack([side_strip(s, reverse=False) for s in sides])
            rev = np.stack([side_strip(s, reverse=True) for s in sides])
        self._cache[pi] = (fwd, rev)
        return fwd, rev

    # ------------------------------------------------------------------
    def _admissible(self, sample):
        types = sample.side_types()
        n = sample.n_pieces
        arr = np.array(types)
        ok = ((arr == SIDE_TAB)[:, None] & (arr == SIDE_BLANK)[None, :]) | \
             ((arr == SIDE_BLANK)[:, None] & (arr == SIDE_TAB)[None, :])
        piece_of = np.repeat(np.arange(n), 4)
        ok &= piece_of[:, None] != piece_of[None, :]
        return ok

    def resample(self):
        """Draw a fresh set of negatives for every positive."""
        self.pairs = []
        for pi in self.indices:
            sample = self.samples[pi]
            ok = self._admissible(sample)
            pos = {(i * 4 + s, j * 4 + t) for (i, s, j, t) in sample.positives}
            pos |= {(b, a) for (a, b) in pos}
            n_sides = ok.shape[0]

            cost = None
            if sample.table is not None:
                cost = sample.table.cost.reshape(n_sides, n_sides)

            for (i, s, j, t) in sample.positives:
                a, b = i * 4 + s, j * 4 + t
                self.pairs.append((pi, a, b, 1))

                cands = np.flatnonzero(ok[a])
                cands = np.array([c for c in cands if (a, c) not in pos])
                if cands.size == 0:
                    continue
                n_hard = int(round(self.k * HARD_NEGATIVE_FRACTION))
                chosen = []
                if cost is not None and n_hard > 0:
                    order = cands[np.argsort(cost[a, cands])]
                    chosen.extend(order[:n_hard].tolist())
                rest = self.k - len(chosen)
                if rest > 0:
                    pool = np.setdiff1d(cands, np.array(chosen, dtype=cands.dtype))
                    if pool.size:
                        chosen.extend(self.rng.choice(
                            pool, size=min(rest, pool.size), replace=False).tolist())
                for c in chosen:
                    self.pairs.append((pi, a, int(c), 0))
        self.rng.shuffle(self.pairs)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def _side(self, sample, flat_index):
        return sample.descriptions[flat_index // 4].sides[flat_index % 4]

    def get(self, k: int):
        """One example: ``(feature_a, feature_b, label, relative_orientation)``."""
        pi, a, b, label = self.pairs[k]
        fwd, rev = self._features(pi)
        fa = fwd[a]
        fb = rev[b]                               # mating sides run opposite
        if self.mode != "vector" and self.augment:
            fa = augment_strip(fa, self.rng)
            fb = augment_strip(fb, self.rng)
        rel = relative_orientation(a % 4, b % 4)
        return fa, fb, float(label), rel

    def arrays(self):
        """The whole split as stacked arrays ``(A, B, y, rel)``.

        Built by gathering from the per-puzzle caches rather than one example
        at a time, so a fresh epoch of negatives costs an index operation.
        """
        if not self.pairs:
            return (np.zeros((0, 4, 1, 1), np.float32),
                    np.zeros((0, 4, 1, 1), np.float32),
                    np.zeros(0, np.float32), np.zeros(0, np.int64))
        by_puzzle: dict = {}
        for k, (pi, a, b, lab) in enumerate(self.pairs):
            by_puzzle.setdefault(pi, []).append((k, a, b))

        first = self._features(self.pairs[0][0])[0]
        shape = first.shape[1:]
        n = len(self.pairs)
        A = np.empty((n,) + shape, dtype=np.float32)
        B = np.empty((n,) + shape, dtype=np.float32)
        for pi, items in by_puzzle.items():
            fwd, rev = self._features(pi)
            ks = np.array([i[0] for i in items])
            A[ks] = fwd[np.array([i[1] for i in items])]
            B[ks] = rev[np.array([i[2] for i in items])]

        if self.mode != "vector" and self.augment:
            # augment the batch in one pass: a per-example gain, contrast,
            # colour balance and noise on the colour channels only
            r = self.rng
            for arr in (A, B):
                col = arr[:, :3]
                col *= r.uniform(0.85, 1.15, size=(n, 1, 1, 1))
                mean = col.mean(axis=(1, 2, 3), keepdims=True)
                col[:] = mean + (col - mean) * r.uniform(0.8, 1.25,
                                                         size=(n, 1, 1, 1))
                col *= r.uniform(0.93, 1.07, size=(n, 3, 1, 1))
                col += r.normal(0.0, 0.02, size=col.shape)
                np.clip(col, 0.0, 1.0, out=col)

        y = np.array([p[3] for p in self.pairs], dtype=np.float32)
        rel = np.array([relative_orientation(p[1] % 4, p[2] % 4)
                        for p in self.pairs], dtype=np.int64)
        return A, B, y, rel

    def balance(self) -> dict:
        pos = sum(1 for p in self.pairs if p[3] == 1)
        return {"n_pairs": len(self.pairs), "positive": pos,
                "negative": len(self.pairs) - pos,
                "positive_fraction": pos / max(len(self.pairs), 1)}
