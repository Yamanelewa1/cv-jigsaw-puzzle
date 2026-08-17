"""Milestone 2 / task 5 -- evaluations that actually discriminate the methods.

On the standard test split all three matchers reconstruct every puzzle
perfectly, so the headline comparison is saturated and cannot say which is
better.  A ceiling is not a result, so two harder conditions are added.

**Low-texture puzzles.**  Milestone 1 established that colour carries most of
the discriminative power and that the dataset's own puzzle fails precisely
because its picture is nearly uniform.  :func:`low_texture_source` reproduces
that difficulty on demand by fading a generated picture towards flat grey, so
the methods can be compared as the signal is withdrawn.

**The real photographs.**  The ultimate test, and a genuine domain shift: the
models are trained on generated puzzles and asked about photographs of a real
jigsaw.  Side-level labels do not exist there (see :mod:`src.ml.dataset`), so
only the reconstruction metrics are available -- which are the ones that
matter anyway.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .. import evaluation as ev
from .. import segmentation as seg
from ..assembly import assemble
from ..contour_extraction import extract_pieces
from ..edge_matching import build_compatibility
from ..enhancement import to_uint8
from ..piece_description import describe_pieces
from .dataset import PuzzleSample, build_puzzle_sample
from .infer import gnn_table, siamese_table

__all__ = [
    "low_texture_source",
    "build_low_texture_sample",
    "evaluate_texture_sweep",
    "real_scenes",
    "build_real_sample",
    "evaluate_on_real_photographs",
]

#: Class names of the Roboflow export, in class-index order.
DATASET_CLASS_NAMES = ['1', '10', '11', '12', '13', '14', '15', '16', '17',
                       '18', '19', '2', '20', '21', '22', '23', '24', '25',
                       '26', '27', '28', '29', '3', '30', '31', '32', '33',
                       '34', '35', '4', '5', '6', '7', '8', '9']
DATASET_GRID = (5, 7)

#: Matcher settings Milestone 1 uses for **these photographs**
#: (``main.DATASET_SOLVER``).  The pieces lie all over a table under uneven
#: light, so two strips facing each other across a true seam differ by an
#: offset and a gain even where the picture continues perfectly; removing each
#: strip's own mean and spread is what Milestone 1 measured as lifting its
#: top-1 rate from 0.193 to 0.251.
#:
#: The classical baseline must be compared in the configuration its own
#: milestone runs, or the comparison measures the configuration rather than
#: the method.  Generated puzzles are lit uniformly by construction and take
#: the Milestone 1 default (``colour_norm="none"``), which is why this is
#: applied to the photographs only.
REAL_PHOTO_MATCHER = {"colour_norm": "meanstd"}


def low_texture_source(height: int, width: int, seed: int,
                       texture: float = 1.0) -> np.ndarray:
    """A generated picture faded towards flat grey.

    ``texture = 1`` is the ordinary picture; ``texture = 0`` is uniform grey,
    where no photometric measure can say anything at all and only the cut
    geometry remains.  Intermediate values interpolate, so the methods can be
    compared as the picture's information is withdrawn -- the axis along
    which the real dataset's puzzle is hard.
    """
    img = ev.synthetic_source_image(height, width, seed=seed).astype(np.float64)
    grey = np.full_like(img, img.mean())
    return to_uint8((texture * img + (1.0 - texture) * grey) / 255.0)


def build_low_texture_sample(rows: int, cols: int, seed: int,
                             texture: float, cell: int = 110):
    """A labelled puzzle whose picture has been faded to ``texture``."""
    src = low_texture_source(cell * rows, cell * cols, seed=10_000 + seed,
                             texture=texture)
    scrambled, gt = ev.generate_puzzle(src, rows=rows, cols=cols,
                                       rotate=True, seed=seed)
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

    step = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
    opp = {"N": "S", "E": "W", "S": "N", "W": "E"}
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
            j = by_cell.get((r + step[d][0], c + step[d][1]))
            if j is None:
                continue
            try:
                t = dirs[j].index(opp[d])
            except ValueError:
                continue
            if i < j:
                positives.append((i, s, j, t))
    return PuzzleSample(descriptions=descs, positives=positives, cells=cells,
                        grid_shape=(rows, cols),
                        table=build_compatibility(descs),
                        source=f"{rows}x{cols}_tex{texture:.2f}_seed{seed}")


def _methods(siamese, gnn, matcher: dict | None = None):
    """The three methods, with the classical one in ``matcher``'s configuration.

    ``matcher`` is the keyword configuration Milestone 1 would use on this kind
    of input -- ``{}`` for generated puzzles, :data:`REAL_PHOTO_MATCHER` for the
    photographs.  The graph network's candidate shortlist is drawn from the same
    table, so all three methods see the same classical evidence.
    """
    kw = matcher or {}
    out = [("classical", lambda s: build_compatibility(s.descriptions, **kw))]
    if siamese is not None:
        out.append(("siamese_cnn", lambda s: siamese_table(siamese, s.descriptions)))
    if gnn is not None:
        out.append(("graph_nn", lambda s: gnn_table(gnn, s.descriptions, s.table)))
    return out


def evaluate_texture_sweep(siamese, gnn, textures=(1.0, 0.6, 0.35, 0.2, 0.1),
                           rows: int = 4, cols: int = 5, n_puzzles: int = 4,
                           seed: int = 5000, verbose: bool = True) -> dict:
    """Reconstruction accuracy as the picture's texture is faded out."""
    out = {}
    for texture in textures:
        samples = []
        for k in range(n_puzzles):
            s = build_low_texture_sample(rows, cols, seed + k, texture)
            if s is not None:
                samples.append(s)
        if not samples:
            continue
        row = {}
        for name, fn in _methods(siamese, gnn):
            nbrs, poss = [], []
            for s in samples:
                table = fn(s)
                asm = assemble(s.descriptions, table, s.grid_shape)
                nbrs.append(ev.neighbour_accuracy_from_cells(
                    asm, s.cells)["neighbour_accuracy"])
                poss.append(ev.position_accuracy_from_cells(
                    asm, s.cells)["position_accuracy"])
            row[name] = {"neighbour_accuracy": float(np.mean(nbrs)),
                         "position_accuracy": float(np.mean(poss)),
                         "perfect": int(sum(1 for p in poss if p >= 0.999)),
                         "n_puzzles": len(samples)}
        out[f"texture_{texture:.2f}"] = row
        if verbose:
            print("  texture %.2f  " % texture + "  ".join(
                f"{k} nbr {v['neighbour_accuracy']:.2f}" for k, v in row.items()),
                flush=True)
    return out


# ==========================================================================
# the real photographs
# ==========================================================================
def _real_cells(pieces, boxes, shape):
    rows, cols = DATASET_GRID
    px = []
    h, w = shape[:2]
    for pid, cx, cy, bw, bh in boxes:
        px.append((pid, cy * h - bh * h / 2, cx * w - bw * w / 2,
                   cy * h + bh * h / 2, cx * w + bw * w / 2))
    claimed = [False] * len(px)
    out = {}
    for p in pieces:
        cy = p.bbox[0] + p.centroid[0]
        cx = p.bbox[1] + p.centroid[1]
        for k, (pid, y0, x0, y1, x1) in enumerate(px):
            if claimed[k]:
                continue
            if y0 <= cy <= y1 and x0 <= cx <= x1:
                claimed[k] = True
                if 1 <= pid <= rows * cols:
                    out[p.index] = ((pid - 1) // cols, (pid - 1) % cols)
                break
    return out


def real_scenes(detection_dir: str, limit: int | None = None) -> list:
    """``(image path, label path)`` for the photographs that show a full puzzle.

    Scenes with fewer than 35 annotated pieces are skipped: the jigsaw has 35,
    so a shorter label file means pieces are missing or occluded and the
    arrangement cannot be scored against the grid.
    """
    import glob

    scenes = []
    for split in ("train", "valid"):
        for lab in sorted(glob.glob(os.path.join(detection_dir, "labels",
                                                 split, "*.txt"))):
            if sum(1 for l in open(lab) if l.strip()) < 35:
                continue
            base = os.path.splitext(os.path.basename(lab))[0]
            img = os.path.join(detection_dir, "images", split, base + ".jpg")
            if os.path.exists(img):
                scenes.append((img, lab))
    return scenes[:limit] if limit else scenes


def build_real_sample(img_path: str, lab_path: str, max_side: int = 1920):
    """Segment, describe and label one dataset photograph.

    Returns ``None`` when the annotations cannot be matched to the segmented
    pieces, in which case there is nothing to score the arrangement against.
    """
    from PIL import Image

    im = Image.open(img_path).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)),
                       Image.BILINEAR)
    img = np.asarray(im, np.uint8)

    boxes = []
    for line in open(lab_path):
        q = line.split()
        if len(q) < 5:
            continue
        boxes.append((int(DATASET_CLASS_NAMES[int(q[0])]), float(q[1]),
                      float(q[2]), float(q[3]), float(q[4])))

    mask = seg.foreground_mask(img, "background", open_radius=2,
                               close_radius=2)
    labels, _ = seg.connected_components(mask)
    labels, stats = seg.filter_components(labels, min_area_ratio=0.45)
    labels, n_split = seg.split_touching(labels)
    if n_split:
        labels, stats = seg.filter_components(labels, min_area_ratio=0.45,
                                              max_area_ratio=1.7)
    pieces = extract_pieces(img, labels, stats=stats)
    descs = describe_pieces(pieces)
    cells = _real_cells(pieces, boxes, img.shape)
    if not cells:
        return None
    return PuzzleSample(descriptions=descs, positives=[], cells=cells,
                        grid_shape=DATASET_GRID,
                        table=build_compatibility(descs, **REAL_PHOTO_MATCHER),
                        source=os.path.basename(img_path))


def evaluate_on_real_photographs(siamese, gnn, detection_dir: str,
                                 limit: int = 8, max_side: int = 1920,
                                 verbose: bool = True) -> dict:
    """All three methods on the actual dataset photographs.

    This is a genuine domain shift for the networks -- trained on generated
    puzzles, asked about photographs of a real jigsaw -- and the report treats
    it as such.  Only reconstruction metrics are available, because the
    photographs carry no side-level ground truth.
    """
    scenes = real_scenes(detection_dir, limit)
    if not scenes:
        return {}

    methods = _methods(siamese, gnn, REAL_PHOTO_MATCHER)
    rows_out = {name: [] for name, _ in methods}
    for img_path, lab_path in scenes:
        sample = build_real_sample(img_path, lab_path, max_side)
        if sample is None:
            continue
        descs, cells = sample.descriptions, sample.cells
        for name, fn in methods:
            t0 = time.perf_counter()
            table = fn(sample)
            asm = assemble(descs, table, DATASET_GRID)
            rows_out[name].append({
                "neighbour_accuracy": ev.neighbour_accuracy_from_cells(
                    asm, cells)["neighbour_accuracy"],
                "position_accuracy": ev.position_accuracy_from_cells(
                    asm, cells)["position_accuracy"],
                "quality": ev.seam_quality(asm, table)["quality"],
                "seconds": time.perf_counter() - t0,
            })
        if verbose:
            print("  %-26s " % os.path.basename(img_path)[:26] + "  ".join(
                f"{k} {v[-1]['neighbour_accuracy']:.2f}"
                for k, v in rows_out.items() if v), flush=True)

    summary = {}
    for name, rows in rows_out.items():
        if not rows:
            continue
        summary[name] = {
            "neighbour_accuracy": float(np.mean([r["neighbour_accuracy"] for r in rows])),
            "position_accuracy": float(np.mean([r["position_accuracy"] for r in rows])),
            "quality": float(np.mean([r["quality"] for r in rows])),
            "seconds_per_puzzle": float(np.mean([r["seconds"] for r in rows])),
            "n_images": len(rows),
        }
    return summary
