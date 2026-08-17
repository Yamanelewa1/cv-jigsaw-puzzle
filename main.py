#!/usr/bin/env python
"""CSE480 Machine Vision -- Milestone 1 command line entry point.

The headline result is the **real dataset**: every photograph in
``detection/`` that shows all 35 pieces of the jigsaw scattered on the cloth::

    python main.py                      # == python main.py --run dataset
    python main.py --run dataset --limit 10

Solve one arbitrary scrambled puzzle image::

    python main.py --input some_photo.jpg --grid 5x7

Regenerate every figure, result and metric used in the report::

    python main.py --run all

Individual stages (all of which run on real photographs)::

    python main.py --run stages
    python main.py --run enhancement | thresholding | edges | segmentation
                        | matching | weights

``--run validate`` is the one deliberately synthetic step: reconstruction
accuracy cannot be measured without an answer key, and the dataset carries
none, so puzzles with known ground truth are cut from a source picture to
measure it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from src import assembly as asm                               # noqa: E402
from src import contour_extraction as ce                      # noqa: E402
from src import edge_detection as ed                          # noqa: E402
from src import edge_matching as em                           # noqa: E402
from src import enhancement as enh                            # noqa: E402
from src import evaluation as ev                              # noqa: E402
from src import piece_description as pdsc                     # noqa: E402
from src import segmentation as seg                           # noqa: E402
from src import thresholding as th                            # noqa: E402
from src.enhancement import resize_bilinear, to_float, to_uint8   # noqa: E402
from src import PuzzleSolver                                  # noqa: E402



# ==========================================================================
# Image IO and visualisation helpers
#
# Pillow is used *only* to decode and encode JPEG/PNG files; every pixel
# operation lives in the ``src`` package and is implemented from scratch.
# ==========================================================================
def ensure_dir(path: str) -> str:
    """Create ``path`` (a directory) if needed and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def imread(path: str, gray: bool = False, max_side: int | None = None) -> np.ndarray:
    """Read an image as ``uint8``.

    Parameters
    ----------
    gray:
        Decode as single-channel instead of RGB.
    max_side:
        If given, the image is downscaled (aspect preserved) so that its
        longest side is at most ``max_side`` pixels.  Useful to keep the
        1920x1080 dataset photos manageable.
    """
    img = Image.open(path)
    img = img.convert("L" if gray else "RGB")
    if max_side is not None and max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        new_size = (max(1, int(round(img.size[0] * scale))),
                    max(1, int(round(img.size[1] * scale))))
        img = img.resize(new_size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def imwrite(path: str, image: np.ndarray) -> str:
    """Write ``image`` (uint8, float in [0, 1], or bool) to ``path``."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    arr = np.asarray(image)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        arr = to_uint8(arr)
    Image.fromarray(arr).save(path)
    return path


# --------------------------------------------------------------------------
# visualisation
# --------------------------------------------------------------------------
def colorize_labels(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Map an integer label image to random RGB colours (label 0 -> black)."""
    rng = np.random.default_rng(seed)
    n = int(labels.max()) + 1
    palette = rng.integers(60, 256, size=(max(n, 1), 3), dtype=np.int32)
    palette[0] = 0
    return palette[labels].astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color: Sequence[int] = (255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    """Blend a boolean ``mask`` over ``image`` in ``color``."""
    img = to_float(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    col = np.asarray(color, dtype=np.float64) / 255.0
    out = img.copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * img[m] + alpha * col
    return to_uint8(out)


def draw_points(image: np.ndarray, points: Iterable[Sequence[float]],
                color: Sequence[int] = (255, 0, 0), radius: int = 2) -> np.ndarray:
    """Stamp small squares of ``color`` at ``points`` given as ``(y, x)``."""
    out = np.array(to_uint8(to_float(image)), dtype=np.uint8, copy=True)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=2)
    h, w = out.shape[:2]
    col = np.asarray(color, dtype=np.uint8)
    for p in points:
        y, x = int(round(float(p[0]))), int(round(float(p[1])))
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        if y0 < y1 and x0 < x1:
            out[y0:y1, x0:x1] = col
    return out


def make_grid(images: Sequence[np.ndarray], cols: int = 4, pad_px: int = 4,
              bg: int = 255) -> np.ndarray:
    """Tile images (any sizes) into a single contact sheet."""
    imgs = []
    for im in images:
        a = to_uint8(to_float(im))
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=2)
        imgs.append(a)
    if not imgs:
        return np.zeros((1, 1, 3), np.uint8)
    cell_h = max(i.shape[0] for i in imgs)
    cell_w = max(i.shape[1] for i in imgs)
    rows = (len(imgs) + cols - 1) // cols
    out = np.full((rows * (cell_h + pad_px) + pad_px,
                   cols * (cell_w + pad_px) + pad_px, 3), bg, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y = pad_px + r * (cell_h + pad_px)
        x = pad_px + c * (cell_w + pad_px)
        out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


# ==========================================================================
# The provided dataset (Roboflow YOLO export in ``detection/``)
#
# The annotations are not a segmentation ground truth, but the boxes are
# enough to score the piece-detection half of the pipeline: how many of the
# pieces present in a photograph the segmentation actually isolates.
# ==========================================================================
class Box(tuple):
    """``(cls, cx, cy, w, h)`` in normalised image coordinates."""
    __slots__ = ()

    @property
    def cls(self) -> int:
        return int(self[0])

    def pixel_box(self, shape) -> tuple[int, int, int, int]:
        """``(y0, x0, y1, x1)`` in pixels for an image of ``shape``."""
        h, w = shape[:2]
        cx, cy, bw, bh = self[1] * w, self[2] * h, self[3] * w, self[4] * h
        return (int(round(cy - bh / 2)), int(round(cx - bw / 2)),
                int(round(cy + bh / 2)), int(round(cx + bw / 2)))


def load_yolo_labels(path: str) -> list[Box]:
    """Read one YOLO ``.txt`` annotation file."""
    out: list[Box] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            out.append(Box((int(parts[0]), float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))))
    return out


def find_scenes(root: str, split: str = "train", min_objects: int = 35,
                limit: int | None = None) -> list[tuple[str, str, int]]:
    """Locate photographs containing at least ``min_objects`` annotated pieces.

    Returns ``[(image_path, label_path, n_objects), ...]`` sorted by name.
    """
    lab_dir = os.path.join(root, "labels", split)
    img_dir = os.path.join(root, "images", split)
    out = []
    for lp in sorted(glob.glob(os.path.join(lab_dir, "*.txt"))):
        n = len(load_yolo_labels(lp))
        if n < min_objects:
            continue
        base = os.path.splitext(os.path.basename(lp))[0]
        for ext in (".jpg", ".jpeg", ".png"):
            ip = os.path.join(img_dir, base + ext)
            if os.path.exists(ip):
                out.append((ip, lp, n))
                break
        if limit and len(out) >= limit:
            break
    return out


def detection_metrics(stats, boxes: list[Box], shape) -> dict:
    """Score segmented components against the annotated boxes.

    A component counts as a hit when its centroid falls inside an annotated
    box and that box has not already been claimed (so two components landing
    on one piece count as one hit and one false positive).  Reported as
    recall (fraction of annotated pieces isolated), precision (fraction of
    components that are pieces) and the resulting F1.
    """
    px = [b.pixel_box(shape) for b in boxes]
    claimed = [False] * len(px)
    hits = 0
    for st in stats:
        cy, cx = st.centroid
        for k, (y0, x0, y1, x1) in enumerate(px):
            if claimed[k]:
                continue
            if y0 <= cy <= y1 and x0 <= cx <= x1:
                claimed[k] = True
                hits += 1
                break
    n_boxes = max(len(px), 1)
    n_comp = max(len(stats), 1)
    recall = hits / n_boxes
    precision = hits / n_comp
    f1 = (2 * recall * precision / max(recall + precision, 1e-9))
    return {"n_annotated": len(px), "n_components": len(stats),
            "n_matched": hits, "recall": recall, "precision": precision,
            "f1": f1}


# ==========================================================================
# Scene visualisation
# ==========================================================================
def render_scattered_overview(image: np.ndarray, descriptions, labels=None):
    """Annotated overview of the input: contours, corners and side types."""
    out = to_uint8(to_float(image))
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=2)
    colours = {"tab": (0, 230, 0), "blank": (0, 140, 255), "flat": (255, 220, 0)}
    for d in descriptions:
        y0, x0 = d.piece.bbox[0], d.piece.bbox[1]
        for s in d.sides:
            pts = s.points[::3] + np.array([y0, x0])
            out = draw_points(out, pts, colours[s.type], 1)
        out = draw_points(out, d.corners + np.array([y0, x0]), (255, 0, 0), 3)
    return out


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
DETECTION = os.path.join(ROOT, "detection")

R_ENH = os.path.join(RESULTS, "enhanced_images")
R_MASK = os.path.join(RESULTS, "masks")
R_CONT = os.path.join(RESULTS, "contours")
R_EDGE = os.path.join(RESULTS, "edge_visualisations")
R_REC = os.path.join(RESULTS, "reconstructed_images")
R_EVAL = os.path.join(RESULTS, "evaluation_results")

#: Puzzles used by the ``validate`` run to show the assembly stage is
#: correct when its input is clean.
BENCHMARK = [
    # (rows, cols, seed, rotated)
    (2, 3, 11, True),
    (3, 4, 12, True),
    (3, 4, 13, True),
    (4, 5, 14, True),
    (4, 6, 15, True),
    (5, 7, 16, True),
    (5, 7, 17, False),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _dirs():
    for d in (DATA, os.path.join(DATA, "input"),
              os.path.join(DATA, "ground_truth"),
              os.path.join(DATA, "sample_pieces"),
              R_ENH, R_MASK, R_CONT, R_EDGE, R_REC, R_EVAL):
        ensure_dir(d)


def _sample_photo(max_side: int = 1280):
    """A real scattered-pieces photograph from the provided dataset."""
    scenes = find_scenes(DETECTION, "train", min_objects=35, limit=1)
    if not scenes:
        return None, None
    return imread(scenes[0][0], max_side=max_side), scenes[0][0]


def _synthetic_puzzle(rows=3, cols=4, seed=12, rotate=True, cell=120):
    """A puzzle cut from a generated picture, with its answer key.

    Used only where a *known* answer is needed: the ``validate`` run, which
    shows the assembly stage is correct on clean input, and the matching
    study.  Nothing here is written into ``data/``.
    """
    src = ev.synthetic_source_image(cell * rows, cell * cols, seed=100 + seed)
    scrambled, gt = ev.generate_puzzle(src, rows=rows, cols=cols,
                                       rotate=rotate, seed=seed,
                                       source_name=f"synthetic_{seed}")
    return src, scrambled, gt


def _save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (
            o.tolist() if isinstance(o, np.ndarray) else
            float(o) if isinstance(o, np.floating) else
            int(o) if isinstance(o, np.integer) else str(o)))
    print(f"  wrote {os.path.relpath(path, ROOT)}")
    return path


# --------------------------------------------------------------------------
# demo 1 -- enhancement
# --------------------------------------------------------------------------
def demo_enhancement():
    """Noise reduction, contrast adjustment and sharpening (task 1)."""
    print("[enhancement]")
    _dirs()
    photo, path = _sample_photo(max_side=900)
    if photo is None:
        print("  dataset not found, using a synthetic image")
        photo = ev.synthetic_source_image(360, 480, seed=5)
    crop = photo[:360, :480]

    rng = np.random.default_rng(0)
    noisy = np.clip(crop / 255.0 + rng.normal(0, 0.05, crop.shape), 0, 1)
    sp = noisy.copy()
    salt = rng.random(sp.shape[:2]) < 0.02
    pepper = rng.random(sp.shape[:2]) < 0.02
    sp[salt] = 1.0
    sp[pepper] = 0.0

    outputs = {
        "00_original": crop,
        "01_noisy_gaussian": noisy,
        "02_noisy_salt_pepper": sp,
        "03_gaussian_blur_s1.5": enh.gaussian_blur(sp, 1.5),
        "04_median_3": enh.median_filter(sp, 3),
        "05_median_5": enh.median_filter(sp, 5),
        "06_histogram_equalised": enh.histogram_equalization(crop),
        "07_contrast_stretched": enh.contrast_stretch(crop, 2, 98),
        "08_unsharp_mask": enh.unsharp_mask(crop, sigma=1.5, amount=1.2),
        "09_laplacian_sharpen": enh.laplacian_sharpen(crop, alpha=0.35),
        "10_pipeline_default": enh.enhance_for_segmentation(crop),
    }
    for name, img in outputs.items():
        imwrite(os.path.join(R_ENH, name + ".png"), img)
    imwrite(os.path.join(R_ENH, "contact_sheet.png"),
            make_grid(list(outputs.values()), cols=4))

    # quantitative: which denoiser restores the salt-and-pepper image best?
    report = {}
    for name, img in [("gaussian_1.5", enh.gaussian_blur(sp, 1.5)),
                      ("gaussian_2.5", enh.gaussian_blur(sp, 2.5)),
                      ("median_3", enh.median_filter(sp, 3)),
                      ("median_5", enh.median_filter(sp, 5))]:
        report[name] = ev.image_metrics(img, crop, allow_rotation=False)
    report["no_filtering"] = ev.image_metrics(sp, crop, allow_rotation=False)

    # Contrast study.  Entropy of the intensity histogram measures how evenly
    # the available levels are used, which is exactly what equalisation
    # maximises.  It is reported for the photograph *and* for a deliberately
    # low-contrast patch, because a photograph dominated by one huge dark
    # region has a step in its CDF that no monotone mapping can spread.
    def entropy(img):
        h = enh.histogram(img, 256)
        p = h / max(h.sum(), 1.0)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    low_contrast = 0.42 + 0.09 * np.asarray(crop, dtype=np.float64) / 255.0
    report["contrast"] = {}
    for label, img in [("photograph", crop), ("low_contrast_patch", low_contrast)]:
        report["contrast"][label] = {
            "entropy_bits": {
                "original": entropy(img),
                "equalised": entropy(enh.histogram_equalization(img)),
                "stretched": entropy(enh.contrast_stretch(img, 2, 98)),
            },
            "intensity_std": {
                "original": float(np.std(enh.to_gray(img))),
                "equalised": float(np.std(enh.to_gray(
                    enh.histogram_equalization(img)))),
                "stretched": float(np.std(enh.to_gray(
                    enh.contrast_stretch(img, 2, 98)))),
            },
        }
    imwrite(os.path.join(R_ENH, "11_low_contrast_patch.png"), low_contrast)
    imwrite(os.path.join(R_ENH, "12_low_contrast_equalised.png"),
            enh.histogram_equalization(low_contrast))
    _save_json(report, os.path.join(R_EVAL, "enhancement_study.json"))
    best = min((k for k in report if "mse" in report[k]),
               key=lambda k: report[k]["mse"])
    print(f"  best salt-and-pepper denoiser by MSE: {best} "
          f"(PSNR {report[best]['psnr_db']:.2f} dB, "
          f"unfiltered {report['no_filtering']['psnr_db']:.2f} dB)")
    lc = report["contrast"]["low_contrast_patch"]["entropy_bits"]
    print(f"  low-contrast patch entropy: {lc['original']:.2f} -> "
          f"{lc['equalised']:.2f} bits after equalisation")


# --------------------------------------------------------------------------
# demo 2 -- thresholding
# --------------------------------------------------------------------------
def demo_thresholding():
    """Global, Otsu and adaptive thresholding (task 1d)."""
    print("[thresholding]")
    _dirs()
    photo, _ = _sample_photo(max_side=900)
    if photo is None:
        _, photo, _ = _synthetic_puzzle()
    gray = enh.gaussian_blur(photo, 1.0)

    t_otsu = th.otsu_threshold(gray)
    t_iso = th.isodata_threshold(gray)
    variants = {
        "00_input": photo,
        "01_global_0.5": th.global_threshold(gray, 0.5),
        f"02_isodata_{t_iso:.3f}": th.global_threshold(gray, t_iso),
        f"03_otsu_{t_otsu:.3f}": th.otsu(gray),
        "04_adaptive_mean_51": th.adaptive_threshold(gray, 51, 0.02, "mean"),
        "05_adaptive_gaussian_51": th.adaptive_threshold(gray, 51, 0.02, "gaussian"),
        "06_background_distance": seg.background_distance(photo),
        "07_foreground_mask": seg.foreground_mask(photo, "background",
                                                  open_radius=2, close_radius=2),
    }
    for name, img in variants.items():
        imwrite(os.path.join(R_MASK, "threshold_" + name + ".png"), img)
    imwrite(os.path.join(R_MASK, "threshold_contact_sheet.png"),
            make_grid(list(variants.values()), cols=4))
    _save_json({"otsu_threshold": t_otsu, "isodata_threshold": t_iso},
               os.path.join(R_EVAL, "thresholds.json"))
    print(f"  Otsu {t_otsu:.3f}, isodata {t_iso:.3f}")


# --------------------------------------------------------------------------
# demo 3 -- edge detection
# --------------------------------------------------------------------------
def demo_edges():
    """Sobel, Prewitt and the full Canny detector (task 2)."""
    print("[edges]")
    _dirs()
    photo, _ = _sample_photo(max_side=900)
    if photo is None:
        _, photo, _ = _synthetic_puzzle()
    crop = photo[:400, :600]

    _, _, sob_mag, sob_dir = ed.sobel(crop)
    _, _, pre_mag, pre_dir = ed.prewitt(crop)
    edges, stages = ed.canny(crop, sigma=1.4, low=0.05, high=0.15,
                             return_stages=True)

    def angle_image(theta, mag):
        """Orientation shown as hue-like RGB, modulated by magnitude."""
        a = (np.rad2deg(theta) % 180.0) / 180.0
        rgb = np.stack([np.abs(np.sin(np.pi * a)),
                        np.abs(np.sin(np.pi * (a + 1 / 3))),
                        np.abs(np.sin(np.pi * (a + 2 / 3)))], axis=2)
        return rgb * mag[:, :, None]

    out = {
        "00_input": crop,
        "01_sobel_magnitude": sob_mag,
        "02_sobel_orientation": angle_image(sob_dir, sob_mag),
        "03_prewitt_magnitude": pre_mag,
        "04_prewitt_orientation": angle_image(pre_dir, pre_mag),
        "05_canny_smoothed": stages["smoothed"],
        "06_canny_nms": stages["nms"],
        "07_canny_strong": stages["strong"],
        "08_canny_weak": stages["weak"],
        "09_canny_edges": edges,
    }
    for sigma in (0.8, 1.4, 2.5):
        out[f"10_canny_sigma_{sigma}"] = ed.canny(crop, sigma=sigma,
                                                  low=0.05, high=0.15)
    for lo, hi in ((0.02, 0.06), (0.05, 0.15), (0.10, 0.25)):
        out[f"11_canny_{lo}_{hi}"] = ed.canny(crop, 1.4, lo, hi)
    for name, img in out.items():
        imwrite(os.path.join(R_EDGE, name + ".png"), img)
    imwrite(os.path.join(R_EDGE, "contact_sheet.png"),
            make_grid(list(out.values()), cols=4))

    stats = {"canny_low": stages["low"], "canny_high": stages["high"],
             "edge_pixel_fraction": {}}
    for name, img in out.items():
        if np.asarray(img).dtype == np.bool_:
            stats["edge_pixel_fraction"][name] = float(np.mean(img))
    _save_json(stats, os.path.join(R_EVAL, "edge_study.json"))
    print(f"  Canny kept {np.mean(edges) * 100:.2f}% of pixels as edges "
          f"(sigma 1.4, low 0.05, high 0.15)")


# --------------------------------------------------------------------------
# demo 4 -- segmentation
# --------------------------------------------------------------------------
def demo_segmentation():
    """Foreground mask, connected components, contours (task 3)."""
    print("[segmentation]")
    _dirs()
    photo, path = _sample_photo()
    synthetic = photo is None
    if synthetic:
        _, photo, _ = _synthetic_puzzle()

    mask = seg.foreground_mask(photo, "background", open_radius=2,
                               close_radius=2)
    labels_raw, n_raw = seg.connected_components(mask)
    labels, stats = seg.filter_components(labels_raw, min_area_ratio=0.45)
    pieces = ce.extract_pieces(photo, labels, stats=stats)

    imwrite(os.path.join(R_MASK, "scene_mask.png"), mask)
    imwrite(os.path.join(R_MASK, "scene_mask_overlay.png"),
            overlay_mask(photo, mask))
    imwrite(os.path.join(R_CONT, "scene_labels_raw.png"),
            colorize_labels(labels_raw))
    imwrite(os.path.join(R_CONT, "scene_labels_filtered.png"),
            colorize_labels(labels))

    outline = np.zeros(mask.shape, dtype=bool)
    for p in pieces:
        y0, x0 = p.bbox[0], p.bbox[1]
        outline[p.contour[:, 0] + y0, p.contour[:, 1] + x0] = True
    imwrite(os.path.join(R_CONT, "scene_contours.png"),
            overlay_mask(photo, seg.dilate(outline, 1), (255, 0, 0), 0.9))
    imwrite(os.path.join(R_CONT, "scene_pieces.png"),
            make_grid([p.image for p in pieces], cols=7))
    imwrite(os.path.join(R_CONT, "scene_pieces_normalised.png"),
            make_grid([ce.normalize_piece(p).image for p in pieces], cols=7))

    report = {"source": os.path.relpath(path, ROOT) if path else "synthetic",
              "raw_components": n_raw, "kept_components": len(stats),
              "areas": [s.area for s in stats]}
    _save_json(report, os.path.join(R_EVAL, "segmentation_scene.json"))
    print(f"  {n_raw} raw components -> {len(stats)} pieces")


# --------------------------------------------------------------------------
# demo 6 -- matching + weight study
# --------------------------------------------------------------------------
def _matching_stats(res, gt):
    assoc = ev.associate_with_ground_truth(res.pieces, gt)
    return ev.matching_accuracy(res.table, res.descriptions, gt, assoc), assoc


def demo_matching():
    """Compatibility measure: separability of true and false seams (task 5)."""
    print("[matching]")
    _dirs()
    _, scrambled, gt = _synthetic_puzzle(4, 5, 14, True)
    res = PuzzleSolver().solve(scrambled, grid_shape=(4, 5))
    acc, assoc = _matching_stats(res, gt)

    # distributions of each term for true vs. all admissible pairs
    tab = res.table
    n = len(res.descriptions)
    cell, dirs = {}, {}
    for i in range(n):
        r, c, ang = gt.placements[assoc[i]]
        cell[i] = (int(r), int(c))
        dirs[i] = ev.gt_side_directions(res.descriptions[i], ang)
    by_cell = {v: k for k, v in cell.items()}
    step = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
    opp = {"N": "S", "E": "W", "S": "N", "W": "E"}

    true_s, true_c, true_l = [], [], []
    for i in range(n):
        r, c = cell[i]
        for s in range(4):
            d = dirs[i][s]
            j = by_cell.get((r + step[d][0], c + step[d][1]))
            if j is None:
                continue
            t = dirs[j].index(opp[d])
            true_s.append(float(tab.shape[i, s, j, t]))
            true_c.append(float(tab.colour[i, s, j, t]))
            true_l.append(float(tab.length[i, s, j, t]))
    fin = np.isfinite(tab.cost)
    report = {
        "puzzle": "4x5 rotated",
        "n_true_seams": len(true_s),
        "true_seam": {"shape_mean": float(np.mean(true_s)),
                      "colour_mean": float(np.mean(true_c)),
                      "length_mean": float(np.mean(true_l))},
        "all_admissible": {"shape_mean": float(tab.shape[fin].mean()),
                           "colour_mean": float(tab.colour[fin].mean()),
                           "length_mean": float(tab.length[fin].mean())},
        "accuracy": acc,
        "n_best_buddies": len(em.best_buddies(tab)),
        "weights": res.table.weights.as_dict(),
    }
    _save_json(report, os.path.join(R_EVAL, "matching_study.json"))
    print(f"  true seams: shape {np.mean(true_s):.3f} colour {np.mean(true_c):.3f}"
          f" | all admissible: shape {tab.shape[fin].mean():.3f} "
          f"colour {tab.colour[fin].mean():.3f}")
    print(f"  top-1 {acc['top1_accuracy']:.2f}, top-3 {acc['top3_accuracy']:.2f}")


def demo_weights():
    """Sweep the shape/colour weights and record the effect (task 5)."""
    print("[weights]")
    _dirs()
    puzzles = [(3, 4, 12, True), (4, 5, 14, True), (5, 7, 16, True)]
    combos = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
              (1.0, 1.0, 0.5), (2.0, 1.0, 0.5), (1.0, 2.0, 0.5),
              (1.0, 0.5, 0.5), (1.0, 1.0, 1.0)]
    rows = []
    for ws, wc, wl in combos:
        top1, nbr, pos = [], [], []
        for (r, c, seed, rot) in puzzles:
            _, scrambled, gt = _synthetic_puzzle(r, c, seed, rot)
            res = PuzzleSolver(weights=em.MatchWeights(ws, wc, wl)).solve(
                scrambled, grid_shape=(r, c))
            assoc = ev.associate_with_ground_truth(res.pieces, gt)
            top1.append(ev.matching_accuracy(res.table, res.descriptions, gt,
                                             assoc)["top1_accuracy"])
            nbr.append(ev.neighbour_accuracy(res.assembly, gt,
                                             assoc)["neighbour_accuracy"])
            pos.append(ev.direct_accuracy(res.assembly, res.descriptions, gt,
                                          assoc)["position_accuracy"])
        rows.append({"w_shape": ws, "w_colour": wc, "w_length": wl,
                     "top1": float(np.mean(top1)),
                     "neighbour": float(np.mean(nbr)),
                     "position": float(np.mean(pos))})
        print(f"  w=({ws}, {wc}, {wl}) -> top1 {np.mean(top1):.2f} "
              f"neighbour {np.mean(nbr):.2f} position {np.mean(pos):.2f}")
    _save_json({"puzzles": [list(p) for p in puzzles], "results": rows},
               os.path.join(R_EVAL, "weight_study.json"))


def demo_benchmark():
    """Validate the assembly stage on clean input with a known answer."""
    print("[benchmark]")
    _dirs()
    rows = []
    for (r, c, seed, rot) in BENCHMARK:
        src, scrambled, gt = _synthetic_puzzle(r, c, seed, rot)
        tag = f"puzzle_{r}x{c}_seed{seed}" + ("" if rot else "_norot")
        t0 = time.perf_counter()
        # Metrics only: this run exists to prove the assembly stage is correct
        # on clean input, and its numbers do that.  Pictures of generated
        # puzzles are deliberately not written into results/, which holds
        # output from the real photographs.
        res, rep = solve_one(scrambled, (r, c), tag, reference=src, gt=gt,
                             save_images=False)
        row = {
            "puzzle": tag, "pieces": r * c, "rotated": rot,
            "pieces_found": res.n_pieces,
            "seconds": round(time.perf_counter() - t0, 2),
            "top1_matching": round(rep.matching.get("top1_accuracy", float("nan")), 3),
            "neighbour_accuracy": round(rep.neighbour.get("neighbour_accuracy", float("nan")), 3),
            "position_accuracy": round(rep.placement.get("position_accuracy", float("nan")), 3),
            "position_rotation_accuracy": round(
                rep.placement.get("position_and_rotation_accuracy", float("nan")), 3),
            "rotation_accuracy": round(rep.rotation.get("rotation_accuracy", float("nan")), 3),
            "quality": round(res.quality.get("quality", float("nan")), 3),
            "psnr_db": round(rep.image.get("psnr_db", float("nan")), 2) if rep.image else None,
            "ssim": round(rep.image.get("ssim", float("nan")), 3) if rep.image else None,
        }
        rows.append(row)
        print(f"  {tag:24s} pieces {row['pieces_found']:3d}/{row['pieces']:<3d}"
              f" nbr {row['neighbour_accuracy']:.2f} pos {row['position_accuracy']:.2f}"
              f" ssim {row['ssim']} {row['seconds']}s")

    summary = {
        "mean_neighbour_accuracy": float(np.mean([r["neighbour_accuracy"] for r in rows])),
        "mean_position_accuracy": float(np.mean([r["position_accuracy"] for r in rows])),
        "perfect_reconstructions": int(sum(1 for r in rows
                                           if r["position_accuracy"] >= 0.999)),
        "n_puzzles": len(rows),
    }
    _save_json({"rows": rows, "summary": summary},
               os.path.join(R_EVAL, "benchmark.json"))
    print(f"  {summary['perfect_reconstructions']}/{summary['n_puzzles']} "
          f"perfect, mean neighbour accuracy "
          f"{summary['mean_neighbour_accuracy']:.3f}")


# --------------------------------------------------------------------------
# demo 8 -- the provided dataset
# --------------------------------------------------------------------------
# ==========================================================================
# running the whole pipeline on one image
# ==========================================================================
def solve_one(image, grid=None, name="puzzle", reference=None, gt=None,
              solver_kwargs=None, save_images=True):
    """Run the whole pipeline on one image and write every artefact."""
    solver = PuzzleSolver(**(solver_kwargs or {}))
    res = solver.solve(image, grid)

    if save_images:
        imwrite(os.path.join(R_MASK, f"{name}_mask.png"), res.mask)
        imwrite(os.path.join(R_CONT, f"{name}_labels.png"),
                colorize_labels(res.labels))
        if res.descriptions:
            imwrite(os.path.join(R_CONT, f"{name}_described.png"),
                    render_scattered_overview(image, res.descriptions))
        if res.reconstruction is not None:
            imwrite(os.path.join(R_REC, f"{name}_reconstructed.png"),
                    res.reconstruction)

    report = ev.EvaluationReport(name=name, grid_shape=res.grid_shape,
                                 n_pieces=res.n_pieces,
                                 timings=res.timings, seam=res.quality,
                                 notes=list(res.notes))
    if gt is not None and res.table is not None:
        assoc = ev.associate_with_ground_truth(res.pieces, gt)
        report.matching = ev.matching_accuracy(res.table, res.descriptions,
                                               gt, assoc)
        report.placement = ev.direct_accuracy(res.assembly, res.descriptions,
                                              gt, assoc)
        report.neighbour = ev.neighbour_accuracy(res.assembly, gt, assoc)
        report.rotation = ev.rotation_accuracy(res.assembly, res.descriptions,
                                               gt, assoc)
    if reference is not None and res.reconstruction is not None:
        body = res.reconstructed_body()
        if body is not None and body.size:
            report.image = ev.image_metrics(body, reference)
    ev.save_report(report, os.path.join(R_EVAL, f"{name}.json"))
    return res, report


#: Grid layout of the jigsaw photographed in the provided dataset (35 pieces).
DATASET_GRID = (5, 7)
#: Class names of the Roboflow export, in class-index order.  The class of a
#: box is the *identity* of the piece (1-35), not its position in the frame.
DATASET_CLASS_NAMES = ['1', '10', '11', '12', '13', '14', '15', '16', '17',
                       '18', '19', '2', '20', '21', '22', '23', '24', '25',
                       '26', '27', '28', '29', '3', '30', '31', '32', '33',
                       '34', '35', '4', '5', '6', '7', '8', '9']


def dataset_true_cells(pieces, boxes, shape):
    """Ground-truth grid cell of every segmented piece, from the annotations.

    The dataset labels each piece with its **identity** (1-35), and those ids
    turn out to be the row-major positions of the finished 5x7 puzzle: id
    ``k`` sits at ``((k-1)//7, (k-1)%7)``.  The evidence is the flat sides --
    counting them per id across photographs gives zero flats for all fifteen
    ids the hypothesis calls interior, and at least one for the ids it calls
    border (``python main.py --run dataset`` records the per-image counts).

    That turns the identity labels into a real answer key, so reconstruction
    accuracy can be measured on the actual photographs and not only on
    synthetic puzzles.  Returns ``{piece_index: (row, col)}`` for the pieces
    that could be matched to an annotation.
    """
    rows, cols = DATASET_GRID
    px = [(int(DATASET_CLASS_NAMES[b.cls]),) + b.pixel_box(shape) for b in boxes]
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
#: Solver settings used for the dataset photographs.  Full resolution matters:
#: at 1280 px the piece body is ~75 px and the descriptors are measurably
#: noisier than at the native 1920 px, where it is ~114 px.
DATASET_SOLVER = {"open_radius": 2, "close_radius": 2, "min_area_ratio": 0.45,
                  "max_area_ratio": 1.7, "colour_norm": "meanstd"}


def demo_data(limit: int = 8):
    """Populate ``data/`` from the provided dataset.

    ``data/input/`` gets the scrambled-puzzle photographs the pipeline is run
    on, ``data/ground_truth/`` the answer key for each (the true 5x7 cell of
    every annotated piece, recovered from the identity labels as described in
    :func:`dataset_true_cells`), and ``data/sample_pieces/`` a handful of
    individual pieces cropped by the segmentation stage.
    """
    print("[data] populating data/ from the dataset")
    _dirs()
    scenes = find_scenes(DETECTION, "train", min_objects=35)
    scenes += find_scenes(DETECTION, "valid", min_objects=35)
    if not scenes:
        print(f"  dataset not found under {DETECTION}; skipping")
        return
    scenes = scenes[:limit]

    rows, cols = DATASET_GRID
    for i, (img_path, lab_path, n_obj) in enumerate(scenes):
        name = os.path.splitext(os.path.basename(img_path))[0][:20]
        img = imread(img_path, max_side=1920)
        imwrite(os.path.join(DATA, "input", name + ".png"), img)

        key = {}
        for b in load_yolo_labels(lab_path):
            pid = int(DATASET_CLASS_NAMES[b.cls])
            y0, x0, y1, x1 = b.pixel_box(img.shape)
            key[str(pid)] = {"cell": [(pid - 1) // cols, (pid - 1) % cols],
                             "box_yxyx": [y0, x0, y1, x1]}
        _save_json({"image": name + ".png", "grid": [rows, cols],
                    "n_pieces": n_obj,
                    "note": "cell = row-major position of the piece identity "
                            "in the finished puzzle; see report section 9.1",
                    "pieces": key},
                   os.path.join(DATA, "ground_truth", name + ".json"))

        if i == 0:                       # a few individual pieces
            res = PuzzleSolver(**DATASET_SOLVER).solve(img, DATASET_GRID)
            for p in res.pieces[:8]:
                imwrite(os.path.join(DATA, "sample_pieces",
                                     f"piece_{p.index:02d}.png"), p.image)
    print(f"  {len(scenes)} puzzle(s) in data/input with their answer keys")


def demo_dataset(limit: int | None = None, max_side: int = 1920):
    """Run the whole pipeline on the real dataset photographs.

    This is the primary result of the milestone: every photograph in the
    provided dataset that shows all 35 pieces scattered is segmented,
    described, matched and assembled.  Segmentation is scored against the
    supplied YOLO boxes (the only ground truth the dataset carries) and the
    reconstruction against the puzzle's known 5x7 layout.
    """
    print("[dataset] real photographs from detection/")
    _dirs()
    scenes = find_scenes(DETECTION, "train", min_objects=35)
    scenes += find_scenes(DETECTION, "valid", min_objects=35)
    if limit:
        scenes = scenes[:limit]
    if not scenes:
        print(f"  dataset not found under {DETECTION}; skipping")
        return

    rows = []
    for i, (img_path, lab_path, n_obj) in enumerate(scenes):
        img = imread(img_path, max_side=max_side)
        boxes = load_yolo_labels(lab_path)
        name = os.path.splitext(os.path.basename(img_path))[0][:20]
        res, rep = solve_one(img, DATASET_GRID, f"dataset_{name}",
                             solver_kwargs=DATASET_SOLVER,
                             save_images=i < 8)

        det = detection_metrics(
            [type("S", (), {"centroid": (p.bbox[0] + p.centroid[0],
                                         p.bbox[1] + p.centroid[1])})()
             for p in res.pieces], boxes, img.shape)
        flats = sum(1 for d in res.descriptions for s in d.sides if s.is_flat)

        # ground truth from the identity labels (see dataset_true_cells)
        nbr = pos = float("nan")
        if res.assembly is not None:
            cells = dataset_true_cells(res.pieces, boxes, img.shape)
            if cells:
                nbr = ev.neighbour_accuracy_from_cells(
                    res.assembly, cells)["neighbour_accuracy"]
                pos = ev.position_accuracy_from_cells(
                    res.assembly, cells)["position_accuracy"]

        rows.append({
            "image": os.path.basename(img_path), "annotated": n_obj,
            **det, "pieces": res.n_pieces,
            "flat_sides": flats,
            "expected_flat_sides": 2 * (DATASET_GRID[0] + DATASET_GRID[1]),
            "corner_pieces": sum(1 for d in res.descriptions
                                 if d.is_corner_piece),
            "placed": res.assembly.n_placed if res.assembly else 0,
            "forced": res.assembly.n_forced if res.assembly else 0,
            "neighbour_accuracy": round(float(nbr), 3),
            "position_accuracy": round(float(pos), 3),
            "quality": round(res.quality.get("quality", float("nan")), 3),
            "seconds": round(res.timings.get("total", 0.0), 1)})
        print(f"  {name:22s} pieces {res.n_pieces:2d}/{n_obj:2d} "
              f"rec {det['recall']:.2f} prec {det['precision']:.2f} "
              f"flats {flats:2d}/{rows[-1]['expected_flat_sides']} "
              f"| nbr {nbr:.2f} pos {pos:.2f} q {rows[-1]['quality']:.3f}")

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "n_images": len(rows),
        "grid": list(DATASET_GRID),
        "mean_recall": mean("recall"),
        "mean_precision": mean("precision"),
        "mean_f1": mean("f1"),
        "mean_pieces_found": mean("pieces"),
        "images_with_all_35": int(sum(1 for r in rows if r["pieces"] == 35)),
        "mean_flat_sides": mean("flat_sides"),
        "mean_corner_pieces": mean("corner_pieces"),
        "mean_neighbour_accuracy": mean("neighbour_accuracy"),
        "mean_position_accuracy": mean("position_accuracy"),
        "mean_quality": mean("quality"),
        "mean_seconds": mean("seconds"),
    }
    _save_json({"summary": summary, "rows": rows},
               os.path.join(R_EVAL, "dataset_study.json"))
    print(f"\n  {len(rows)} photographs")
    print(f"  segmentation: recall {summary['mean_recall']:.3f}, "
          f"precision {summary['mean_precision']:.3f}, "
          f"F1 {summary['mean_f1']:.3f}")
    print(f"  pieces isolated {summary['mean_pieces_found']:.1f}/35 on average, "
          f"all 35 on {summary['images_with_all_35']}/{len(rows)} images")
    print(f"  flat sides {summary['mean_flat_sides']:.1f}/24 expected, "
          f"corner pieces {summary['mean_corner_pieces']:.1f}/4")
    print(f"  reconstruction: neighbour accuracy "
          f"{summary['mean_neighbour_accuracy']:.3f}, position accuracy "
          f"{summary['mean_position_accuracy']:.3f}, "
          f"quality {summary['mean_quality']:.3f}")


# --------------------------------------------------------------------------
#: ``--run`` targets.  ``dataset`` is the headline: the real photographs.
#: The stage figures also run on real photographs; only ``validate`` and the
#: puzzles it needs use the synthetic generator, because reconstruction
#: accuracy cannot be measured without an answer key and the dataset has none.
RUNS = {
    "data": demo_data,
    "dataset": demo_dataset,
    "enhancement": demo_enhancement,
    "thresholding": demo_thresholding,
    "edges": demo_edges,
    "segmentation": demo_segmentation,
    "matching": demo_matching,
    "weights": demo_weights,
    "validate": demo_benchmark,
}
STAGES = ["enhancement", "thresholding", "edges", "segmentation"]
ALL_ORDER = ["data", "dataset"] + STAGES + ["matching", "weights", "validate"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="scrambled puzzle image to reconstruct")
    ap.add_argument("--grid", help="grid size as ROWSxCOLS (else inferred)")
    ap.add_argument("--reference", help="original picture, for PSNR/SSIM")
    ap.add_argument("--out", default=None, help="output name (default: input stem)")
    ap.add_argument("--max-side", type=int, default=1920,
                    help="downscale the input so its longest side is at most this")
    ap.add_argument("--limit", type=int, default=None,
                    help="dataset run: only the first N photographs")
    ap.add_argument("--run", choices=list(RUNS) + ["all", "stages"],
                    help="run the dataset, a single stage, or everything")
    ap.add_argument("--demo", dest="run", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.run:
        if args.run == "all":
            names = ALL_ORDER
        elif args.run == "stages":
            names = STAGES
        else:
            names = [args.run]
        t0 = time.perf_counter()
        for name in names:
            if name == "dataset" and args.limit:
                RUNS[name](limit=args.limit)
            else:
                RUNS[name]()
        print(f"done in {time.perf_counter() - t0:.1f}s")
        return 0

    if not args.input:
        # no arguments at all: the dataset is what this project is about
        demo_dataset(limit=args.limit)
        return 0

    if not args.input:
        ap.print_help()
        return 1

    image = imread(args.input, max_side=args.max_side)
    grid = None
    if args.grid:
        r, c = args.grid.lower().split("x")
        grid = (int(r), int(c))
    reference = imread(args.reference) if args.reference else None
    name = args.out or os.path.splitext(os.path.basename(args.input))[0]

    # An answer key next to the input is used automatically.  Two schemas
    # exist: the dataset keys written by ``--run data`` (piece identity ->
    # cell) and the synthetic ones (full placements), so pick them apart
    # rather than assuming either.
    gt = None
    cells_key = None
    gt_path = os.path.join(DATA, "ground_truth", name + ".json")
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if "placements" in raw:
            gt = ev.load_ground_truth(gt_path)
            ref_path = os.path.join(DATA, "ground_truth", name + "_original.png")
            if reference is None and os.path.exists(ref_path):
                reference = imread(ref_path)
        elif "pieces" in raw:
            cells_key = raw
            if grid is None:
                grid = tuple(raw.get("grid", DATASET_GRID))

    _dirs()
    solver_kwargs = DATASET_SOLVER if cells_key else None
    res, rep = solve_one(image, grid, name, reference=reference, gt=gt,
                         solver_kwargs=solver_kwargs)
    print(res.summary())
    for note in res.notes:
        print("  " + note)

    if cells_key and res.assembly is not None:
        # score against the dataset answer key
        px = [(int(pid), *v["box_yxyx"]) for pid, v in cells_key["pieces"].items()]
        claimed = [False] * len(px)
        cells = {}
        for p in res.pieces:
            cy = p.bbox[0] + p.centroid[0]
            cx = p.bbox[1] + p.centroid[1]
            for k, (pid, y0, x0, y1, x1) in enumerate(px):
                if claimed[k]:
                    continue
                if y0 <= cy <= y1 and x0 <= cx <= x1:
                    claimed[k] = True
                    cells[p.index] = tuple(cells_key["pieces"][str(pid)]["cell"])
                    break
        if cells:
            nbr = ev.neighbour_accuracy_from_cells(res.assembly, cells)
            pos = ev.position_accuracy_from_cells(res.assembly, cells)
            print(f"  vs answer key: neighbour accuracy "
                  f"{nbr['neighbour_accuracy']:.3f}, position accuracy "
                  f"{pos['position_accuracy']:.3f}")

    print(f"  reconstruction -> results/reconstructed_images/{name}_reconstructed.png")
    print(f"  metrics        -> results/evaluation_results/{name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
