"""Classical computer-vision library for jigsaw-puzzle reconstruction.

CSE480 Machine Vision -- Milestone 1.

The package is organised to mirror the six tasks of the brief:

===============================  ===============================================
:mod:`src.enhancement`           shared array primitives; noise reduction,
                                 contrast adjustment, sharpening (task 1)
:mod:`src.thresholding`          global / Otsu / adaptive thresholding (task 1)
:mod:`src.edge_detection`        Sobel, Prewitt, full Canny (task 2)
:mod:`src.segmentation`          foreground mask, morphology, connected
                                 components (task 3)
:mod:`src.contour_extraction`    boundary tracing, piece cropping, normalised
                                 orientation (task 3)
:mod:`src.piece_description`     corners, four sides, tab/blank/flat, colour
                                 strip (task 4)
:mod:`src.edge_matching`         pairwise side compatibility, shape + colour
                                 (task 5)
:mod:`src.assembly`              greedy best-first placement with rotation,
                                 and rendering the result (task 6)
:mod:`src.evaluation`            synthetic puzzles with ground truth and every
                                 reconstruction-quality metric
===============================  ===============================================

This module itself provides the **end-to-end routine** the brief asks for:
:class:`PuzzleSolver` / :func:`solve_puzzle` accept a scrambled puzzle image
and return the reconstructed image together with a numerical measure of
reconstruction quality.

Only NumPy is used for the algorithms; Pillow (in ``main.py``) is used
strictly for reading and writing image files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import (assembly, contour_extraction, edge_detection, edge_matching,
               enhancement, evaluation, piece_description, segmentation,
               thresholding)
from . import segmentation as seg
from .assembly import Assembly, assemble, infer_grid_shape, render_assembly
from .contour_extraction import extract_pieces
from .edge_matching import CompatibilityTable, MatchWeights, build_compatibility
from .enhancement import enhance_for_segmentation
from .evaluation import seam_quality
from .piece_description import describe_pieces

__version__ = "1.0.0"

__all__ = [
    "enhancement", "thresholding", "edge_detection", "segmentation",
    "contour_extraction", "piece_description", "edge_matching", "assembly",
    "evaluation",
    "SolveResult", "PuzzleSolver", "solve_puzzle",
]

@dataclass
class SolveResult:
    """Everything the pipeline produced for one puzzle."""
    image: np.ndarray                       # the input (possibly resized)
    enhanced: np.ndarray | None = None
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    pieces: list = field(default_factory=list)
    descriptions: list = field(default_factory=list)
    table: CompatibilityTable | None = None
    assembly: Assembly | None = None
    reconstruction: np.ndarray | None = None
    #: ``(pad_y, pad_x, cell_h, cell_w)`` of :attr:`reconstruction`
    geometry: tuple[int, int, int, int] = (0, 0, 0, 0)
    grid_shape: tuple[int, int] = (0, 0)
    quality: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def n_pieces(self) -> int:
        return len(self.pieces)

    def reconstructed_body(self) -> np.ndarray | None:
        """The reconstruction cropped to exactly ``rows x cols`` cells.

        This is the crop to compare against the original picture, since the
        full canvas also contains the margin that keeps border tabs visible.
        """
        if self.reconstruction is None:
            return None
        pad_y, pad_x, ch, cw = self.geometry
        rows, cols = self.grid_shape
        return self.reconstruction[pad_y:pad_y + rows * ch,
                                   pad_x:pad_x + cols * cw]

    def summary(self) -> str:
        q = self.quality.get("quality", float("nan"))
        return (f"{self.n_pieces} pieces, grid {self.grid_shape[0]}x"
                f"{self.grid_shape[1]}, quality {q:.3f}, "
                f"mean seam cost {self.quality.get('mean_seam_cost', float('nan')):.4f}, "
                f"{self.timings.get('total', 0.0):.2f}s")


class PuzzleSolver:
    """Configurable end-to-end solver.

    Parameters mirror the individual stages; the defaults are the ones used
    for every figure and number in the report.
    """

    def __init__(self,
                 enhance: bool = True,
                 median_size: int = 3,
                 gaussian_sigma: float = 1.0,
                 threshold_method: str = "auto",
                 open_radius: int = 2,
                 close_radius: int = 2,
                 min_area_ratio: float = 0.3,
                 max_area_ratio: float | None = None,
                 split_touching: bool = True,
                 drop_border_components: bool = False,
                 n_side_samples: int = 96,
                 flat_tol: float = 0.055,
                 weights: MatchWeights | None = None,
                 colour_norm: str = "none",
                 colour_metric: str = "ssd",
                 depths: tuple = (2.0, 4.0, 6.0),
                 border_mode: str = "hard",
                 verbose: bool = False):
        self.enhance = enhance
        self.median_size = median_size
        self.gaussian_sigma = gaussian_sigma
        self.threshold_method = threshold_method
        self.open_radius = open_radius
        self.close_radius = close_radius
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.split_touching = split_touching
        self.drop_border_components = drop_border_components
        self.n_side_samples = n_side_samples
        self.flat_tol = flat_tol
        self.weights = weights or MatchWeights()
        self.colour_norm = colour_norm
        self.colour_metric = colour_metric
        self.depths = tuple(depths)
        self.border_mode = border_mode
        self.verbose = verbose

    # ------------------------------------------------------------------
    def solve(self, image: np.ndarray,
              grid_shape: tuple[int, int] | None = None) -> SolveResult:
        """Reconstruct the puzzle shown in ``image``."""
        t_all = time.perf_counter()
        res = SolveResult(image=np.asarray(image))
        timings = res.timings

        # 1. enhancement ------------------------------------------------
        t = time.perf_counter()
        if self.enhance:
            res.enhanced = enhance_for_segmentation(
                image, median_size=self.median_size,
                gaussian_sigma=self.gaussian_sigma, stretch=False,
                sharpen_amount=0.0)
        else:
            res.enhanced = image
        timings["enhancement"] = time.perf_counter() - t

        # 2. segmentation ------------------------------------------------
        t = time.perf_counter()
        res.mask = seg.foreground_mask(
            res.enhanced, method=self.threshold_method,
            median_size=1, smooth_sigma=0.0,
            open_radius=self.open_radius, close_radius=self.close_radius)
        labels, n_raw = seg.connected_components(res.mask)
        if self.drop_border_components:
            kept = seg.remove_border_components(res.mask, labels)
            labels, _ = seg.connected_components(kept)
        labels, stats = seg.filter_components(labels,
                                              min_area_ratio=self.min_area_ratio)
        n_kept = len(stats)
        n_split = 0
        if self.split_touching:
            # Pieces that touch in the photograph share one component; the
            # distance-transform watershed separates them.
            labels, n_split = seg.split_touching(labels)
            if n_split:
                labels, stats = seg.filter_components(
                    labels, min_area_ratio=self.min_area_ratio,
                    max_area_ratio=self.max_area_ratio)
        elif self.max_area_ratio is not None:
            labels, stats = seg.filter_components(
                labels, min_area_ratio=self.min_area_ratio,
                max_area_ratio=self.max_area_ratio)
        res.labels = labels
        timings["segmentation"] = time.perf_counter() - t
        res.notes.append(f"{n_raw} raw components, {n_kept} kept, "
                         f"{n_split} blob(s) split into touching pieces, "
                         f"{len(stats)} pieces")

        # 3. contours + pieces -------------------------------------------
        t = time.perf_counter()
        res.pieces = extract_pieces(res.image, labels, stats=stats)
        timings["contours"] = time.perf_counter() - t

        # 4. description ---------------------------------------------------
        t = time.perf_counter()
        res.descriptions = describe_pieces(res.pieces,
                                           n_side_samples=self.n_side_samples,
                                           flat_tol=self.flat_tol,
                                           depths=self.depths)
        timings["description"] = time.perf_counter() - t

        if not res.descriptions:
            res.notes.append("no pieces found -- nothing to assemble")
            timings["total"] = time.perf_counter() - t_all
            return res

        # 5. matching ------------------------------------------------------
        t = time.perf_counter()
        res.table = build_compatibility(res.descriptions, self.weights,
                                        colour_norm=self.colour_norm,
                                        colour_metric=self.colour_metric)
        timings["matching"] = time.perf_counter() - t

        # 6. assembly ------------------------------------------------------
        t = time.perf_counter()
        res.grid_shape = grid_shape or infer_grid_shape(res.descriptions)
        res.assembly = assemble(res.descriptions, res.table, res.grid_shape,
                                verbose=self.verbose,
                                border_mode=self.border_mode)
        res.grid_shape = res.assembly.grid_shape
        timings["assembly"] = time.perf_counter() - t

        # 7. rendering + quality -------------------------------------------
        t = time.perf_counter()
        res.reconstruction, res.geometry = render_assembly(
            res.descriptions, res.assembly, return_geometry=True)
        timings["rendering"] = time.perf_counter() - t

        res.quality = seam_quality(res.assembly, res.table)
        res.quality["n_forced_placements"] = res.assembly.n_forced
        res.quality["complete"] = res.assembly.complete
        timings["total"] = time.perf_counter() - t_all
        return res


def solve_puzzle(image: np.ndarray, grid_shape: tuple[int, int] | None = None,
                 **kwargs) -> SolveResult:
    """Convenience wrapper around :class:`PuzzleSolver`."""
    return PuzzleSolver(**kwargs).solve(image, grid_shape)
