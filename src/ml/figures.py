"""Milestone 2 -- the pictures behind the numbers.

Every table in the Milestone 2 report is produced by :mod:`src.ml.evaluate` and
:mod:`src.ml.hard_eval`.  This module draws the same results so they can be
looked at rather than only read:

* :func:`training_curves` -- loss and validation AUC per epoch for both models.
* :func:`comparison_chart` -- the headline metrics on the unseen test split.
* :func:`texture_chart` -- accuracy as the picture's texture is faded out.
* :func:`reconstruction_panel` -- the reconstructed image each method produces
  for one puzzle, side by side, captioned with its measured accuracy.

Only the actual dataset photographs are drawn.  Generated puzzles are what the
models are trained and scored on, but a picture of a generated puzzle shows
nothing that its numbers do not, so the reconstructions kept here are of the
real jigsaw.  Everything is written into ``results/milestone2/figures/``.
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from .. import evaluation as ev          # noqa: E402
from ..assembly import assemble, render_assembly   # noqa: E402
from ..edge_matching import build_compatibility    # noqa: E402
from .infer import gnn_table, siamese_table        # noqa: E402

__all__ = [
    "METHOD_LABELS",
    "method_tables",
    "training_curves",
    "comparison_chart",
    "texture_chart",
    "reconstruction_panel",
    "real_photograph_panels",
]

METHOD_LABELS = {
    "classical": "Classical (Milestone 1)",
    "siamese_cnn": "Siamese CNN",
    "graph_nn": "Graph neural network",
}
_COLOURS = {"classical": "#8c8c8c", "siamese_cnn": "#1f77b4",
            "graph_nn": "#d62728"}


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def method_tables(sample, siamese=None, gnn=None) -> dict:
    """The compatibility table each method produces for one puzzle."""
    out = {"classical": sample.table if sample.table is not None
           else build_compatibility(sample.descriptions)}
    if siamese is not None:
        out["siamese_cnn"] = siamese_table(siamese, sample.descriptions)
    if gnn is not None:
        out["graph_nn"] = gnn_table(gnn, sample.descriptions,
                                    out["classical"])
    return out


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------
def training_curves(training: dict, path: str) -> str:
    """Loss and validation AUC per epoch, both models on one axis pair."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ("siamese_cnn", "graph_nn"):
        hist = training.get(key, {}).get("history", {})
        train = hist.get("train_loss") or []
        val = hist.get("val_loss") or []
        auc = hist.get("val_auc") or []
        ep = np.arange(1, len(train) + 1)
        colour = _COLOURS[key]
        axes[0].plot(ep, train, color=colour, label=f"{METHOD_LABELS[key]} - train")
        if val:
            axes[0].plot(np.arange(1, len(val) + 1), val, color=colour,
                         linestyle="--", label=f"{METHOD_LABELS[key]} - validation")
        if auc:
            axes[1].plot(np.arange(1, len(auc) + 1), auc, color=colour,
                         label=METHOD_LABELS[key])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("binary cross-entropy")
    axes[0].set_title("Training and validation loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation AUC")
    axes[1].set_title("Ranking quality on held-out puzzles")
    axes[1].set_ylim(0.5, 1.005)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("Milestone 2 - training behaviour", fontweight="bold")
    return _save(fig, path)


def comparison_chart(comparison: dict, path: str) -> str:
    """The headline metrics on the unseen test puzzles, as grouped bars."""
    metrics = [("top1", "top-1 side match"), ("auc", "ranking AUC"),
               ("neighbour_accuracy", "neighbour accuracy"),
               ("position_accuracy", "position accuracy")]
    methods = [m for m in METHOD_LABELS if m in comparison]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    width = 0.8 / max(len(methods), 1)
    for k, m in enumerate(methods):
        block = comparison[m]
        flat = {}
        for group in ("matching", "reconstruction", "cost"):
            flat.update(block.get(group, {}) if isinstance(block.get(group), dict) else {})
        flat.update({k2: v for k2, v in block.items() if not isinstance(v, dict)})
        vals = [float(flat.get(key, np.nan)) for key, _ in metrics]
        pos = np.arange(len(metrics)) + (k - (len(methods) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.92, label=METHOD_LABELS[m],
                      color=_COLOURS[m])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylim(0, 1.12); ax.set_ylabel("score")
    ax.set_title("Unseen test puzzles - all three methods", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def texture_chart(sweep: dict, path: str) -> str:
    """Neighbour accuracy against how much texture the picture still has."""
    keys = sorted(sweep, key=lambda k: float(k.rsplit("_", 1)[1]))
    x = [float(k.rsplit("_", 1)[1]) for k in keys]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m in METHOD_LABELS:
        y = [sweep[k].get(m, {}).get("neighbour_accuracy", np.nan) for k in keys]
        if not any(np.isfinite(v) for v in y):
            continue
        ax.plot(x, y, "o-", color=_COLOURS[m], label=METHOD_LABELS[m])
    ax.invert_xaxis()
    ax.set_xlabel("texture retained in the picture  (1.0 = normal, 0 = flat grey)")
    ax.set_ylabel("neighbour accuracy")
    ax.set_title("What happens as the picture stops helping", fontweight="bold")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, path)


# --------------------------------------------------------------------------
# reconstructions
# --------------------------------------------------------------------------
def reconstruction_panel(sample, path: str, siamese=None, gnn=None,
                         tables: dict | None = None, title: str | None = None,
                         cells: dict | None = None) -> dict:
    """Draw each method's reconstruction of one puzzle side by side.

    Returns the measured accuracy per method so the caller can log it; the
    caption under each image reports the same numbers, because a reconstruction
    that looks plausible is not evidence on its own.
    """
    tables = tables or method_tables(sample, siamese, gnn)
    cells = cells if cells is not None else sample.cells
    names = [m for m in METHOD_LABELS if m in tables]

    images, captions, scores = [], [], {}
    for m in names:
        asm = assemble(sample.descriptions, tables[m], sample.grid_shape)
        images.append(render_assembly(sample.descriptions, asm))
        nbr = ev.neighbour_accuracy_from_cells(asm, cells)["neighbour_accuracy"]
        pos = ev.position_accuracy_from_cells(asm, cells)["position_accuracy"]
        scores[m] = {"neighbour_accuracy": float(nbr),
                     "position_accuracy": float(pos)}
        captions.append(f"neighbour {nbr:.2f}   position {pos:.2f}")

    fig, axes = plt.subplots(1, len(images), figsize=(4.6 * len(images), 5.0))
    axes = np.atleast_1d(axes)
    for ax, img, m, cap in zip(axes, images, names, captions):
        ax.imshow(img)
        ax.set_title(METHOD_LABELS[m], fontsize=10)
        ax.set_xlabel(cap, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title or sample.source, fontweight="bold")
    _save(fig, path)
    return scores


def real_photograph_panels(siamese, gnn, detection_dir: str, out_dir: str,
                           limit: int = 8, verbose: bool = True) -> list:
    """Draw all three reconstructions of each dataset photograph.

    These are the only reconstructed images kept, and none of them is a
    success: the captions report what was actually measured, which is that no
    method solves the real jigsaw.
    """
    import time

    from .hard_eval import build_real_sample, real_scenes

    out = []
    for img_path, lab_path in real_scenes(detection_dir, limit):
        t0 = time.perf_counter()
        sample = build_real_sample(img_path, lab_path)
        if sample is None:
            if verbose:
                print(f"  {os.path.basename(img_path)}: annotations could not "
                      "be matched to the segmented pieces, skipped", flush=True)
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0][:40]
        path = os.path.join(out_dir, f"real_{stem}.png")
        scores = reconstruction_panel(
            sample, path, siamese=siamese, gnn=gnn,
            title=f"Dataset photograph - {os.path.basename(img_path)}")
        out.append({"figure": path, "photograph": os.path.basename(img_path),
                    "scores": scores})
        if verbose:
            print(f"  {os.path.basename(img_path)[:30]:30s} " + "  ".join(
                f"{m} nbr {v['neighbour_accuracy']:.2f}"
                for m, v in scores.items()) +
                f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
    return out
