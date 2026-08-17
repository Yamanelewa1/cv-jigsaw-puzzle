"""Milestone 2 / task 5 -- comparing the two models with the classical method.

All three methods are scored on the **same unseen test puzzles**, and every
one of them hands its compatibility table to the *same* assembler, so the
numbers below differ only because the matchers differ.

The brief asks the comparison to cover matching and neighbour accuracy, piece
position and orientation accuracy, complete reconstruction accuracy, the
reconstruction-quality score, execution and training time, and model size --
:func:`evaluate_method` produces all of them for one method, and
:func:`compare_methods` runs the three side by side.

*Complete reconstruction accuracy* is reported as the fraction of test
puzzles solved **perfectly** (every piece in the right cell), which is the
figure that actually matters to a user: a puzzle 90 % right is still a
puzzle that is wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import evaluation as ev
from ..assembly import assemble, render_assembly
from ..edge_matching import build_compatibility
from .infer import gnn_table, predicted_matches, siamese_table

__all__ = [
    "MethodResult",
    "matching_metrics",
    "evaluate_method",
    "compare_methods",
]

_STEP = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


@dataclass
class MethodResult:
    """Everything measured for one method over the test split."""
    name: str
    matching: dict = field(default_factory=dict)
    reconstruction: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    per_puzzle: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"method": self.name, "matching": self.matching,
                "reconstruction": self.reconstruction, "cost": self.cost,
                "per_puzzle": self.per_puzzle}


# ==========================================================================
# matching quality, independent of the search
# ==========================================================================
def matching_metrics(table, sample) -> dict:
    """How well the table alone ranks the true partner of each side.

    ``top1`` is the fraction of sides whose cheapest admissible partner is
    the true one; ``auc`` is the ranking quality over all admissible pairs;
    ``best_buddy_precision`` is how often a mutual first choice is correct --
    the quantity the assembler leans on hardest.
    """
    n = sample.n_pieces
    cost = table.cost.reshape(n * 4, n * 4)
    pos = {(i * 4 + s, j * 4 + t) for (i, s, j, t) in sample.positives}
    pos |= {(b, a) for (a, b) in pos}
    if not pos:
        return {}

    top1 = total = 0
    ranks = []
    for a in range(n * 4):
        row = cost[a]
        if not np.isfinite(row).any():
            continue
        partners = [b for b in range(n * 4) if (a, b) in pos]
        if not partners:
            continue
        total += 1
        best = int(np.argmin(np.where(np.isfinite(row), row, np.inf)))
        top1 += best in partners
        order = np.argsort(np.where(np.isfinite(row), row, np.inf))
        ranks.append(int(min(np.flatnonzero(np.isin(order, partners)))))

    finite = np.isfinite(cost)
    labels, scores = [], []
    for a in range(n * 4):
        for b in range(n * 4):
            if finite[a, b]:
                labels.append(1.0 if (a, b) in pos else 0.0)
                scores.append(-cost[a, b])
    from .train import roc_auc
    auc = roc_auc(np.array(labels), np.array(scores)) if labels else float("nan")

    best = np.argmin(np.where(finite, cost, np.inf), axis=1)
    bb_ok = bb_n = 0
    for a in range(n * 4):
        b = int(best[a])
        if not finite[a, b]:
            continue
        if int(best[b]) == a and a < b:
            bb_n += 1
            bb_ok += (a, b) in pos
    return {"top1": top1 / max(total, 1),
            "median_rank": float(np.median(ranks)) if ranks else float("nan"),
            "auc": auc,
            "n_best_buddies": bb_n,
            "best_buddy_precision": bb_ok / max(bb_n, 1),
            "n_true_seams": len(pos) // 2}


# ==========================================================================
# one method over the whole test split
# ==========================================================================
def evaluate_method(name: str, samples, indices, table_fn,
                    model=None, render: bool = False) -> MethodResult:
    """Score one method on the test puzzles.

    ``table_fn(sample) -> CompatibilityTable`` is the only thing that differs
    between the classical method and the two networks.
    """
    res = MethodResult(name=name)
    match_rows, recon_rows, times = [], [], []

    for idx in indices:
        sample = samples[idx]
        t0 = time.perf_counter()
        table = table_fn(sample)
        t_match = time.perf_counter() - t0

        t0 = time.perf_counter()
        asm = assemble(sample.descriptions, table, sample.grid_shape)
        t_assemble = time.perf_counter() - t0

        m = matching_metrics(table, sample)
        nbr = ev.neighbour_accuracy_from_cells(asm, sample.cells)
        pos = ev.position_accuracy_from_cells(asm, sample.cells)
        # orientation ground truth is implied by the labels, not stored --
        # see PuzzleSample.true_rotations
        rots = sample.true_rotations()
        ori = (ev.orientation_accuracy_from_cells(asm, rots, sample.cells)
               if rots else {})
        quality = ev.seam_quality(asm, table)

        row = {
            "puzzle": sample.source,
            "pieces": sample.n_pieces,
            "top1": round(m.get("top1", float("nan")), 4),
            "auc": round(m.get("auc", float("nan")), 4),
            "best_buddy_precision": round(m.get("best_buddy_precision",
                                                float("nan")), 4),
            "neighbour_accuracy": round(nbr["neighbour_accuracy"], 4),
            "position_accuracy": round(pos["position_accuracy"], 4),
            "orientation_accuracy": round(ori.get("orientation_accuracy",
                                                  float("nan")), 4),
            "position_and_orientation_accuracy": round(
                ori.get("position_and_orientation_accuracy", float("nan")), 4),
            "perfect": bool(pos["position_accuracy"] >= 0.999),
            "quality": round(quality["quality"], 4),
            "seconds_match": round(t_match, 3),
            "seconds_assemble": round(t_assemble, 3),
        }
        res.per_puzzle.append(row)
        match_rows.append(m)
        recon_rows.append((nbr["neighbour_accuracy"], pos["position_accuracy"],
                           quality["quality"],
                           ori.get("orientation_accuracy", np.nan),
                           ori.get("position_and_orientation_accuracy", np.nan)))
        times.append((t_match, t_assemble))

        if render:
            row["reconstruction"] = render_assembly(sample.descriptions, asm)

    def mean(key):
        vals = [r[key] for r in match_rows if key in r
                and np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    arr = np.array(recon_rows) if recon_rows else np.zeros((0, 5))
    t = np.array(times) if times else np.zeros((0, 2))
    res.matching = {
        "top1": mean("top1"), "auc": mean("auc"),
        "median_rank": mean("median_rank"),
        "best_buddy_precision": mean("best_buddy_precision"),
    }
    res.reconstruction = {
        "neighbour_accuracy": float(arr[:, 0].mean()) if len(arr) else float("nan"),
        "position_accuracy": float(arr[:, 1].mean()) if len(arr) else float("nan"),
        "orientation_accuracy": float(np.nanmean(arr[:, 3])) if len(arr) else float("nan"),
        "position_and_orientation_accuracy": (
            float(np.nanmean(arr[:, 4])) if len(arr) else float("nan")),
        "quality": float(arr[:, 2].mean()) if len(arr) else float("nan"),
        "complete_reconstructions": int(sum(1 for r in res.per_puzzle
                                            if r["perfect"])),
        "n_puzzles": len(res.per_puzzle),
    }
    res.cost = {
        "seconds_match_per_puzzle": float(t[:, 0].mean()) if len(t) else 0.0,
        "seconds_assemble_per_puzzle": float(t[:, 1].mean()) if len(t) else 0.0,
    }
    if model is not None and hasattr(model, "n_parameters"):
        res.cost["parameters"] = int(model.n_parameters())
        res.cost["model_size_mb"] = round(model.size_bytes() / 1e6, 3)
    else:
        res.cost["parameters"] = 0
        res.cost["model_size_mb"] = 0.0
    return res


def compare_methods(samples, test_indices, siamese=None, gnn=None,
                    verbose: bool = True) -> dict:
    """Run classical / Siamese / GNN over the same test puzzles."""
    methods = [("classical", lambda s: build_compatibility(s.descriptions), None)]
    if siamese is not None:
        methods.append(("siamese_cnn",
                        lambda s: siamese_table(siamese, s.descriptions),
                        siamese))
    if gnn is not None:
        methods.append(("graph_nn",
                        lambda s: gnn_table(gnn, s.descriptions, s.table),
                        gnn))

    out = {}
    for name, fn, model in methods:
        if verbose:
            print(f"  evaluating {name} on {len(test_indices)} test puzzles",
                  flush=True)
        r = evaluate_method(name, samples, test_indices, fn, model=model)
        out[name] = r.as_dict()
        if verbose:
            print(f"    top1 {r.matching['top1']:.3f}  auc {r.matching['auc']:.3f}"
                  f"  neighbour {r.reconstruction['neighbour_accuracy']:.3f}"
                  f"  position {r.reconstruction['position_accuracy']:.3f}"
                  f"  orientation {r.reconstruction['orientation_accuracy']:.3f}"
                  f"  perfect {r.reconstruction['complete_reconstructions']}"
                  f"/{r.reconstruction['n_puzzles']}", flush=True)
    return out
